"""Export Qwen3.5-0.8B (macOS recipe) to Core AI, optionally with the rank-1 edit as a
mutable-weight adapter on layers 15 and 19 mlp.down_proj.

Local experiment wrapper: patches the stock recipe after loading; the public model files
are untouched. The adapter vectors u (hidden) and c (intermediate, dense) are
MutableWeights read with .read(), so they are runtime states the host sets between calls,
next to the model's own k_cache / v_cache / conv_states / recurrent_states.

usage: uv run python coreai_export.py --out exports/qwen35_adapter [--no-adapter]
"""

import argparse
import json
import logging
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn

from coreai_models._constants import MAIN_GRAPH_NAME
from coreai_models.export.macos import _build_reference_inputs, export_to_coreai
from coreai_models.export.metadata import build_aimodel_metadata
from coreai_models.export.pipeline import ExportConfig, build_model, resolve_model
from coreai_models.primitives.mutable_weight import MutableWeight

ADAPTER_LAYERS = (15, 19)
HF_ID = "Qwen/Qwen3.5-0.8B"


class Rank1MutableAdapter2D(nn.Module):
    """Same math as Rank1MutableAdapter, as two matmuls on 2-D states: (h @ c) @ u,
    c: (intermediate, 1), u: (1, hidden)."""

    def __init__(self, inner: nn.Linear, prefix: str, dtype: torch.dtype) -> None:
        super().__init__()
        self.inner = inner
        self.u = MutableWeight(f"{prefix}_u", (1, inner.out_features), dtype)
        self.c = MutableWeight(f"{prefix}_c", (inner.in_features, 1), dtype)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        u = self.u.read()
        c = self.c.read()
        return self.inner(h) + (h @ c) @ u


class Rank1MutableAdapter(nn.Module):
    """inner(h) + u * (c . h), with u and c held on device as mutable weights."""

    def __init__(self, inner: nn.Linear, prefix: str, dtype: torch.dtype) -> None:
        super().__init__()
        self.inner = inner
        self.u = MutableWeight(f"{prefix}_u", (inner.out_features,), dtype)
        self.c = MutableWeight(f"{prefix}_c", (inner.in_features,), dtype)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        u = self.u.read()
        c = self.c.read()
        return self.inner(h) + (h @ c).unsqueeze(-1) * u


def adapter_weight_names() -> tuple[str, ...]:
    return tuple(f"l{L}_{v}" for L in ADAPTER_LAYERS for v in ("u", "c"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-adapter", action="store_true")
    ap.add_argument("--compression", default="none")
    ap.add_argument("--max-context-length", type=int, default=9216)
    ap.add_argument("--adapter-2d", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    cfg = ExportConfig(
        hf_model_id=HF_ID,
        compression=args.compression,
        compute_precision="float16",
        max_context_length=args.max_context_length,
    )
    t0 = time.perf_counter()
    resolved = resolve_model(cfg)
    with tempfile.TemporaryDirectory(prefix="wsp_export_") as tmp:
        built = build_model(resolved, tmp)
        model = built.model
        assert built.externalized_model is None, (
            "graph-mode compression not handled here"
        )
        t_load = time.perf_counter() - t0

        names: tuple[str, ...] = ()
        if not args.no_adapter:
            for L in ADAPTER_LAYERS:
                mlp = model.model.layers[L].mlp
                cls = Rank1MutableAdapter2D if args.adapter_2d else Rank1MutableAdapter
                mlp.down_proj = cls(mlp.down_proj, f"l{L}", resolved.target_dtype)
            names = adapter_weight_names()

        ref, dyn = _build_reference_inputs(
            model,
            resolved.hf_config,
            resolved.target_dtype,
            resolved.max_context_length,
        )
        state_names = (*names, *model.export_state_names()[MAIN_GRAPH_NAME])
        print("state_names:", state_names)
        t1 = time.perf_counter()
        program = export_to_coreai(
            model,
            ref,
            dynamic_shapes=dyn,
            input_names=model.export_input_names()[MAIN_GRAPH_NAME],
            output_names=model.export_output_names()[MAIN_GRAPH_NAME],
            state_names=state_names,
            export_prefill_graph=model.exports_prefill_graph,
        )
        t_convert = time.perf_counter() - t1

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        aimodel = out / "model.aimodel"
        t2 = time.perf_counter()
        program.save_asset(aimodel, build_aimodel_metadata(HF_ID))
        t_save = time.perf_counter() - t2

    size = sum(f.stat().st_size for f in aimodel.rglob("*") if f.is_file())
    info = {
        "hf_id": HF_ID,
        "adapter": not args.no_adapter,
        "adapter_2d": args.adapter_2d,
        "adapter_weight_names": list(names),
        "state_names": list(state_names),
        "compression": args.compression,
        "compute_precision": "float16",
        "max_context_length": resolved.max_context_length,
        "seconds_load_and_compress": round(t_load, 1),
        "seconds_convert": round(t_convert, 1),
        "seconds_save": round(t_save, 1),
        "seconds_total": round(time.perf_counter() - t0, 1),
        "aimodel_bytes": size,
    }
    (out / "export_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
