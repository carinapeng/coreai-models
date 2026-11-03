"""Minimal Python driver for the Qwen3.5 Core AI export (main + prefill entrypoints).

Holds every runtime state on the host side of the call: the model's keyCache / valueCache /
convStates / recurrentStates, plus the adapter's mutable weights (l15_u, l15_c, l19_u,
l19_c) when the export has them. States are passed on every call and updated in place by
the runtime; set_adapter replaces only the adapter states.
"""

from __future__ import annotations

import asyncio
import glob
import os
import time
from contextlib import AsyncExitStack
from pathlib import Path

import numpy as np
import torch

H = os.path.expanduser("~/.cache/huggingface/hub")
BASE_DIR = glob.glob(f"{H}/models--Qwen--Qwen3.5-0.8B/snapshots/*")[0]
EDIT_DIR = glob.glob(
    f"{H}/models--RikiyaT--Qwen3.5-0.8B-weight-space-programming/snapshots/*"
)[0]
EXP = Path(__file__).resolve().parent
ADAPTER_LAYERS = (15, 19)


_EDIT = None


def _edit():
    global _EDIT
    if _EDIT is None:
        _EDIT = torch.load(EXP / "rank1_edit.pt")
    return _EDIT


def dense_adapter(scale: float = 1.0, dtype=torch.float16) -> dict[str, torch.Tensor]:
    """The published edit as dense rank-1 vectors {l{L}_u, l{L}_c}, u scaled by ``scale``."""
    edit = _edit()
    out = {}
    for L in ADAPTER_LAYERS:
        e = edit[L]
        c = torch.zeros(3584)
        c[e["cols"]] = e["coef"]
        out[f"l{L}_u"] = (scale * e["u"]).to(dtype).contiguous()
        out[f"l{L}_c"] = c.to(dtype).contiguous()
    return out


def text_config(max_ctx: int):
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(BASE_DIR).text_config
    cfg.max_position_embeddings = max_ctx
    return cfg


class CoreAIQwen35:
    def __init__(
        self,
        aimodel: Path | str,
        max_ctx: int = 9216,
        adapter: bool = True,
        compute: str = "default",
    ) -> None:
        from coreai_models.models.macos.qwen3_5 import Qwen3_5ForCausalLM

        self.path = Path(aimodel)
        self.max_ctx = max_ctx
        self.cfg = text_config(max_ctx)
        self._cls = Qwen3_5ForCausalLM
        self.has_adapter = adapter
        # The 2-D variant declares u as (1, hidden) and c as (intermediate, 1).
        self.adapter_2d = "adapter2d" in str(self.path)
        self.compute = compute
        self.loop = asyncio.new_event_loop()
        self._stack = AsyncExitStack()
        t0 = time.perf_counter()
        self.loop.run_until_complete(self._load())
        self.load_seconds = time.perf_counter() - t0
        self.adapter_states: dict = {}
        if adapter:
            self.set_adapter(0.0)
        self.reset()

    async def _load(self) -> None:
        from coreai.authoring.asset import AIModelAsset
        from coreai.runtime import ComputeUnitKind, SpecializationOptions

        asset = AIModelAsset.load(self.path)
        # "default": no specialization options, the runtime's own choice.
        opts = (
            None
            if self.compute == "default"
            else (
                SpecializationOptions.from_preferred_compute_unit_kind(
                    getattr(ComputeUnitKind, self.compute)()
                )
            )
        )
        model = await self._stack.enter_async_context(asset.executable(opts))
        self.fn_main = model.load_function("main")
        self.fn_prefill = model.load_function("prefill")

    def reset(self) -> None:
        """Fresh zero KV/conv/recurrent states (the adapter states are left as they are)."""
        from coreai.runtime import NDArray

        k, v = self._cls.create_kv_cache_tensors(
            self.cfg, dtype=torch.float16, seq_len=self.max_ctx
        )
        conv, rec = self._cls.create_delta_cache_tensors(self.cfg, dtype=torch.float16)
        self.cache_states = {
            "keyCache": NDArray(data=k),
            "valueCache": NDArray(data=v),
            "convStates": NDArray(data=conv),
            "recurrentStates": NDArray(data=rec),
        }
        self.pos = 0

    def set_weight(self, name: str, value: torch.Tensor) -> None:
        from coreai.runtime import NDArray

        if name not in ("l15_u", "l15_c", "l19_u", "l19_c"):
            raise KeyError(name)
        if self.adapter_2d:
            value = (
                value.reshape(1, -1) if name.endswith("_u") else value.reshape(-1, 1)
            )
        self.adapter_states[name] = NDArray(data=value.to(torch.float16).contiguous())

    def set_adapter(self, scale: float) -> float:
        """Set the adapter to ``scale`` x the published edit; returns host-side seconds."""
        t0 = time.perf_counter()
        for name, value in dense_adapter(scale).items():
            self.set_weight(name, value)
        return time.perf_counter() - t0

    def _call(self, fn, ids: list[int]):
        from coreai.runtime import NDArray

        n = len(ids)
        inputs = {
            "input_ids": NDArray(data=torch.tensor([ids], dtype=torch.int32)),
            "position_ids": NDArray(
                data=torch.arange(self.pos + n, dtype=torch.int32)
                .unsqueeze(0)
                .contiguous()
            ),
        }
        state = {**self.adapter_states, **self.cache_states}
        out = self.loop.run_until_complete(fn(inputs, state=state))
        self.pos += n
        return out

    def forward(self, ids: list[int]) -> np.ndarray:
        """Run ``main`` on ``ids`` after the current position; returns (len(ids), vocab) float32."""
        out = self._call(self.fn_main, ids)
        return np.asarray(out["logits"].numpy())[0].astype(np.float32)

    def prefill(self, ids: list[int]) -> None:
        """Run ``prefill`` (no LM head) on ``ids`` after the current position."""
        self._call(self.fn_prefill, ids)

    def close(self) -> None:
        self.loop.run_until_complete(self._stack.aclose())
        self.loop.close()
