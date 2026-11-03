"""Find where the per-call time goes: state backing (bytes vs metal) and shape reuse."""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from coreai_llm import EXP, CoreAIQwen35  # noqa: E402
from coreai.runtime import NDArray, StorageKind  # noqa: E402

export = sys.argv[1]
m = CoreAIQwen35(
    EXP / export / "model.aimodel",
    adapter="adapter" in export,
    compute=(sys.argv[2] if len(sys.argv) > 2 else "default"),
)


def call(n_pos, backing_states):
    inputs = {
        "input_ids": NDArray(data=torch.tensor([[1234]], dtype=torch.int32)),
        "position_ids": NDArray(
            data=torch.arange(n_pos, dtype=torch.int32).unsqueeze(0).contiguous()
        ),
    }
    t = time.perf_counter()
    m.loop.run_until_complete(m.fn_main(inputs, state=backing_states))
    return 1e3 * (time.perf_counter() - t)


for base, kind in (
    (20, StorageKind.METAL),
    (40, StorageKind.BYTES),
    (60, StorageKind.METAL),
):
    m.reset()
    st = {}
    for k, v in {**m.adapter_states, **m.cache_states}.items():
        st[k] = NDArray(data=torch.from_numpy(v.numpy().copy()), backing=kind)
    same = [call(base, st) for _ in range(4)]
    grow = [call(base + 1 + i, st) for i in range(8)]
    print(
        f"{kind.value:6s} same-shape ms {[round(x) for x in same]}  growing-shape ms {[round(x) for x in grow]}",
        flush=True,
    )
m.close()
