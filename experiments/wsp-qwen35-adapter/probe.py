"""Quick per-call latency probe: one prefill and a few decode steps, printed as they happen."""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from coreai_llm import EXP, CoreAIQwen35  # noqa: E402

export = sys.argv[1] if len(sys.argv) > 1 else "exports/qwen35_4bit_adapter"
compute = sys.argv[2] if len(sys.argv) > 2 else "default"
adapter = "adapter" in export
t0 = time.perf_counter()
m = CoreAIQwen35(EXP / export / "model.aimodel", adapter=adapter, compute=compute)
print(
    f"runtime={'bundled' if os.environ.get('USE_LOCAL_COREAI') else 'os'} compute={compute} load {time.perf_counter() - t0:.1f}s",
    flush=True,
)
if adapter:
    m.set_adapter(1.0)
for n in (8, 64):
    m.reset()
    t = time.perf_counter()
    m.prefill(list(range(1000, 1000 + n)))
    print(f"prefill {n}: {time.perf_counter() - t:.3f}s", flush=True)
tok = 1234
for i in range(12):
    t = time.perf_counter()
    tok = int(m.forward([tok])[-1].argmax())
    print(f"decode step {i}: {1e3 * (time.perf_counter() - t):.1f} ms", flush=True)
m.close()
