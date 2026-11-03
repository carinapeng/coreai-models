"""Which call shapes abort during specialization? main/prefill at given query lengths."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from coreai_llm import EXP, CoreAIQwen35  # noqa: E402

export, fn = sys.argv[1], sys.argv[2]
m = CoreAIQwen35(EXP / export / "model.aimodel", adapter="adapter" in export)
for q in [int(x) for x in sys.argv[3:]]:
    m.reset()
    t = time.perf_counter()
    (m.forward if fn == "main" else m.prefill)(list(range(1000, 1000 + q)))
    print(f"{fn} q={q} ok {time.perf_counter() - t:.2f}s", flush=True)
