"""main(q tokens) after prefill(p tokens): which (p, q) compile on the OS framework."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from coreai_llm import EXP, CoreAIQwen35  # noqa: E402

m = CoreAIQwen35(EXP / sys.argv[1] / "model.aimodel", adapter="adapter" in sys.argv[1])
for pq in sys.argv[2:]:
    p, q = map(int, pq.split(":"))
    m.reset()
    if p:
        m.prefill(list(range(2000, 2000 + p)))
    t = time.perf_counter()
    m.forward(list(range(1000, 1000 + q)))
    print(f"prefill {p} then main q={q} ok {time.perf_counter() - t:.2f}s", flush=True)
