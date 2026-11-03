"""prefill 1 token, then main one token at a time; print each position as it compiles."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from coreai_llm import EXP, CoreAIQwen35  # noqa: E402

m = CoreAIQwen35(EXP / sys.argv[1] / "model.aimodel", adapter="adapter" in sys.argv[1])
m.reset()
m.prefill([1000])
for i in range(int(sys.argv[2])):
    t = time.perf_counter()
    m.forward([1001 + i])
    print(f"seq_len={m.pos} ok {time.perf_counter() - t:.2f}s", flush=True)
