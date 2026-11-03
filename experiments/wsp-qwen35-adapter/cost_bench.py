"""Step 2: cost on device. Adapter bytes, set_weight toggle time, decode / prefill
throughput with the adapter on vs off on the adapter export, vs the plain export.

Runs on whichever runtime the environment selects (USE_LOCAL_COREAI=1: bundled runtime).
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from coreai_llm import EXP, CoreAIQwen35, dense_adapter  # noqa: E402

PREFILL_LEN, DECODE_STEPS, REPS = 512, 128, 3
rng = np.random.default_rng(0)
prompt = rng.integers(1000, 100000, PREFILL_LEN).tolist()
res = {
    "runtime": "bundled" if os.environ.get("USE_LOCAL_COREAI") else "os",
    "prefill_len": PREFILL_LEN,
    "decode_steps": DECODE_STEPS,
    "reps": REPS,
}

ad = dense_adapter(1.0)
res["adapter_bytes_fp16_dense"] = int(
    sum(t.numel() * t.element_size() for t in ad.values())
)
res["adapter_nonzero_params"] = int(sum(int((t != 0).sum()) for t in ad.values()))


def bench(m):
    pre, dec = [], []
    for _ in range(REPS):
        m.reset()
        t0 = time.perf_counter()
        m.prefill(prompt)
        pre.append(time.perf_counter() - t0)
        tok = 1234
        t0 = time.perf_counter()
        for _ in range(DECODE_STEPS):
            tok = int(m.forward([tok])[-1].argmax())
        dec.append(time.perf_counter() - t0)
    return {
        "prefill_tok_per_s": PREFILL_LEN / float(np.median(pre)),
        "decode_tok_per_s": DECODE_STEPS / float(np.median(dec)),
        "prefill_s": pre,
        "decode_s": dec,
    }


plain = CoreAIQwen35(EXP / "exports/qwen35_4bit_plain/model.aimodel", adapter=False)
bench(plain)  # warm-up
res["plain"] = bench(plain)
plain.close()

m = CoreAIQwen35(EXP / "exports/qwen35_4bit_adapter/model.aimodel")
m.set_adapter(0.0)
bench(m)  # warm-up
m.set_adapter(0.0)
res["adapter_off"] = bench(m)
m.set_adapter(1.0)
res["adapter_on"] = bench(m)

# Toggle cost: host-side set_weight for all 4 states, and the first decode step after a
# toggle vs steady-state decode steps (any upload of the new state shows up there).
m.reset()
m.prefill(prompt[:64])
tok, steady, after, host = 1234, [], [], []
for i in range(40):
    if i % 4 == 0:
        host.append(m.set_adapter(1.0 if (i // 4) % 2 else 0.0))
        t0 = time.perf_counter()
        tok = int(m.forward([tok])[-1].argmax())
        after.append(time.perf_counter() - t0)
    else:
        t0 = time.perf_counter()
        tok = int(m.forward([tok])[-1].argmax())
        steady.append(time.perf_counter() - t0)
res["toggle"] = {
    "set_adapter_host_ms_median": 1e3 * float(np.median(host)),
    "set_adapter_host_ms_max": 1e3 * float(np.max(host)),
    "decode_step_after_toggle_ms_median": 1e3 * float(np.median(after)),
    "decode_step_steady_ms_median": 1e3 * float(np.median(steady)),
    "n_toggles": len(host),
}
m.close()
for k in ("plain", "adapter_off", "adapter_on"):
    print(
        f"{k:12s} prefill {res[k]['prefill_tok_per_s']:8.1f} tok/s   decode {res[k]['decode_tok_per_s']:7.1f} tok/s"
    )
print(json.dumps(res["toggle"], indent=1))
print(
    "adapter bytes (dense fp16):",
    res["adapter_bytes_fp16_dense"],
    " nonzero params:",
    res["adapter_nonzero_params"],
)
tag = res["runtime"]
(EXP / f"results/cost_{tag}.json").write_text(json.dumps(res, indent=1))
