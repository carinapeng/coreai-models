"""A cheap looping metric on generated text: how much of the tail is a repeated cycle."""

import json


def tail_repetition(text: str, tail_chars: int = 4000, n: int = 30) -> float:
    """Share of distinct-position n-char windows in the tail that occur more than once."""
    t = text[-tail_chars:]
    if len(t) < n * 2:
        return 0.0
    grams = [t[i : i + n] for i in range(0, len(t) - n, 10)]
    seen, rep = {}, 0
    for g in grams:
        seen[g] = seen.get(g, 0) + 1
    rep = sum(c for c in seen.values() if c > 1)
    return rep / len(grams)


def has_boxed(text: str) -> bool:
    return "\\boxed" in text[-2000:]


for f in ["samples_base_math500.jsonl", "samples_edited_math500.jsonl"]:
    rows = [json.loads(line) for line in open(f)]
    reps = [tail_repetition(r["text"]) for r in rows]
    lens = [len(r["text"]) for r in rows]
    loops = sum(x > 0.5 for x in reps)
    boxed = sum(has_boxed(r["text"]) for r in rows)
    print(
        f"{f:32s} n={len(rows)}  mean_chars={sum(lens) / len(lens):8.0f}  "
        f"tail_repetition mean={sum(reps) / len(reps):.2f}  looping(>0.5)={loops}/{len(rows)}  final \\boxed={boxed}/{len(rows)}"
    )
