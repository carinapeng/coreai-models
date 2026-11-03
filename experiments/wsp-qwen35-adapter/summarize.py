"""Combine scored runs into Markdown tables for RESULTS.md.

usage: python summarize.py   (reads results/gen_{main,sweep,cap32k}.scored.jsonl when present)
"""

import json
import math
from collections import defaultdict
from pathlib import Path

EXP = Path(__file__).parent
METRICS = [
    ("loop_tail_rep_gt_0.5", "tail rep > 0.5"),
    ("hit_cap", "hit cap"),
    ("closed_think", "closed `</think>`"),
    ("boxed_after_think", "`\\boxed` after `</think>`"),
    ("boxed_tail2000", "`\\boxed` in last 2000 chars"),
    ("correct", "accuracy (strict)"),
    ("correct_lenient", "accuracy (lenient)"),
]


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def load(name):
    p = EXP / f"results/{name}.scored.jsonl"
    return [json.loads(line) for line in open(p)] if p.exists() else []


def table(recs, title):
    by = defaultdict(dict)
    for r in recs:
        by[r["a"]][(r["idx"], r["sample"])] = r
    if not by:
        return f"### {title}\n\n(no data yet)\n"
    a_vals = sorted(by)
    base = by.get(0.0, {})
    out = [
        f"### {title}",
        "",
        "| metric | " + " | ".join(f"a={a:g} (n={len(by[a])})" for a in a_vals) + " |",
        "|---|" + "---|" * len(a_vals),
    ]
    for key, label in METRICS + [("gen_tokens", "mean generated tokens")]:
        cells = []
        for a in a_vals:
            xs = list(by[a].values())
            if key == "gen_tokens":
                cells.append(f"{sum(x[key] for x in xs) / len(xs):.0f}")
                continue
            k = sum(bool(x[key]) for x in xs)
            cell = f"{k}/{len(xs)}"
            if a != 0.0 and base:
                common = [p for p in by[a] if p in base]
                b = sum(bool(base[p][key]) and not bool(by[a][p][key]) for p in common)
                c = sum(bool(by[a][p][key]) and not bool(base[p][key]) for p in common)
                cell += f" (p={mcnemar_p(b, c):.2g})"
            cells.append(cell)
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    out.append("")
    out.append(
        "p = exact two-sided McNemar test vs a=0 on the same problems and seeds."
    )
    return "\n".join(out) + "\n"


main = load("gen_main")
sweep = load("gen_sweep")
cap32k = load("gen_cap32k")
md = [
    table(main, "Cap 8192: base (a=0) vs adapter (a=6)"),
    table(main + sweep, "Strength curve, cap 8192 (a=0 and a=6 from the main run)"),
    table(cap32k, "Cap 32768: base (a=0) vs adapter (a=6)"),
]
text = "\n".join(md)
(EXP / "results/summary_tables.md").write_text(text)
print(text)
