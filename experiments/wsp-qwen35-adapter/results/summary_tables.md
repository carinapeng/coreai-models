### Cap 8192: base (a=0) vs adapter (a=6)

| metric | a=0 (n=50) | a=6 (n=50) |
|---|---|---|
| tail rep > 0.5 | 48/50 | 14/50 (p=1.2e-10) |
| hit cap | 49/50 | 44/50 (p=0.062) |
| closed `</think>` | 1/50 | 21/50 (p=1.9e-06) |
| `\boxed` after `</think>` | 1/50 | 7/50 (p=0.031) |
| `\boxed` in last 2000 chars | 1/50 | 8/50 (p=0.016) |
| accuracy (strict) | 1/50 | 3/50 (p=0.5) |
| accuracy (lenient) | 1/50 | 3/50 (p=0.5) |
| mean generated tokens | 8087 | 7816 |

p = exact two-sided McNemar test vs a=0 on the same problems and seeds.

### Strength curve, cap 8192 (a=0 and a=6 from the main run)

| metric | a=0 (n=50) | a=1 (n=50) | a=3 (n=50) | a=6 (n=50) | a=9 (n=50) |
|---|---|---|---|---|---|
| tail rep > 0.5 | 48/50 | 46/50 (p=0.69) | 26/50 (p=3e-06) | 14/50 (p=1.2e-10) | 4/50 (p=1.1e-13) |
| hit cap | 49/50 | 50/50 (p=1) | 46/50 (p=0.38) | 44/50 (p=0.062) | 46/50 (p=0.25) |
| closed `</think>` | 1/50 | 0/50 (p=1) | 12/50 (p=0.00098) | 21/50 (p=1.9e-06) | 20/50 (p=3.8e-06) |
| `\boxed` after `</think>` | 1/50 | 0/50 (p=1) | 5/50 (p=0.22) | 7/50 (p=0.031) | 7/50 (p=0.031) |
| `\boxed` in last 2000 chars | 1/50 | 0/50 (p=1) | 6/50 (p=0.12) | 8/50 (p=0.016) | 8/50 (p=0.016) |
| accuracy (strict) | 1/50 | 0/50 (p=1) | 4/50 (p=0.38) | 3/50 (p=0.5) | 2/50 (p=1) |
| accuracy (lenient) | 1/50 | 0/50 (p=1) | 4/50 (p=0.38) | 3/50 (p=0.5) | 2/50 (p=1) |
| mean generated tokens | 8087 | 8192 | 7963 | 7816 | 8050 |

p = exact two-sided McNemar test vs a=0 on the same problems and seeds.

### Cap 32768: base (a=0) vs adapter (a=6)

(no data yet)
