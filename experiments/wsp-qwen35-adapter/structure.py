import glob
import os
import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))


H = os.path.expanduser("~/.cache/huggingface/hub")
bf = glob.glob(f"{H}/models--Qwen--Qwen3.5-0.8B/snapshots/*/*.safetensors")[0]
ef = glob.glob(
    f"{H}/models--RikiyaT--Qwen3.5-0.8B-weight-space-programming/snapshots/*/*.safetensors"
)[0]
us, out = {}, {}
with safe_open(bf, "pt") as b, safe_open(ef, "pt") as e:
    for L in (15, 19):
        k = f"model.language_model.layers.{L}.mlp.down_proj.weight"
        W, We = b.get_tensor(k).float(), e.get_tensor(k).float()
        D = We - W
        cols = torch.nonzero(D.abs().sum(0) > 0).flatten()
        U, S, Vh = torch.linalg.svd(D[:, cols], full_matrices=False)
        u = U[:, 0]
        us[L] = u
        # Is it projection removal along u?  W'[:,c] = W[:,c] - a * u (u^T W[:,c])
        proj = -u[:, None] * (u[None, :] @ W[:, cols])
        a = (D[:, cols] * proj).sum() / (proj * proj).sum()
        resid = (D[:, cols] - a * proj).norm() / D[:, cols].norm()
        # How much of the u-component of those columns remains after the edit?
        before = (u @ W[:, cols]).abs().mean()
        after = (u @ We[:, cols]).abs().mean()
        print(
            f"layer {L}: cols={len(cols)} first={cols[:6].tolist()}  fit D = a * (-u u^T W_cols): a={a:.3f}, "
            f"relative residual={resid:.3%}  |u.W_cols| before={before:.4f} after={after:.4f}"
        )
        out[L] = (cols, u, (D[:, cols].T @ u))  # u, and per-column coefficients
print(f"cosine(u15, u19) = {torch.dot(us[15], us[19]).item():+.3f}")
c15, c19 = set(out[15][0].tolist()), set(out[19][0].tolist())
print(f"shared columns between layers: {len(c15 & c19)} of 64")
torch.save(
    {L: {"cols": out[L][0], "u": out[L][1], "coef": out[L][2]} for L in out},
    HERE + "/rank1_edit.pt",
)
print(
    "rank-1 edit saved:",
    os.path.getsize(HERE + "/rank1_edit.pt"),
    "bytes (fp32, incl. container overhead)",
)
