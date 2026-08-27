"""End-to-end DiffusionGemma block diffusion through the Core AI runtime.

Runs the exported encoder + decoder .aimodel components through the real
block-diffusion loop (entropy-bound accept/renoise, temperature schedule,
self-conditioning, stable+confident stop), proving the exported models compose
correctly for the diffusion path — the same sequence the Swift llm-runner would
drive, exercised here via coreai.runtime.

Contracts:
  encoder.aimodel : inputs (input_ids, position_ids), state (keyCache, valueCache)
                    -> logits. Cache state shape [L,1,8,TRACE=2048,512] fp16.
  decoder.aimodel : inputs (decoder_input_ids, prev_soft_embeds, position_ids,
                    encoder_k, encoder_v, temperature) -> (logits, soft_embeds).
                    encoder_k/v shape [L,1,8,ENC_LEN,512] fp16 (fixed at export).

Usage:
  uv run python python/run_diffusion_gemma_coreai.py \
      --encoder /tmp/dg_full/encf16/encoder.aimodel \
      --decoder /tmp/dg_dec/decoder.aimodel \
      --prompt "What is the capital of France?"
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import numpy as np

HF = "google/diffusiongemma-26b-a4b-it"
NL, NKV, HD, STATE_CTX = 30, 8, 512, 2048
T_MAX, T_MIN = 0.8, 0.4
ENTROPY_BOUND = 0.1
CONF_THRESHOLD = 0.005


def chat_format(p: str) -> str:
    return f"<|turn>user\n{p}<turn|>\n<|turn>model\n"


def _load_fn(model):
    names = model.function_names() if callable(model.function_names) else model.function_names
    return model.load_function(names[0])


def _entropy(logits: np.ndarray) -> np.ndarray:
    """Per-position entropy of softmax(logits) over the vocab. logits: [canvas, vocab]."""
    m = logits.max(axis=-1, keepdims=True)
    e = np.exp(logits - m)
    p = e / e.sum(axis=-1, keepdims=True)
    logp = np.log(np.clip(p, 1e-30, None))
    return -(p * logp).sum(axis=-1)


async def run(encoder_path: str, decoder_path: str, prompt: str, canvas_len: int, max_steps: int):
    import coreai.runtime as R
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer

    rng = np.random.default_rng(0)
    tok = Tokenizer.from_file(
        str(Path(snapshot_download(HF, allow_patterns=["tokenizer.json"])) / "tokenizer.json")
    )
    ids = [2] + tok.encode(chat_format(prompt), add_special_tokens=False).ids
    enc_len = len(ids)
    vocab = tok.get_vocab_size()

    enc = await R.AIModel.load(encoder_path)
    dec = await R.AIModel.load(decoder_path)
    enc_fn, dec_fn = _load_fn(enc), _load_fn(dec)

    # 1) Encoder prefill: fill the KV cache state.
    kbf = np.zeros((NL, 1, NKV, STATE_CTX, HD), dtype=np.float16)
    vbf = np.zeros((NL, 1, NKV, STATE_CTX, HD), dtype=np.float16)
    kS, vS = R.NDArray(kbf), R.NDArray(vbf)
    await enc_fn(
        inputs={
            "input_ids": R.NDArray(np.array([ids], dtype=np.int32)),
            "position_ids": R.NDArray(np.arange(enc_len, dtype=np.int32)[None]),
        },
        state={"keyCache": kS, "valueCache": vS},
    )
    # Extract the populated encoder prefix [L,1,8,enc_len,512] for cross-attention.
    enc_k = np.ascontiguousarray(kS.numpy()[:, :, :, :enc_len, :]).astype(np.float16)
    enc_v = np.ascontiguousarray(vS.numpy()[:, :, :, :enc_len, :]).astype(np.float16)

    # 2) Diffusion loop over a single canvas.
    canvas = rng.integers(0, vocab, size=(1, canvas_len)).astype(np.int32)
    soft = np.zeros((1, canvas_len, 2816), dtype=np.float16)
    cpos = np.arange(enc_len, enc_len + canvas_len, dtype=np.int32)[None]
    ek, ev = R.NDArray(enc_k), R.NDArray(enc_v)

    prev_argmax = None
    argmax_canvas = canvas[0]
    for step in reversed(range(1, max_steps + 1)):
        temp = T_MIN + (T_MAX - T_MIN) * (step / max_steps)
        out = await dec_fn(
            inputs={
                "decoder_input_ids": R.NDArray(canvas),
                "prev_soft_embeds": R.NDArray(soft),
                "position_ids": R.NDArray(cpos),
                "encoder_k": ek,
                "encoder_v": ev,
                "temperature": R.NDArray(np.array([temp], dtype=np.float32)),
            },
        )
        processed = np.asarray(out["logits"].numpy(), dtype=np.float32)[0]  # [canvas, vocab]
        soft = np.asarray(out["soft_embeds"].numpy(), dtype=np.float16)

        argmax_canvas = processed.argmax(axis=-1).astype(np.int32)
        probs = np.exp(processed - processed.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        denoiser = np.array(
            [rng.choice(vocab, p=probs[i]) for i in range(canvas_len)], dtype=np.int32
        )

        ent = _entropy(processed)
        order = np.argsort(ent)
        cum = np.cumsum(ent[order]) - ent[order]
        accept_sorted = cum <= ENTROPY_BOUND
        accept = np.zeros(canvas_len, dtype=bool)
        accept[order] = accept_sorted

        new_canvas = np.where(accept, denoiser, canvas[0])
        rand = rng.integers(0, vocab, size=canvas_len).astype(np.int32)
        new_canvas = np.where(accept, new_canvas, rand)
        canvas = new_canvas[None].astype(np.int32)

        mean_h = float(ent.mean())
        preview = tok.decode([int(i) for i in argmax_canvas if i != 0])
        print(
            f"  step {step:2d} temp={temp:.3f} meanH={mean_h:.3f} "
            f"accept={int(accept.sum())}/{canvas_len} | {preview[:80]!r}",
            flush=True,
        )
        stable = prev_argmax is not None and bool(np.array_equal(argmax_canvas, prev_argmax))
        prev_argmax = argmax_canvas
        if stable and mean_h < CONF_THRESHOLD:
            print(f"stop at step {step}", flush=True)
            break

    text = tok.decode([int(i) for i in argmax_canvas if i != 0])
    print(f"\n=== block diffusion through Core AI ===\ntext: {text!r}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", required=True)
    p.add_argument("--decoder", required=True)
    p.add_argument("--prompt", default="What is the capital of France?")
    p.add_argument("--canvas-length", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=16)
    args = p.parse_args()
    asyncio.run(run(args.encoder, args.decoder, args.prompt, args.canvas_length, args.max_steps))


if __name__ == "__main__":
    main()
