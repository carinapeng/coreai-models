# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""PyTorch parity / sanity validation for the DiffusionGemma port.

Two modes:

  --mode ar         Autoregressive greedy generation using the shared transformer
                    with a causal mask + KV cache (the "encoder" model). This is a
                    deterministic parity check of the transformer arithmetic.

  --mode diffusion  The full block-diffusion loop: encoder prefill builds the KV
                    cache; the bidirectional decoder denoises a random-init canvas
                    using self-conditioning, an entropy-bound accept/renoise
                    sampler, a temperature schedule, and a stable+confident stop.

Ground truth (prompt "What is the capital of France?"):
    "The capital of France is **Paris**."
    ids include [818, 5279, 529, 7001, 563, 5213, 50429, 84750]

Usage:
    uv run python python/validate_diffusion_gemma.py --mode ar \
        --prompt "What is the capital of France?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from coreai_models.models.macos.diffusion_gemma import (  # noqa: E402
    DiffusionGemmaDecoderForCoreAI,
    _mutate_diffusion_gemma_state_dict,
    load_diffusion_gemma_encoder,
)
from coreai_models.models.macos.diffusion_gemma_config import (  # noqa: E402
    DiffusionGemmaConfig,
    DiffusionGemmaGenerationConfig,
)

HF = "google/diffusiongemma-26b-a4b-it"
EOS_TOKENS = {1, 106}
GROUND_TRUTH_IDS = [818, 5279, 529, 7001, 563, 5213, 50429, 84750]


def chat_format(user_prompt: str) -> str:
    return f"<|turn>user\n{user_prompt}<turn|>\n<|turn>model\n"


def _tokenizer():
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer

    snap = Path(snapshot_download(HF, allow_patterns=["tokenizer.json"]))
    return Tokenizer.from_file(str(snap / "tokenizer.json"))


def _encode_prompt(tok, prompt: str) -> list[int]:
    return [2] + tok.encode(chat_format(prompt), add_special_tokens=False).ids


def run_ar(prompt: str, max_tokens: int, num_layers: int | None, dtype: torch.dtype) -> None:
    tok = _tokenizer()
    ids = _encode_prompt(tok, prompt)
    print(f"prompt ids ({len(ids)}): {ids}", flush=True)

    max_ctx = len(ids) + max_tokens + 4
    print(f"loading encoder (dtype={dtype}, layers={num_layers}, ctx={max_ctx})...", flush=True)
    enc = load_diffusion_gemma_encoder(
        HF, target_dtype=dtype, max_context_length=max_ctx, num_layers=num_layers
    )
    nl = enc.num_hidden_layers
    n_kv = enc.num_key_value_heads
    hd = enc.head_dim

    kc = torch.zeros(nl, 1, n_kv, max_ctx, hd, dtype=dtype)
    vc = torch.zeros_like(kc)

    gen = list(ids)
    out_ids: list[int] = []
    with torch.no_grad():
        # Prefill.
        t = torch.tensor([gen], dtype=torch.int32)
        pos = torch.arange(len(gen), dtype=torch.int32).unsqueeze(0)
        logits = enc(t, pos, kc, vc)
        nxt = int(logits[0, -1].argmax())
        gen.append(nxt)
        out_ids.append(nxt)
        # Incremental decode against the (mutated) cache.
        for _ in range(max_tokens - 1):
            if nxt in EOS_TOKENS or len(gen) >= max_ctx:
                break
            t = torch.tensor([[nxt]], dtype=torch.int32)
            pos = torch.arange(len(gen), dtype=torch.int32).unsqueeze(0)
            logits = enc(t, pos, kc, vc)
            nxt = int(logits[0, -1].argmax())
            gen.append(nxt)
            out_ids.append(nxt)

    text = tok.decode(out_ids)
    print(f"\n=== AR greedy ===\ngenerated ids: {out_ids}\ntext: {text!r}", flush=True)


def _load_shared_encoder_decoder(num_layers: int | None, dtype: torch.dtype):
    """Load the decoder (has self-conditioning + shared transformer) and build an
    encoder that SHARES the same underlying transformer weight tensors to avoid a
    second ~50GB copy. The two differ only in SDPA mode and which per-layer scalar
    is applied (both scalars are present in every layer)."""
    from coreai_models.models.macos.diffusion_gemma import (
        DiffusionGemmaEncoderForCoreAI,
        _load_state_dict_from_hub,
    )

    cfg = DiffusionGemmaConfig.from_pretrained(HF).text_config
    if num_layers is not None:
        cfg.num_hidden_layers = num_layers

    dec = DiffusionGemmaDecoderForCoreAI(cfg)
    dec.to(dtype=dtype)
    sd = _load_state_dict_from_hub(HF, dtype, num_layers)
    _mutate_diffusion_gemma_state_dict(sd, dec)
    dec.load_state_dict(sd, strict=False, assign=True)
    if cfg.tie_word_embeddings:
        dec.lm_head.weight = dec.model.embed_tokens.weight
    dec.eval()

    enc = DiffusionGemmaEncoderForCoreAI(cfg)
    # Share every parameter tensor from the decoder's shared transformer.
    enc_sd = {k: v for k, v in dec.state_dict().items() if not k.startswith("self_conditioning.")}
    enc.load_state_dict(enc_sd, strict=False, assign=True)
    if cfg.tie_word_embeddings:
        enc.lm_head.weight = enc.model.embed_tokens.weight
    enc.eval()
    return enc, dec, cfg


def _token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Per-position entropy of the categorical distribution over the vocab."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    p = logp.exp()
    return -(p * logp).sum(dim=-1)


def run_diffusion(
    prompt: str, num_layers: int | None, dtype: torch.dtype, gen_cfg: DiffusionGemmaGenerationConfig
) -> None:
    torch.manual_seed(0)
    tok = _tokenizer()
    ids = _encode_prompt(tok, prompt)
    enc_len = len(ids)
    print(f"prompt ids ({enc_len}): {ids}", flush=True)

    canvas_len = gen_cfg.canvas_length
    max_steps = gen_cfg.max_denoising_steps
    print(f"loading shared encoder+decoder (canvas={canvas_len}, steps={max_steps})...", flush=True)
    enc, dec, cfg = _load_shared_encoder_decoder(num_layers, dtype)

    nl = cfg.num_hidden_layers
    n_kv = cfg.cache_num_key_value_heads
    hd = cfg.cache_head_dim
    vocab = cfg.vocab_size
    H = cfg.hidden_size

    with torch.no_grad():
        # 1) Encoder prefill -> fill the unified KV cache.
        kc = torch.zeros(nl, 1, n_kv, enc_len, hd, dtype=dtype)
        vc = torch.zeros_like(kc)
        t = torch.tensor([ids], dtype=torch.int32)
        pos = torch.arange(enc_len, dtype=torch.int32).unsqueeze(0)
        enc(t, pos, kc, vc)  # mutates kc/vc in place

        # 2) Canvas init = random tokens; soft-cond starts at zeros.
        canvas = torch.randint(0, vocab, (1, canvas_len), dtype=torch.int32)
        soft = torch.zeros(1, canvas_len, H, dtype=dtype)
        # Canvas positions CONTINUE after the encoder prefix (reference:
        # decoder_position_ids = arange(cache_seq_length, cache_seq_length + canvas_length)).
        cpos = torch.arange(enc_len, enc_len + canvas_len, dtype=torch.int32).unsqueeze(0)

        prev_argmax = None
        for step in reversed(range(1, max_steps + 1)):
            temp = gen_cfg.t_min + (gen_cfg.t_max - gen_cfg.t_min) * (step / max_steps)
            temp_t = torch.tensor([temp], dtype=torch.float32)
            processed, soft = dec(canvas, soft, cpos, kc, vc, temp_t)
            processed = processed[0].float()  # [canvas, vocab]

            argmax_canvas = processed.argmax(dim=-1)
            denoiser = torch.multinomial(processed.softmax(dim=-1), 1).squeeze(-1)

            # Entropy-bound accept: accept the lowest-entropy positions.
            ent = _token_entropy(processed)  # [canvas]
            order = torch.argsort(ent)
            cum = torch.cumsum(ent[order], dim=0) - ent[order]
            accept_sorted = cum <= gen_cfg.entropy_bound
            accept = torch.zeros(canvas_len, dtype=torch.bool)
            accept[order] = accept_sorted

            new_canvas = torch.where(accept, denoiser, canvas[0])
            # Renoise non-accepted positions with fresh random tokens.
            rand = torch.randint(0, vocab, (canvas_len,), dtype=torch.int32)
            new_canvas = torch.where(accept, new_canvas, rand)
            canvas = new_canvas.unsqueeze(0).to(torch.int32)

            mean_h = ent.mean().item()
            stable = prev_argmax is not None and bool((argmax_canvas == prev_argmax).all())
            confident = mean_h < gen_cfg.confidence_threshold
            prev_argmax = argmax_canvas
            preview = tok.decode([i for i in argmax_canvas.tolist() if i not in (0,)])
            print(
                f"  step {step:2d} temp={temp:.3f} meanH={mean_h:.3f} "
                f"accept={int(accept.sum())}/{canvas_len} | argmax: {preview[:80]!r}",
                flush=True,
            )
            if stable and confident:
                print(f"stop at step {step} (stable+confident, meanH={mean_h:.4f})", flush=True)
                break

        final = argmax_canvas.tolist()
    text = tok.decode([i for i in final if i not in (0,)])
    print(f"\n=== block diffusion ===\ncanvas argmax ids: {final}\ntext: {text!r}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="DiffusionGemma PyTorch validation")
    p.add_argument("--mode", choices=["ar", "diffusion"], default="ar")
    p.add_argument("--prompt", default="What is the capital of France?")
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--num-layers", type=int, default=None, help="Truncate layers (smoke)")
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--canvas-length", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=16)
    args = p.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    if args.mode == "ar":
        run_ar(args.prompt, args.max_tokens, args.num_layers, dtype)
    else:
        gen_cfg = DiffusionGemmaGenerationConfig(
            canvas_length=args.canvas_length, max_denoising_steps=args.max_steps
        )
        run_diffusion(args.prompt, args.num_layers, dtype, gen_cfg)


if __name__ == "__main__":
    main()
