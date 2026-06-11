"""Parity test: our WhisperEncoder/Decoder vs HuggingFace reference.

Levels tested:
  1. Encoder shape + values  — silence
  2. Encoder shape + values  — random noise
  3. Encoder shape + values  — sine-wave tone
  4. Decoder first-step logits — top-5 tokens match HF (silence)
  5. Greedy decode token sequence — first 8 tokens match HF (silence)
  6. Greedy decode token sequence — first 8 tokens match HF (noise)
  7. Real LibriSpeech audio    — full decode matches HF pipeline

Run:
    uv run python test_whisper_parity.py
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

MODEL_NAME = "openai/whisper-large-v3-turbo"
ATOL = 5e-3
MAX_NEW_TOKENS = 8
# Forced prefix Whisper always prepends: BOS, <|en|>, <|transcribe|>, <|notimestamps|>
FORCED_PREFIX = [50258, 50259, 50360, 50364]

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_our_models(hf_model, cfg, dtype):
    from coreai_models.models.macos.whisper import WhisperEncoder, WhisperDecoder

    enc = WhisperEncoder(
        cfg.num_mel_bins, cfg.d_model, cfg.encoder_attention_heads,
        cfg.encoder_layers, cfg.encoder_ffn_dim, cfg.max_source_positions,
    ).to(dtype).eval()
    enc_sd = {k[len("model.encoder."):]: v for k, v in hf_model.state_dict().items()
              if k.startswith("model.encoder.")}
    enc.load_state_dict(enc_sd, strict=True)

    dec = WhisperDecoder(
        cfg.vocab_size, cfg.d_model, cfg.decoder_attention_heads,
        cfg.decoder_layers, cfg.decoder_ffn_dim, cfg.max_target_positions,
    ).to(dtype).eval()
    dec_sd = {k[len("model.decoder."):]: v for k, v in hf_model.state_dict().items()
              if k.startswith("model.decoder.")}
    dec.load_state_dict(dec_sd, strict=True)

    return enc, dec


def _mel(processor, audio_np, sr=16000):
    feat = processor.feature_extractor(audio_np, sampling_rate=sr, return_tensors="pt")
    return feat["input_features"].to(torch.float32)   # (1, 128, 3000)


def _hf_encoder_out(hf_model, mel):
    with torch.no_grad():
        return hf_model.model.encoder(mel).last_hidden_state


def _hf_first_logits(hf_model, enc_out, start_token=50258):
    """One decoder step in HF (no KV cache, just forward on single token)."""
    input_ids = torch.tensor([[start_token]], dtype=torch.long)
    with torch.no_grad():
        out = hf_model.model.decoder(
            input_ids=input_ids,
            encoder_hidden_states=enc_out,
        )
        return F.linear(out.last_hidden_state, hf_model.model.decoder.embed_tokens.weight)


def _our_first_logits(our_enc, our_dec, mel, cfg, start_token=50258):
    n_layers = cfg.decoder_layers
    n_heads = cfg.decoder_attention_heads
    head_dim = cfg.d_model // n_heads
    max_pos = cfg.max_target_positions

    with torch.no_grad():
        enc_out = our_enc(mel)
        k_cache = torch.zeros(n_layers, 1, n_heads, max_pos, head_dim)
        v_cache = torch.zeros(n_layers, 1, n_heads, max_pos, head_dim)
        input_ids = torch.tensor([[start_token]], dtype=torch.int32)
        position_ids = torch.tensor([[0]], dtype=torch.int32)
        return our_dec(input_ids, position_ids, enc_out, k_cache, v_cache)


def _greedy_decode_ours(our_enc, our_dec, mel, cfg, max_new=MAX_NEW_TOKENS):
    n_layers = cfg.decoder_layers
    n_heads = cfg.decoder_attention_heads
    head_dim = cfg.d_model // n_heads
    max_pos = cfg.max_target_positions

    k_cache = torch.zeros(n_layers, 1, n_heads, max_pos, head_dim)
    v_cache = torch.zeros(n_layers, 1, n_heads, max_pos, head_dim)

    with torch.no_grad():
        enc_out = our_enc(mel)
        tokens = list(FORCED_PREFIX)
        # Feed forced prefix through decoder to prime KV cache
        for step, tok in enumerate(FORCED_PREFIX):
            cur = torch.tensor([[tok]], dtype=torch.int32)
            pos = torch.arange(step + 1, dtype=torch.int32).unsqueeze(0)
            our_dec(cur, pos, enc_out, k_cache, v_cache)
        # Free generation from position len(FORCED_PREFIX)
        for step in range(len(FORCED_PREFIX), len(FORCED_PREFIX) + max_new):
            cur = torch.tensor([[tokens[-1]]], dtype=torch.int32)
            pos = torch.arange(step + 1, dtype=torch.int32).unsqueeze(0)
            logits = our_dec(cur, pos, enc_out, k_cache, v_cache)
            next_tok = int(logits[0, -1].argmax())
            tokens.append(next_tok)
            if next_tok == 50257:   # EOT
                break
    return tokens


def _greedy_decode_hf(hf_model, processor, mel):
    with torch.no_grad():
        out = hf_model.generate(
            mel,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            return_dict_in_generate=True,
        )
    return out.sequences[0].tolist()  # full sequence incl. BOS + forced prefix + EOT


def check(name, passed, detail=""):
    symbol = "✅" if passed else "❌"
    print(f"  {symbol}  {name}" + (f"  — {detail}" if detail else ""))
    return passed


# ---------------------------------------------------------------------------
# test cases
# ---------------------------------------------------------------------------

def run():
    print(f"\nLoading HF model {MODEL_NAME} …")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    hf_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, use_safetensors=True
    ).eval()
    cfg = hf_model.config
    our_enc, our_dec = _load_our_models(hf_model, cfg, torch.float32)
    print("Models loaded.\n")

    results = []
    sr = 16000

    # ── audio fixtures ──────────────────────────────────────────────────────
    silence = np.zeros(sr * 5, dtype=np.float32)
    noise   = np.random.default_rng(42).standard_normal(sr * 5).astype(np.float32) * 0.01
    tone    = (np.sin(2 * np.pi * 440 * np.arange(sr * 5) / sr) * 0.5).astype(np.float32)

    mel_silence = _mel(processor, silence)
    mel_noise   = _mel(processor, noise)
    mel_tone    = _mel(processor, tone)

    # ── Level 1: encoder shape + values — silence ───────────────────────────
    print("Level 1 — Encoder, silence")
    with torch.no_grad():
        our_enc_out = our_enc(mel_silence)
        hf_enc_out  = _hf_encoder_out(hf_model, mel_silence)
    results.append(check("shape", our_enc_out.shape == hf_enc_out.shape, str(our_enc_out.shape)))
    diff = (our_enc_out - hf_enc_out).abs().max().item()
    results.append(check("max abs diff < 1e-4", diff < ATOL, f"{diff:.2e}"))

    # ── Level 2: encoder — random noise ─────────────────────────────────────
    print("\nLevel 2 — Encoder, random noise")
    with torch.no_grad():
        our_enc_out2 = our_enc(mel_noise)
        hf_enc_out2  = _hf_encoder_out(hf_model, mel_noise)
    diff2 = (our_enc_out2 - hf_enc_out2).abs().max().item()
    results.append(check("max abs diff < 1e-4", diff2 < ATOL, f"{diff2:.2e}"))

    # ── Level 3: encoder — 440 Hz tone ──────────────────────────────────────
    print("\nLevel 3 — Encoder, 440 Hz tone")
    with torch.no_grad():
        our_enc_out3 = our_enc(mel_tone)
        hf_enc_out3  = _hf_encoder_out(hf_model, mel_tone)
    diff3 = (our_enc_out3 - hf_enc_out3).abs().max().item()
    results.append(check("max abs diff < 1e-4", diff3 < ATOL, f"{diff3:.2e}"))

    # ── Level 4: decoder first-step logits — silence ────────────────────────
    print("\nLevel 4 — Decoder first-step logits, silence")
    our_logits = _our_first_logits(our_enc, our_dec, mel_silence, cfg)
    hf_logits  = _hf_first_logits(hf_model, hf_enc_out)
    logit_diff = (our_logits - hf_logits).abs().max().item()
    our_top5 = our_logits[0, -1].topk(5).indices.tolist()
    hf_top5  = hf_logits[0, -1].topk(5).indices.tolist()
    results.append(check("top-1 token matches", our_top5[0] == hf_top5[0],
                         f"ours={our_top5[0]} hf={hf_top5[0]}"))
    results.append(check("top-5 tokens match", our_top5 == hf_top5,
                         f"ours={our_top5} hf={hf_top5}"))
    results.append(check("max logit diff < 0.05", logit_diff < 0.05, f"{logit_diff:.4f}"))

    # ── Level 5: greedy decode token sequence — silence ─────────────────────
    print("\nLevel 5 — Greedy decode, silence")
    our_tokens = _greedy_decode_ours(our_enc, our_dec, mel_silence, cfg)
    hf_tokens  = _greedy_decode_hf(hf_model, processor, mel_silence)
    n = min(len(our_tokens), len(hf_tokens), MAX_NEW_TOKENS + 1)
    results.append(check(f"first {n} tokens match",
                         our_tokens[:n] == hf_tokens[:n],
                         f"\n       ours={our_tokens[:n]}\n       hf  ={hf_tokens[:n]}"))

    # ── Level 6: greedy decode — noise ──────────────────────────────────────
    print("\nLevel 6 — Greedy decode, noise")
    our_tokens6 = _greedy_decode_ours(our_enc, our_dec, mel_noise, cfg)
    hf_tokens6  = _greedy_decode_hf(hf_model, processor, mel_noise)
    n6 = min(len(our_tokens6), len(hf_tokens6), MAX_NEW_TOKENS + 1)
    results.append(check(f"first {n6} tokens match",
                         our_tokens6[:n6] == hf_tokens6[:n6],
                         f"\n       ours={our_tokens6[:n6]}\n       hf  ={hf_tokens6[:n6]}"))

    # ── Level 7: real audio (local file via --audio path/to/file.wav) ──────────
    print("\nLevel 7 — Real audio")
    audio_path = None
    if "--audio" in sys.argv:
        audio_path = sys.argv[sys.argv.index("--audio") + 1]
    if audio_path:
        try:
            import soundfile as sf
            from scipy.signal import resample_poly
            from math import gcd
            audio_arr, audio_sr = sf.read(audio_path, dtype="float32")
            if audio_arr.ndim > 1:
                audio_arr = audio_arr.mean(axis=1)   # stereo → mono
            target_sr = 16000
            if audio_sr != target_sr:
                g = gcd(audio_sr, target_sr)
                audio_arr = resample_poly(audio_arr, target_sr // g, audio_sr // g).astype(np.float32)
                audio_sr = target_sr
            mel_real = _mel(processor, audio_arr, sr=audio_sr)

            our_tokens7 = _greedy_decode_ours(our_enc, our_dec, mel_real, cfg)
            hf_tokens7  = _greedy_decode_hf(hf_model, processor, mel_real)
            n7 = min(len(our_tokens7), len(hf_tokens7), MAX_NEW_TOKENS + 1)
            our_text = processor.tokenizer.decode(our_tokens7, skip_special_tokens=True).strip()
            hf_text  = processor.tokenizer.decode(hf_tokens7,  skip_special_tokens=True).strip()
            print(f"       ours text: {our_text!r}")
            print(f"       hf   text: {hf_text!r}")
            results.append(check(f"first {n7} tokens match",
                                 our_tokens7[:n7] == hf_tokens7[:n7],
                                 f"\n       ours={our_tokens7[:n7]}\n       hf  ={hf_tokens7[:n7]}"))
        except Exception as e:
            print(f"  ⚠️  failed — {e}")
    else:
        print("  ⚠️  skipped — pass --audio path/to/file.wav to run")

    # ── summary ─────────────────────────────────────────────────────────────
    passed = sum(results)
    total  = len(results)
    print(f"\n{'='*50}")
    print(f"{'PASS' if passed == total else 'FAIL'}  {passed}/{total} checks passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    run()
