# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "coreai-core==1.0.0b1",
#     "coreai-torch==0.4.0",
#     "transformers==4.57.3",
#     "coreai-models",
# ]
#
# [tool.uv]
# index-url       = "https://pypi.org/simple"
# prerelease      = "allow"
# index-strategy  = "unsafe-best-match"
#
# [tool.uv.sources]
# coreai-models = { path = "python", editable = true }
# ///
"""Smoke-test for the Whisper CoreAI export.

Usage:
    # First export the models:
    uv run models/whisper/export.py --mode coreai --dtype float32 --overwrite

    # Then run this test:
    uv run python test_whisper.py
"""

import asyncio
import sys
from pathlib import Path

import numpy as np
import torch

EXPORTS_DIR = Path(__file__).resolve().parent / "exports"
MODEL_NAME = "openai/whisper-large-v3-turbo"
VARIANT = f"{Path(MODEL_NAME).name}_float32"
BUNDLE_DIR = EXPORTS_DIR / f"{VARIANT}_coreai"
ENC_PATH = BUNDLE_DIR / "encoder.aimodel"
DEC_PATH = BUNDLE_DIR / "decoder.aimodel"

# Whisper decoder config (whisper-large-v3-turbo)
N_DEC_LAYERS = 4
N_DEC_HEADS = 20
HEAD_DIM = 64          # 1280 / 20
MAX_TARGET_POS = 448
N_MEL = 128
ENC_SEQ_LEN = 1500
D_MODEL = 1280
VOCAB_SIZE = 51866

SOT_TOKEN = 50258       # <|startoftranscript|>
EOT_TOKEN = 50257       # <|endoftext|>
MAX_NEW_TOKENS = 20


async def _run_coreai() -> list[int]:
    """Load the exported .aimodel files and greedy-decode MAX_NEW_TOKENS tokens."""
    from coreai.runtime import AIModel, NDArray

    print(f"[INFO] Loading encoder from {ENC_PATH}")
    enc_model = await AIModel.load(ENC_PATH)
    print(f"[INFO] Loading decoder from {DEC_PATH}")
    dec_model = await AIModel.load(DEC_PATH)

    enc_fn = enc_model.load_function("main")
    dec_fn = dec_model.load_function("main")

    # Dummy mel features (30 s of silence → shape (1, 128, 3000))
    mel = np.zeros((1, N_MEL, 3000), dtype=np.float32)
    enc_out = await enc_fn({"input_features": NDArray(data=mel)})
    encoder_hidden = enc_out["encoder_hidden_states"]  # NDArray (1, 1500, 1280)
    print(f"[INFO] Encoder done — hidden shape: {encoder_hidden.numpy().shape}")

    # Pre-allocate KV caches (filled with zeros; they are mutated as states)
    k_cache = np.zeros(
        (N_DEC_LAYERS, 1, N_DEC_HEADS, MAX_TARGET_POS, HEAD_DIM), dtype=np.float32
    )
    v_cache = np.zeros(
        (N_DEC_LAYERS, 1, N_DEC_HEADS, MAX_TARGET_POS, HEAD_DIM), dtype=np.float32
    )
    state = {
        "keyCache": NDArray(data=k_cache),
        "valueCache": NDArray(data=v_cache),
    }

    tokens = [SOT_TOKEN]
    for step in range(MAX_NEW_TOKENS):
        # position_ids: [0, 1, ..., step]  shape (1, step+1)
        # The decoder derives offset = seq_len - query_len = (step+1) - 1 = step
        cur_token = np.array([[tokens[-1]]], dtype=np.int32)
        pos_ids = np.arange(step + 1, dtype=np.int32).reshape(1, -1)

        dec_inputs = {
            "input_ids": NDArray(data=cur_token),
            "position_ids": NDArray(data=pos_ids),
            "encoder_hidden_states": encoder_hidden,
        }
        dec_out = await dec_fn(dec_inputs, state=state)
        logits = dec_out["logits"].numpy()           # (1, 1, vocab_size)
        next_token = int(np.argmax(logits[0, -1]))
        tokens.append(next_token)
        print(f"  step {step:2d}: token {next_token}")
        if next_token == EOT_TOKEN:
            print("[INFO] EOT reached.")
            break

    return tokens


def _run_torch() -> list[int]:
    """Fallback: run WhisperEncoder + WhisperDecoder in pure PyTorch (random weights)."""
    from coreai_models.models.macos.whisper import WhisperDecoder, WhisperEncoder
    from coreai_models.primitives.macos.cache import KVCache

    print("[INFO] Running PyTorch forward (random weights — sanity check only)")
    dtype = torch.float32

    encoder = WhisperEncoder(N_MEL, D_MODEL, N_DEC_HEADS, N_DEC_LAYERS, 5120, ENC_SEQ_LEN).to(dtype).eval()
    decoder = WhisperDecoder(VOCAB_SIZE, D_MODEL, N_DEC_HEADS, N_DEC_LAYERS, 5120, MAX_TARGET_POS).to(dtype).eval()

    with torch.no_grad():
        mel = torch.zeros(1, N_MEL, 3000, dtype=dtype)
        enc_out = encoder(mel)
        print(f"[INFO] Encoder output shape: {enc_out.shape}")

        k_cache = torch.zeros(N_DEC_LAYERS, 1, N_DEC_HEADS, MAX_TARGET_POS, HEAD_DIM, dtype=dtype)
        v_cache = torch.zeros(N_DEC_LAYERS, 1, N_DEC_HEADS, MAX_TARGET_POS, HEAD_DIM, dtype=dtype)

        tokens = [SOT_TOKEN]
        for step in range(MAX_NEW_TOKENS):
            cur_token = torch.tensor([[tokens[-1]]], dtype=torch.int32)
            pos_ids = torch.arange(step + 1, dtype=torch.int32).unsqueeze(0)
            logits = decoder(cur_token, pos_ids, enc_out, k_cache, v_cache)
            next_token = int(logits[0, -1].argmax())
            tokens.append(next_token)
            print(f"  step {step:2d}: token {next_token}")
            if next_token == EOT_TOKEN:
                print("[INFO] EOT reached.")
                break

    return tokens


def main() -> None:
    if ENC_PATH.exists() and DEC_PATH.exists():
        print("[INFO] Found exported .aimodel files — using CoreAI runtime.")
        tokens = asyncio.run(_run_coreai())
    else:
        print(
            f"[WARN] Exported models not found at {BUNDLE_DIR}.\n"
            "       Run: uv run models/whisper/export.py --mode coreai --dtype float32 --overwrite\n"
            "       Falling back to PyTorch forward pass with random weights."
        )
        tokens = _run_torch()

    print(f"\n[RESULT] Generated token IDs: {tokens}")


if __name__ == "__main__":
    main()
