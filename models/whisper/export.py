# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

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
# coreai-models = { path = "../../python", editable = true }
# ///
import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from coreai.runtime import AIModelAssetMetadata
from coreai_torch import TorchConverter, get_decomp_table


class WhisperModule(torch.nn.Module):
    def __init__(self, model_name: str, dtype: torch.dtype):
        super().__init__()
        self._model = transformers.AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=dtype,
            use_safetensors=True,
        )

    def forward(self, input_features, decoder_input_ids):
        outputs = self._model(
            input_features=input_features, decoder_input_ids=decoder_input_ids
        )
        return outputs.logits


def reference_inputs(model_name: str, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    processor = transformers.AutoProcessor.from_pretrained(model_name)
    # 5 seconds of 16kHz mono audio; the feature extractor pads/trims to 30s.
    dummy_audio = np.random.randn(16000 * 5).astype(np.float32)
    feature = processor.feature_extractor(dummy_audio, sampling_rate=16000)
    return {
        "input_features": torch.tensor(feature["input_features"]).to(dtype),
        # Whisper's <|startoftranscript|> token.
        "decoder_input_ids": torch.tensor([[50258]], dtype=torch.int32),
    }


def _default_output_dir() -> str:
    return str(Path(__file__).resolve().parents[2] / "exports")


def _variant_name(model_name: str, dtype: torch.dtype) -> str:
    safe_name = Path(model_name).name
    dtype_name = str(dtype).split(".")[-1]
    return f"{safe_name}_{dtype_name}"


def _asset_path(output_dir: str, model_name: str, dtype: torch.dtype) -> Path:
    return Path(output_dir) / f"{_variant_name(model_name, dtype)}.aimodel"


def _save_asset(coreai_program, model_path: Path, overwrite: bool) -> None:
    if model_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{model_path} already exists. Pass --overwrite to replace it."
            )
        if model_path.is_dir():
            shutil.rmtree(model_path)
        else:
            model_path.unlink()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    coreai_program.save_asset(model_path, _build_aimodel_metadata())


def _build_aimodel_metadata() -> AIModelAssetMetadata:
    # Source: https://huggingface.co/openai/whisper-large-v3
    metadata = AIModelAssetMetadata()
    metadata.author = "A. Radford et al."
    metadata.license = "Apache-2.0"
    metadata.model_description = "Whisper is an automatic speech recognition (ASR) encoder-decoder model from OpenAI, trained on a large multilingual and multitask supervised dataset. Source: https://huggingface.co/openai/whisper-large-v3"
    metadata.creation_date = int(time.time())
    return metadata


def create_whisper(
    output_dir: str,
    model_name: str,
    dtype: torch.dtype,
    overwrite: bool,
):
    print("[INFO] Sourcing model...")
    model = WhisperModule(model_name, dtype)
    model.eval()
    print("[INFO] Model sourced. Running torch export with decompositions...")

    example_inputs = reference_inputs(model_name, dtype)

    with torch.autocast(device_type="cpu", dtype=dtype):
        exported = torch.export.export(model, args=(), kwargs=example_inputs)
    exported = exported.run_decompositions(get_decomp_table())
    print("[INFO] Model exported. Converting to Core AI...")

    converter = TorchConverter().add_exported_program(
        exported_program=exported,
        input_names=["input_features", "decoder_input_ids"],
        output_names=["logits"],
    )
    coreai_program = converter.to_coreai()
    print("[INFO] Model converted.")
    coreai_program.optimize()
    print("[INFO] Model optimized.")

    model_path = _asset_path(output_dir, model_name, dtype)
    _save_asset(coreai_program, model_path, overwrite)
    print(f"[INFO] Successfully created and saved Core AI model to {model_path}.")


def create_whisper_coreai(
    output_dir: str,
    model_name: str,
    dtype: torch.dtype,
    overwrite: bool,
) -> None:
    from coreai_models.export._constants import KEY_CACHE_NAME, VALUE_CACHE_NAME
    from coreai_models.export.macos import export_to_coreai
    from coreai_models.models.macos.whisper import WhisperDecoder, WhisperEncoder
    from coreai_models.primitives.macos.cache import KVCache

    print("[INFO] Loading HuggingFace model for weight extraction...")
    hf_model = transformers.AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name, torch_dtype=dtype, use_safetensors=True
    )
    cfg = hf_model.config

    d_model = cfg.d_model
    head_dim = d_model // cfg.encoder_attention_heads

    # ---- encoder ----
    encoder = WhisperEncoder(
        num_mel_bins=cfg.num_mel_bins,
        d_model=d_model,
        n_heads=cfg.encoder_attention_heads,
        n_layers=cfg.encoder_layers,
        ffn_dim=cfg.encoder_ffn_dim,
        max_source_positions=cfg.max_source_positions,
    )
    enc_sd = {
        k[len("model.encoder.") :]: v
        for k, v in hf_model.state_dict().items()
        if k.startswith("model.encoder.")
    }
    encoder.load_state_dict(enc_sd, strict=True)
    encoder = encoder.to(dtype).eval()

    # ---- decoder ----
    decoder = WhisperDecoder(
        vocab_size=cfg.vocab_size,
        d_model=d_model,
        n_heads=cfg.decoder_attention_heads,
        n_layers=cfg.decoder_layers,
        ffn_dim=cfg.decoder_ffn_dim,
        max_target_positions=cfg.max_target_positions,
    )
    dec_sd = {
        k[len("model.decoder.") :]: v
        for k, v in hf_model.state_dict().items()
        if k.startswith("model.decoder.")
    }
    decoder.load_state_dict(dec_sd, strict=False)
    decoder = decoder.to(dtype).eval()

    del hf_model

    # ---- export encoder (static graph, no KV cache) ----
    print("[INFO] Exporting encoder...")
    enc_inputs = {"input_features": torch.zeros(1, cfg.num_mel_bins, 3000, dtype=dtype)}
    enc_program = export_to_coreai(
        encoder,
        enc_inputs,
        input_names=("input_features",),
        output_names=("encoder_hidden_states",),
    )
    enc_program.optimize()

    # ---- export decoder (autoregressive, KV cache as state) ----
    print("[INFO] Exporting decoder...")
    # input_ids is always (1, 1) — single token per step — so we make it static.
    # position_ids grows from length 1 (step 0) to max_target_positions - 1, so it's dynamic.
    # Trace with a non-trivial offset so torch.export sees offset > 0.
    _to = 5   # trace offset: position_ids trace length = 1 + _to = 6
    _tc = 32  # trace KV cache seq len (must be ≤ max_target_positions)
    _max = cfg.max_target_positions  # 448

    k_cache = torch.zeros(
        cfg.decoder_layers, 1, cfg.decoder_attention_heads, _tc, head_dim, dtype=dtype
    )
    v_cache = torch.zeros(
        cfg.decoder_layers, 1, cfg.decoder_attention_heads, _tc, head_dim, dtype=dtype
    )
    dec_ref = {
        "input_ids": torch.randint(0, cfg.vocab_size, (1, 1), dtype=torch.int32),
        "position_ids": torch.arange(1 + _to, dtype=torch.int32).unsqueeze(0),
        "encoder_hidden_states": torch.zeros(1, cfg.max_source_positions, d_model, dtype=dtype),
        "k_cache": k_cache,
        "v_cache": v_cache,
    }
    dynamic_shapes = {
        "input_ids": {},  # always (1, 1) — static
        "position_ids": {
            1: torch.export.Dim("dec_pos_len", min=1, max=_max - 1)
        },
        "encoder_hidden_states": {},
        "k_cache": {
            KVCache.seq_len_dim(): torch.export.Dim(
                "k_dec_seq_len", min=_tc, max=_max
            )
        },
        "v_cache": {
            KVCache.seq_len_dim(): torch.export.Dim(
                "v_dec_seq_len", min=_tc, max=_max
            )
        },
    }
    dec_program = export_to_coreai(
        decoder,
        dec_ref,
        dynamic_shapes=dynamic_shapes,
        input_names=("input_ids", "position_ids", "encoder_hidden_states"),
        output_names=("logits",),
        state_names=(KEY_CACHE_NAME, VALUE_CACHE_NAME),
    )
    dec_program.optimize()

    # ---- save bundle ----
    bundle_dir = Path(output_dir) / f"{_variant_name(model_name, dtype)}_coreai"
    if bundle_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{bundle_dir} already exists. Pass --overwrite to replace it."
            )
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    enc_path = bundle_dir / "encoder.aimodel"
    dec_path = bundle_dir / "decoder.aimodel"
    enc_program.save_asset(enc_path, _build_aimodel_metadata())
    dec_program.save_asset(dec_path, _build_aimodel_metadata())
    print(f"[INFO] Saved encoder  → {enc_path}")
    print(f"[INFO] Saved decoder  → {dec_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create and save a Core AI AIProgram for Whisper."
    )
    parser.add_argument(
        "--model",
        choices=["openai/whisper-large-v3-turbo", "openai/whisper-large-v3"],
        default="openai/whisper-large-v3-turbo",
        help="Model variant to convert.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for the .aimodel asset (default: <repo-root>/exports/)",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float32",
        help="Torch dtype to use for the model.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing .aimodel asset at the output path.",
    )
    parser.add_argument(
        "--mode",
        choices=["legacy", "coreai"],
        default="legacy",
        help="Export mode: 'legacy' uses TorchConverter directly; 'coreai' uses CoreAI primitives.",
    )
    args = parser.parse_args()

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    output_dir = args.output_dir or _default_output_dir()

    if args.mode == "coreai":
        create_whisper_coreai(output_dir, args.model, dtype, args.overwrite)
    else:
        create_whisper(output_dir, args.model, dtype, args.overwrite)


if __name__ == "__main__":
    main()
