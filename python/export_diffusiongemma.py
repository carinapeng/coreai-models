# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Export DiffusionGemma-26B-A4B to Core AI

Produces a bundle with .aimodel components plus tokenizer + metadata:

  encoder.aimodel           — autoregressive prompt encoder (fills KV cache)
  decoder.aimodel           — bidirectional block-diffusion canvas denoiser
  self_conditioning.aimodel — self-conditioning MLP (applied on steps > 0)

Usage:
    uv run python python/export_diffusiongemma.py \\
        --model google/diffusiongemma-26b-a4b-it \\
        --max-ctx 4096 --canvas-length 256 --output-dir ./exports/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from coreai_models.export.macos import export_macos_model, export_to_coreai  # noqa: E402
from coreai_models.export.metadata import build_aimodel_metadata  # noqa: E402
from coreai_models.export.pipeline import ExportConfig  # noqa: E402
from coreai_models.models.macos.diffusion_gemma import (  # noqa: E402
    DiffusionGemmaDecoderForCoreAI,
    DiffusionGemmaSelfConditioningForCoreAI,
    load_diffusion_gemma_decoder,
    load_diffusion_gemma_encoder,
    _mutate_diffusion_gemma_state_dict,
)
from coreai_models.models.macos.diffusion_gemma_config import (  # noqa: E402
    DiffusionGemmaConfig,
    DiffusionGemmaGenerationConfig,
)

logger = logging.getLogger("export_diffusiongemma")

# Trace-time encoder context length used for the decoder's cross-attention KV.
_TRACE_ENC_CTX = 8


def _resolve_dtype(precision: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[precision]


def export_diffusiongemma(
    hf_model_id: str,
    max_ctx: int = 4096,
    canvas_length: int = 256,
    compression: str = "none",
    output_dir: str = "./exports",
    output_name: str | None = None,
    num_layers: int | None = None,
    overwrite: bool = False,
    compute_precision: str = "bfloat16",
    encoder_only: bool = False,
) -> str:
    """Export DiffusionGemma to a Core AI bundle."""
    dtype = _resolve_dtype(compute_precision)
    name = output_name or "diffusiongemma_26b_a4b"
    bundle = Path(output_dir) / name
    bundle.mkdir(parents=True, exist_ok=True)

    full_cfg = DiffusionGemmaConfig.from_pretrained(hf_model_id)
    gen_cfg = DiffusionGemmaGenerationConfig.from_pretrained(hf_model_id)
    text_cfg = full_cfg.text_config

    meta = build_aimodel_metadata(hf_model_id)

    # Encoder (autoregressive, fills KV cache)
    logger.info("Loading + exporting ENCODER ...")
    encoder = load_diffusion_gemma_encoder(
        hf_model_id, target_dtype=dtype, max_context_length=max_ctx, num_layers=num_layers
    )

    if compression and compression != "none":
        encoder = _quantize_encoder(encoder, hf_model_id, compression, dtype, max_ctx)

    export_cfg = ExportConfig(
        hf_model_id=hf_model_id,
        max_context_length=max_ctx,
        compute_precision=compute_precision,
        compression=compression,
    )
    enc_prog = export_macos_model(encoder, encoder, export_cfg)
    enc_path = bundle / "encoder.aimodel"
    if enc_path.exists() and overwrite:
        import shutil
        shutil.rmtree(enc_path)
    enc_prog.save_asset(enc_path, meta)
    logger.info("  encoder.aimodel: %d KB", (enc_path / "main.mlirb").stat().st_size // 1024)
    n_layers_eff = encoder.config.num_hidden_layers
    cache_n_kv = encoder.num_key_value_heads
    cache_hd = encoder.head_dim
    del encoder

    if encoder_only:
        _save_tokenizer(hf_model_id, bundle / "tokenizer")
        _write_bundle_metadata(bundle, name, hf_model_id, text_cfg, gen_cfg, full_cfg,
                               max_ctx, canvas_length, compression, num_layers,
                               encoder_only=True)
        logger.info("Encoder-only bundle written to %s", bundle)
        return str(bundle)

    # Decoder (bidirectional canvas denoiser)
    logger.info("Loading + exporting DECODER ...")
    decoder = load_diffusion_gemma_decoder(
        hf_model_id, target_dtype=dtype, num_layers=num_layers
    )
    H = text_cfg.hidden_size
    canvas = torch.randn(1, canvas_length, H, dtype=dtype)
    pos = torch.arange(canvas_length, dtype=torch.int32).unsqueeze(0)
    enc_k = torch.zeros(n_layers_eff, 1, cache_n_kv, _TRACE_ENC_CTX, cache_hd, dtype=dtype)
    enc_v = torch.zeros_like(enc_k)
    dec_prog = export_to_coreai(
        decoder,
        {"canvas_embeds": canvas, "position_ids": pos, "encoder_k": enc_k, "encoder_v": enc_v},
        output_names=("logits",),
    )
    dec_path = bundle / "decoder.aimodel"
    if dec_path.exists() and overwrite:
        import shutil
        shutil.rmtree(dec_path)
    dec_prog.save_asset(dec_path, meta)
    logger.info("  decoder.aimodel: %d KB", (dec_path / "main.mlirb").stat().st_size // 1024)
    del decoder

    # Self-conditioning MLP
    logger.info("Loading + exporting SELF-CONDITIONING ...")
    sc = _load_self_conditioning(hf_model_id, dtype, num_layers)
    sc_prog = export_to_coreai(
        sc, {"canvas_embeds": canvas}, output_names=("conditioned_embeds",)
    )
    sc_path = bundle / "self_conditioning.aimodel"
    if sc_path.exists() and overwrite:
        import shutil
        shutil.rmtree(sc_path)
    sc_prog.save_asset(sc_path, meta)
    logger.info("  self_conditioning.aimodel: %d KB", (sc_path / "main.mlirb").stat().st_size // 1024)
    del sc

    # Tokenizer
    _save_tokenizer(hf_model_id, bundle / "tokenizer")

    # Bundle metadata
    _write_bundle_metadata(bundle, name, hf_model_id, text_cfg, gen_cfg, full_cfg,
                           max_ctx, canvas_length, compression, num_layers,
                           encoder_only=False)
    logger.info("Bundle written to %s", bundle)
    return str(bundle)


def _write_bundle_metadata(bundle, name, hf_model_id, text_cfg, gen_cfg, full_cfg,
                           max_ctx, canvas_length, compression, num_layers, encoder_only):
    """Write the bundle-level metadata.json."""
    assets = {"encoder": "encoder.aimodel"}
    if not encoder_only:
        assets["decoder"] = "decoder.aimodel"
        assets["self_conditioning"] = "self_conditioning.aimodel"
    bundle_meta = {
        "metadata_version": "0.2",
        "kind": "diffusion_llm",
        "name": name,
        "assets": assets,
        "language": {
            "tokenizer": "tokenizer",
            "vocab_size": text_cfg.vocab_size,
            "max_context_length": max_ctx,
            "embedded_tokenizer": True,
        },
        "diffusion": {
            "canvas_length": canvas_length,
            "max_denoising_steps": gen_cfg.max_denoising_steps,
            "entropy_bound": gen_cfg.entropy_bound,
            "confidence_threshold": gen_cfg.confidence_threshold,
            "stability_threshold": gen_cfg.stability_threshold,
            "t_max": gen_cfg.t_max,
            "t_min": gen_cfg.t_min,
            "mask_token_id": getattr(full_cfg, "mask_token_id", full_cfg.image_token_id),
        },
        "source": {
            "hf_model_id": hf_model_id,
            "model_definition": "torch",
            "compression": compression,
            "num_layers": num_layers,
            "encoder_only": encoder_only,
        },
    }
    with open(bundle / "metadata.json", "w") as f:
        json.dump(bundle_meta, f, indent=2)


def _quantize_encoder(encoder, hf_model_id, compression, dtype, max_ctx):
    """Apply pre-export weight quantization (e.g. 4bit) to the encoder.

    The encoder's forward contract is (input_ids, position_ids, k_cache, v_cache),
    identical to a standard LLM, so we can reuse the shared quantization preset
    and helper from the main export pipeline.
    """
    import torch
    from coreai_models.export.compression import quantize_pytorch_model
    from coreai_models.export.presets import get_preset
    from coreai_models.export._constants import (
        QUANT_TRACE_OFFSET,
        QUANT_TRACE_QUERY_LEN,
        TRACE_KV_CACHE_SEQ_LEN,
    )
    from coreai_models.primitives.macos.cache import KVCache

    preset = get_preset(compression)
    quant_cfg = preset.get("torch_quantization_config")
    if quant_cfg is None:
        logger.warning("Compression '%s' has no torch_quantization_config; skipping", compression)
        return encoder
    quant_cfg = dict(quant_cfg)

    cfg = encoder.config
    vocab = cfg.vocab_size
    input_ids = torch.randint(1, vocab, (1, QUANT_TRACE_QUERY_LEN), dtype=torch.int32)
    position_ids = (
        torch.arange(QUANT_TRACE_QUERY_LEN + QUANT_TRACE_OFFSET, dtype=torch.int32)
        .unsqueeze(0)
    )
    # Build a trace-sized cache via the encoder's cache-sizing attributes.
    k_cache = torch.zeros(
        encoder.num_hidden_layers, 1, encoder.num_key_value_heads,
        TRACE_KV_CACHE_SEQ_LEN, encoder.head_dim, dtype=dtype,
    )
    v_cache = torch.zeros_like(k_cache)

    dynamic_shapes = {
        "input_ids": {1: torch.export.Dim("seq_ids", max=TRACE_KV_CACHE_SEQ_LEN - 2)},
        "position_ids": {1: torch.export.Dim(
            "seq_pos", min=QUANT_TRACE_QUERY_LEN, max=TRACE_KV_CACHE_SEQ_LEN - 1)},
        "k_cache": None,
        "v_cache": None,
    }

    logger.info("Applying %s weight quantization to encoder ...", compression)
    return quantize_pytorch_model(
        encoder,
        (input_ids, position_ids, k_cache, v_cache),
        dynamic_shapes,
        quant_cfg,
    )


def _load_self_conditioning(hf_model_id, dtype, num_layers):
    """Load the self-conditioning MLP weights from the checkpoint."""
    from huggingface_hub import snapshot_download
    from coreai_models.models.base import (
        _build_safetensors_key_index,
        _load_tensors_for_keys,
        _resolve_safetensors_files,
    )

    full_cfg = DiffusionGemmaConfig.from_pretrained(hf_model_id)
    cfg = full_cfg.text_config
    if num_layers is not None:
        cfg.num_hidden_layers = num_layers
    sc = DiffusionGemmaSelfConditioningForCoreAI(cfg).to(dtype)

    model_dir = snapshot_download(
        hf_model_id, allow_patterns=["*.safetensors", "*.safetensors.index.json"]
    )
    sf_files = _resolve_safetensors_files(model_dir)
    per_layer, shared = _build_safetensors_key_index(sf_files, num_layers=None, hf_state_dict_prefix="")
    sd = _load_tensors_for_keys(shared, dtype)
    # Keep only self_conditioning keys, rename to module namespace.
    sc_sd = {}
    for k, v in sd.items():
        if k.startswith("model.decoder.self_conditioning."):
            new_k = k.replace("model.decoder.self_conditioning.", "self_conditioning.")
            sc_sd[new_k] = v
    sc.load_state_dict(sc_sd, strict=True, assign=True)
    return sc.eval()


def _save_tokenizer(hf_model_id: str, dest: Path) -> None:
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(hf_model_id)
        dest.mkdir(parents=True, exist_ok=True)
        tok.save_pretrained(str(dest))
        logger.info("  tokenizer saved to %s", dest)
    except Exception as e:  # noqa: BLE001
        logger.warning("  tokenizer save failed (%s); bundle will need an external tokenizer", e)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="export_diffusiongemma",
        description="Export DiffusionGemma to Core AI .aimodel format (encoder + decoder + self-conditioning).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--max-ctx", type=int, default=4096, help="Encoder KV-cache max context (default 4096)")
    parser.add_argument("--canvas-length", type=int, default=256, help="Canvas length (default 256)")
    parser.add_argument("--compression", default="none", help="Compression preset or 'none'")
    parser.add_argument("--compute-precision", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--output-dir", default="./exports")
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--num-layers", type=int, default=None, help="Truncate to N layers (smoke test)")
    parser.add_argument("--encoder-only", action="store_true",
                        help="Export only the encoder (a complete autoregressive LM). "
                             "Use for real text-in/text-out validation without the diffusion loop.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    result = export_diffusiongemma(
        hf_model_id=args.model,
        max_ctx=args.max_ctx,
        canvas_length=args.canvas_length,
        compression=args.compression,
        output_dir=args.output_dir,
        output_name=args.output_name,
        num_layers=args.num_layers,
        overwrite=args.overwrite,
        compute_precision=args.compute_precision,
        encoder_only=args.encoder_only,
    )
    print(f"Export complete: {result}")


if __name__ == "__main__":
    main()
