# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Smoke tests for the DiffusionGemma encoder/decoder modules.

transformers 4.57.x does not implement the ``diffusion_gemma`` architecture, so
these tests exercise the module structure and forward-pass shapes on a tiny
random-initialized config rather than comparing against a HuggingFace reference.
"""

import torch

from coreai_models.models.macos.diffusion_gemma import (
    DiffusionGemmaDecoderForCoreAI,
    DiffusionGemmaEncoderForCoreAI,
)
from coreai_models.models.macos.diffusion_gemma_config import DiffusionGemmaTextConfig


def _tiny_config() -> DiffusionGemmaTextConfig:
    # 2 layers: one sliding + one full (exercises tied-KV + proportional RoPE).
    return DiffusionGemmaTextConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=16,
        global_head_dim=32,
        num_key_value_heads=2,
        num_global_key_value_heads=1,
        intermediate_size=48,
        moe_intermediate_size=24,
        num_experts=4,
        top_k_experts=2,
        vocab_size=100,
        max_position_embeddings=64,
        sliding_window=8,
        layer_types=["sliding_attention", "full_attention"],
    )


def _init_linears(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.02)


def test_encoder_forward_shapes() -> None:
    cfg = _tiny_config()
    torch.manual_seed(0)
    enc = DiffusionGemmaEncoderForCoreAI(cfg).eval()
    _init_linears(enc)

    seq_len, ctx = 5, 16
    n_kv, head_dim = enc.num_key_value_heads, enc.head_dim
    input_ids = torch.randint(0, cfg.vocab_size, (1, seq_len), dtype=torch.int32)
    position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
    k_cache = torch.zeros(cfg.num_hidden_layers, 1, n_kv, ctx, head_dim)
    v_cache = torch.zeros_like(k_cache)

    with torch.no_grad():
        logits = enc(input_ids, position_ids, k_cache, v_cache)

    assert logits.shape == (1, seq_len, cfg.vocab_size)
    assert torch.isfinite(logits).all()


def test_decoder_forward_shapes() -> None:
    cfg = _tiny_config()
    torch.manual_seed(0)
    dec = DiffusionGemmaDecoderForCoreAI(cfg).eval()
    _init_linears(dec)

    canvas, enc_len = 8, 6
    n_kv, head_dim = dec.num_key_value_heads, dec.head_dim
    decoder_input_ids = torch.randint(0, cfg.vocab_size, (1, canvas), dtype=torch.int32)
    prev_soft = torch.zeros(1, canvas, cfg.hidden_size)
    position_ids = torch.arange(enc_len, enc_len + canvas, dtype=torch.int32).unsqueeze(0)
    encoder_k = torch.zeros(cfg.num_hidden_layers, 1, n_kv, enc_len, head_dim)
    encoder_v = torch.zeros_like(encoder_k)
    temperature = torch.tensor([0.8])

    with torch.no_grad():
        logits, soft_out = dec(
            decoder_input_ids, prev_soft, position_ids, encoder_k, encoder_v, temperature
        )

    assert logits.shape == (1, canvas, cfg.vocab_size)
    assert soft_out.shape == (1, canvas, cfg.hidden_size)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(soft_out).all()
