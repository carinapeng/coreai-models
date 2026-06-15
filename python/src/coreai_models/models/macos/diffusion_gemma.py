# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""DiffusionGemma-26B-A4B text decoder for Core AI export.

Architecture summary
--------------------
  * Encoder-decoder with shared transformer weights (30 layers).
  * Encoder pass  - autoregressive, causal mask, builds KV cache from prompt.
  * Decoder pass  - block diffusion over a 256-token canvas; bidirectional
                    (no causal mask); attends to encoder KV via prepended K/V.
  * Per-layer scalar differentiates encoder / decoder modes.
  * Each layer has both a dense MLP and a sparse MoE MLP in parallel.
  * Asymmetric attention heads:
      - sliding layers (25/30): 16 Q heads @ head_dim=256, 8 KV @ 256, window=1024
      - full layers   (every 6th): 16 Q heads @ head_dim=512, 2 KV @ 512, no window
  * Full-attention layers use Partial RoPE (proportional rope_type).
  * MoE: 128 experts / top-8 active; weights are pre-stacked in checkpoint.
  * Final logit softcapping at 30.0.

The forward-pass arithmetic in this module mirrors the reference
``transformers`` DiffusionGemma implementation (Encoder/Decoder text layers,
attention, router, experts and RoPE) exactly. See the reference for details.

Two exported models
-------------------
  DiffusionGemmaEncoderForCoreAI  - causal, KV-cache, encoder scalars
  DiffusionGemmaDecoderForCoreAI  - bidirectional, 256-token canvas, decoder scalars

Weight key mapping (_mutate_diffusion_gemma_state_dict)
-------------------------------------------------------
  HF checkpoint layout                              -> Our layout
  model.decoder.embed_tokens.weight                 -> model.embed_tokens.weight
  model.decoder.norm.weight                         -> model.norm.weight
  model.decoder.self_conditioning.*                 -> model.self_conditioning.*
  model.decoder.layers.{i}.layer_scalar             -> model.layers.{i}.decoder_scalar
  model.encoder.language_model.layers.{i}.layer_scalar
                                                    -> model.layers.{i}.encoder_scalar
  model.decoder.layers.{i}.self_attn.q_proj.weight  -> self_attn.q_proj.weight
  model.decoder.layers.{i}.self_attn.k_proj.weight  -> self_attn.k_proj.weight
  model.decoder.layers.{i}.self_attn.v_proj.weight  -> self_attn.v_proj.weight (sliding only)
  model.decoder.layers.{i}.self_attn.q_norm.weight  -> self_attn.q_norm.weight
  model.decoder.layers.{i}.self_attn.k_norm.weight  -> self_attn.k_norm.weight
  model.decoder.layers.{i}.experts.gate_up_proj     -> moe.switch_glu.{gate,up}_proj.weight
  model.decoder.layers.{i}.experts.down_proj        -> moe.switch_glu.down_proj.weight
  model.decoder.layers.{i}.router.proj.weight       -> moe.router_proj.weight
  model.decoder.layers.{i}.router.scale             -> moe.router_scale
  model.decoder.layers.{i}.router.per_expert_scale  -> moe.per_expert_scale
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import Self, override

from coreai_models.models.base import BaseForCausalLM
from coreai_models.models.macos.diffusion_gemma_config import (
    DiffusionGemmaConfig,
    DiffusionGemmaTextConfig,
)
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives._ops import mutable_slice_update
from coreai_models.primitives.macos.sdpa import SDPA
from coreai_models.primitives.macos.switch import SwitchGLU


def _update_cache_subregion(
    cache: "KVCache",
    layer_idx: int,
    offset: int,
    k: torch.Tensor,            # [B, n_kv, T, hd]  native per-layer dims
    v: torch.Tensor,
    seq_len: int,
    n_kv: int,
    hd: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write native-sized K/V into a sub-region of the unified cache
    """
    kc, vc = cache._k_cache, cache._v_cache  # [L, 1, cache_n_kv, ctx, cache_hd]
    device = kc.device
    T = k.shape[-2]

    def _write(buf: torch.Tensor, upd: torch.Tensor) -> None:
        # upd: [B, n_kv, T, hd] -> [1(layer), B, n_kv, T, hd]
        begin = torch.tensor((layer_idx, 0, 0, offset, 0), dtype=torch.int32, device=device)
        end = torch.tensor(
            (layer_idx + 1, buf.size(1), n_kv, offset + T, hd),
            dtype=torch.int32, device=device,
        )
        mutable_slice_update(x=buf, update=upd.unsqueeze(0), begin=begin, end=end)

    _write(kc, k)
    _write(vc, v)

    # Read back accumulated K/V for this layer: [1, B, n_kv, seq_len, hd] -> [B, n_kv, seq_len, hd]
    k_out = kc.narrow(0, layer_idx, 1).narrow(2, 0, n_kv).narrow(-2, 0, seq_len).narrow(-1, 0, hd).squeeze(0)
    v_out = vc.narrow(0, layer_idx, 1).narrow(2, 0, n_kv).narrow(-2, 0, seq_len).narrow(-1, 0, hd).squeeze(0)
    return k_out, v_out


# RMSNorm

class DiffusionGemmaRMSNorm(nn.Module):
    """Plain RMSNorm with no +1 weight offset, computed in float32.

        normed = x.float() * (mean(x.float()**2, -1) + eps) ** -0.5
        out    = normed * weight.float()       (only when with_scale=True)
    """

    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        mean_sq = x.pow(2).mean(-1, keepdim=True) + self.eps
        # torch.pow(., -0.5) (not rsqrt) to match the reference numerics.
        return x * torch.pow(mean_sq, -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self._norm(x.float())
        if self.with_scale:
            normed = normed * self.weight.float()
        return normed.type_as(x)


# Rotary position embedding via the Core AI native `rope` composite op


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 2,
) -> torch.Tensor:
    """Apply RoPE to ``x`` [B, T, n_heads, head_dim] via the native rope op.

    cos/sin are HALF-dim [B, T, head_dim/2]; unsqueeze at ``unsqueeze_dim`` (=2)
    so they broadcast over the head axis -> [B, T, 1, head_dim/2]. The native
    `rope` op computes the standard split-half rotation
    (`y1 = cos*x1 - sin*x2; y2 = sin*x1 + cos*x2`)
    """
    from coreai_torch.composite_ops._rope import rope as _native_rope

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return _native_rope(x, cos=cos, sin=sin)


# Scaled token embedding

class ScaledEmbedding(nn.Embedding):
    """nn.Embedding that scales outputs by ``embed_scale`` (= sqrt(hidden_size)).

    Gemma / DiffusionGemma normalize token embeddings by sqrt(hidden_size) before
    the first transformer layer.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, embed_scale: float = 1.0):
        super().__init__(num_embeddings, embedding_dim)
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * torch.tensor(
            self.embed_scale, dtype=self.weight.dtype, device=self.weight.device
        )


# Attention

class DiffusionGemmaAttention(nn.Module):
    """Shared attention block for both sliding and full-attention layers.
      * Separate q_norm / k_norm / v_norm (DiffusionGemmaRMSNorm(head_dim)).
        v_norm has ``with_scale=False`` (pure RMS, no learnable weight).
      * Order per head: q = q_norm(q_proj(x)); rope(q);
                        k = k_norm(k_proj(x)); rope(k);
                        v = v_norm(value);     no rope on v.
      * Full-attention (is_sliding=False) layers have NO v_proj weight in the
        checkpoint -> value reuses the *raw* k_proj output (pre-norm, pre-rope),
        then v_norm is applied to it.
      * scaling = 1.0 (no query_pre_attn_scalar).
      * RoPE: sliding -> "default" rope (full head_dim rotated, theta=10000);
              full    -> "proportional" rope (first 25% of head_dim rotated,
                         theta=1e6, remaining dims have zero frequency = no rotation).

    Sliding (25/30): head_dim=256, n_kv=8, window=1024.
    Full    (5/30):  head_dim=512, n_kv=2, no window.

    Two SDPA modes are baked in at construction (is_encoder selects causal vs.
    bidirectional) so the export graph has no dynamic branch.
    """

    def __init__(self, config: DiffusionGemmaTextConfig, layer_idx: int, is_encoder: bool = True) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.is_full = config.is_full_attention_layer(layer_idx)
        self.is_encoder = is_encoder

        self.n_heads = config.num_attention_heads  # 16 always
        dim = config.hidden_size  # 2816

        if self.is_full:
            self.n_kv_heads = config.num_global_key_value_heads  # 2
            self.head_dim = config.global_head_dim  # 512
            rope_full = config.rope_full if isinstance(config.rope_full, dict) else {}
            partial_rotary_factor = rope_full.get("partial_rotary_factor", 0.25)
            rope_base = rope_full.get("rope_theta", 1_000_000.0)
            sdpa_window = 0
            inv_freq = self._proportional_inv_freq(self.head_dim, rope_base, partial_rotary_factor)
        else:
            self.n_kv_heads = config.num_key_value_heads  # 8
            self.head_dim = config.head_dim  # 256
            rope_base = (
                config.rope_sliding.get("rope_theta", 10_000.0)
                if isinstance(config.rope_sliding, dict)
                else 10_000.0
            )
            sdpa_window = config.sliding_window  # 1024
            inv_freq = self._default_inv_freq(self.head_dim, rope_base)

        n_h = self.n_heads
        n_kv = self.n_kv_heads
        hd = self.head_dim

        self.q_proj = nn.Linear(dim, n_h * hd, bias=config.attention_bias)
        self.k_proj = nn.Linear(dim, n_kv * hd, bias=config.attention_bias)
        self.v_proj = (
            None if self.is_full else nn.Linear(dim, n_kv * hd, bias=config.attention_bias)
        )
        self.o_proj = nn.Linear(n_h * hd, dim, bias=config.attention_bias)

        self.q_norm = DiffusionGemmaRMSNorm(hd, eps=config.rms_norm_eps)
        self.k_norm = DiffusionGemmaRMSNorm(hd, eps=config.rms_norm_eps)
        self.v_norm = DiffusionGemmaRMSNorm(hd, eps=config.rms_norm_eps, with_scale=False)

        # inv_freq buffer drives the per-layer-type cos/sin (non-persistent).
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # is_encoder=True  -> causal SDPA (encoder pass)
        # is_encoder=False -> bidirectional SDPA (decoder diffusion pass)
        if is_encoder:
            self.sdpa = SDPA(scale=1.0, window_size=sdpa_window, is_causal=True)
        else:
            self.sdpa = SDPA(scale=1.0, window_size=0, is_causal=False)

        # Cache head_dim used when padding smaller layers for the unified cache.
        self.cache_head_dim = max(config.head_dim, config.global_head_dim)
        self.cache_n_kv_heads = max(config.num_key_value_heads, config.num_global_key_value_heads)

    @staticmethod
    def _default_inv_freq(head_dim: int, base: float) -> torch.Tensor:
        """"default" rope: 1 / base^(arange(0, head_dim, 2) / head_dim)."""
        return 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim)
        )

    @staticmethod
    def _proportional_inv_freq(head_dim: int, base: float, partial_rotary_factor: float) -> torch.Tensor:
        """"proportional" rope: only the first 25% of head_dim is rotated.

        rope_angles  = int(partial_rotary_factor * head_dim // 2)   # 64 for hd=512
        inv_freq_rot = 1 / base^(arange(0, 2*rope_angles, 2) / head_dim)
        nope_angles  = head_dim // 2 - rope_angles                  # 192
        inv_freq     = cat(inv_freq_rot, zeros(nope_angles))        # zero freq => no rotation
        """
        rope_angles = int(partial_rotary_factor * head_dim // 2)
        nope_angles = head_dim // 2 - rope_angles
        inv_freq_rotated = 1.0 / (
            base
            ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.int64).float() / head_dim)
        )
        return torch.cat([inv_freq_rotated, torch.zeros(nope_angles, dtype=inv_freq_rotated.dtype)])

    def _cos_sin(self, position_ids: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute half-dim cos/sin [B, T, head_dim/2] from position_ids [B, T].
        """
        inv_freq = self.inv_freq
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)  # [B, T, hd/2]
        return freqs.cos().to(dtype), freqs.sin().to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
        encoder_k: torch.Tensor | None = None,
        encoder_v: torch.Tensor | None = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """Compute attention output.

        Args:
            x:            Hidden states [B, T, hidden_size].
            position_ids: Absolute position ids [B, seq_len] (may be longer than
                          T when an offset is in use for encoder decode).
            cache:        KVCache updated in-place during the encoder pass.
            encoder_k:    Per-layer encoder K slice [1, n_kv, enc_len, hd_max].
            encoder_v:    Per-layer encoder V slice (same shape as encoder_k).
            is_causal:    True for encoder pass (causal + cache), False for
                          decoder pass (bidirectional + encoder cross-attn).

        Returns:
            Attention output [B, T, hidden_size].
        """
        B, T, _ = x.shape
        n_h, n_kv = self.n_heads, self.n_kv_heads
        hd = self.head_dim

        seq_len = position_ids.shape[-1]
        torch._check_is_size(T)
        torch._check_is_size(seq_len)
        offset = seq_len - T
        torch._check_is_size(offset)
        rope_positions = position_ids.narrow(-1, offset, T)  # [B, T]

        cos, sin = self._cos_sin(rope_positions, x.dtype)

        # -- Q / K / V projections (channels-last head layout [B, T, heads, hd]) --
        query = self.q_proj(x).reshape(B, T, n_h, hd)
        query = self.q_norm(query)
        query = apply_rotary_pos_emb(query, cos, sin, unsqueeze_dim=2)
        query = query.transpose(1, 2)  # [B, n_h, T, hd]

        key_raw = self.k_proj(x).reshape(B, T, n_kv, hd)
        # Full-attention layers reuse the raw (pre-norm, pre-rope) K as the value.
        value_raw = self.v_proj(x).reshape(B, T, n_kv, hd) if self.v_proj is not None else key_raw

        key = self.k_norm(key_raw)
        key = apply_rotary_pos_emb(key, cos, sin, unsqueeze_dim=2)
        key = key.transpose(1, 2)  # [B, n_kv, T, hd]

        value = self.v_norm(value_raw)
        value = value.transpose(1, 2)  # [B, n_kv, T, hd]

        # -- KV cache or encoder cross-attention context ---------------------
        if cache is not None and is_causal:
            # Encoder pass
            key, value = _update_cache_subregion(
                cache, self.layer_idx, offset, key, value,
                seq_len=seq_len, n_kv=n_kv, hd=hd,
            )
        elif encoder_k is not None and encoder_v is not None:
            # Decoder pass: prepend per-layer encoder context to canvas K/V.
            enc_k_layer = encoder_k[:, :n_kv, :, :hd]
            enc_v_layer = encoder_v[:, :n_kv, :, :hd]
            key = torch.cat([enc_k_layer, key], dim=-2)
            value = torch.cat([enc_v_layer, value], dim=-2)

        # -- Scaled dot-product attention (mode baked in at construction) -----
        out = self.sdpa(query=query, key=key, value=value)

        out = out.permute(0, 2, 1, 3).reshape(B, T, n_h * hd)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# Dense MLP
# ---------------------------------------------------------------------------


class DiffusionGemmaDenseMLP(nn.Module):
    """Standard gated MLP (SwiGLU-style with gelu_pytorch_tanh activation).

    Checkpoint weight shapes (hidden_size=2816, intermediate_size=2112):
      gate_proj : [2112, 2816]
      up_proj   : [2112, 2816]
      down_proj : [2816, 2112]
    """

    def __init__(self, config: DiffusionGemmaTextConfig) -> None:
        super().__init__()
        H = config.hidden_size
        I = config.intermediate_size
        self.gate_proj = nn.Linear(H, I, bias=False)
        self.up_proj = nn.Linear(H, I, bias=False)
        self.down_proj = nn.Linear(I, H, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
        )


# ---------------------------------------------------------------------------
# MoE MLP (pre-stacked expert weights)
# ---------------------------------------------------------------------------


class _GeGLUTanh(nn.Module):
    """GeGLU activation for SwitchGLU: gelu_tanh(gate) * up (DiffusionGemma MoE)."""

    def forward(self, up: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return F.gelu(gate, approximate="tanh") * up


class DiffusionGemmaMoEMLP(nn.Module):
    """Sparse Mixture-of-Experts MLP built on the SwitchGLU primitive.
      Router (operates on the raw residual):
        h      = norm(residual)                 # RMSNorm, with_scale=False
        h      = h * scale * hidden_size**-0.5
        scores = softmax(router_proj(h), dim=-1, dtype=float32)
        w, idx = topk(scores, k=top_k)
        w      = w / w.sum(-1, keepdim=True)
        w      = w * per_expert_scale[idx]      # per-expert scale folded in HERE

      Experts (operate on pre_feedforward_layernorm_2(residual)):
        for each selected expert e:
            gate, up = linear(x, gate_up_proj[e]).chunk(2, dim=-1)
            y_e      = down_proj[e](gelu_tanh(gate) * up)
        out = sum_e w_e * y_e

    The expert math is realized by SwitchGLU (gate_proj=gate_up[:, :moe_int],
    up_proj=gate_up[:, moe_int:], activation=gelu_tanh(gate)*up), which exactly
    reproduces ``chunk(2, dim=-1)`` of the fused linear output. ``per_expert_scale``
    is applied exactly once, inside the router, folded into the routing weights.

    Checkpoint layout:
      experts.gate_up_proj  [128, 1408, 2816]  (gate=:704, up=704:)
      experts.down_proj     [128, 2816,  704]
      router.proj.weight    [128, 2816]
      router.scale          [2816]
      router.per_expert_scale [128]
    """

    def __init__(self, config: DiffusionGemmaTextConfig) -> None:
        super().__init__()
        self.n_experts = config.num_experts            # 128
        self.top_k = config.top_k_experts              # 8
        self.moe_int = config.moe_intermediate_size    # 704
        H = config.hidden_size                          # 2816
        self.scalar_root_size = H ** -0.5

        self.switch_glu = SwitchGLU(
            hidden_size=H,
            moe_intermediate_size=self.moe_int,
            num_experts=self.n_experts,
            bias=False,
            activation=_GeGLUTanh(),
        )

        # Router components
        self.router_norm = DiffusionGemmaRMSNorm(H, eps=config.rms_norm_eps, with_scale=False)
        self.router_proj = nn.Linear(H, self.n_experts, bias=False)
        self.router_scale = nn.Parameter(torch.ones(H))
        self.per_expert_scale = nn.Parameter(torch.ones(self.n_experts))

    def forward(self, routing_input: torch.Tensor, experts_input: torch.Tensor) -> torch.Tensor:
        """Route tokens to top-k experts and return the weighted sum output.

        Args:
            routing_input:  RAW residual [B, T, H] (the router applies its own norm).
            experts_input:  pre_feedforward_layernorm_2(residual) [B, T, H].

        Returns:
            MoE output [B, T, H].
        """
        # -- Router ----------------------------------------------------------
        h = self.router_norm(routing_input)
        h = h * self.router_scale * self.scalar_root_size
        scores = F.softmax(self.router_proj(h).float(), dim=-1)        # [B, T, E] (float32)

        top_w, top_idx = torch.topk(scores, self.top_k, dim=-1)        # [B, T, k]
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)
        # Fold per-expert scale into the routing weights (applied exactly once).
        top_w = top_w * self.per_expert_scale[top_idx]                 # [B, T, k]
        top_w = top_w.to(experts_input.dtype)
        # uint16 indices: required by SwitchGLU's GatherMM for correct Core AI
        # lowering (matches the working Qwen3-MoE path; int32 produces garbage).
        top_idx = top_idx.to(torch.uint16)

        # -- Expert computation via SwitchGLU (GatherMM dispatch) ------------
        y = self.switch_glu(experts_input, top_idx)                    # [B, T, k, H]

        # Weighted sum over top-k experts.
        out = (y * top_w.unsqueeze(-1)).sum(dim=2)                     # [B, T, H]
        return out.to(experts_input.dtype)


# ---------------------------------------------------------------------------
# Self-conditioning MLP (applied between diffusion steps)
# ---------------------------------------------------------------------------


class DiffusionGemmaSelfConditioning(nn.Module):
    """Injects the previous denoising step's prediction into the canvas embeddings.

    Applied before the transformer on all diffusion steps except the first.

    Checkpoint weight shapes (hidden_size=2816, intermediate_size=2112):
      pre_norm   : [2816]          (RMSNorm weight)
      gate_proj  : [2112, 2816]
      up_proj    : [2112, 2816]
      down_proj  : [2816, 2112]
    """

    def __init__(self, config: DiffusionGemmaTextConfig) -> None:
        super().__init__()
        H = config.hidden_size
        I = config.intermediate_size  # 2112
        self.pre_norm = DiffusionGemmaRMSNorm(H, eps=config.rms_norm_eps)
        self.gate_proj = nn.Linear(H, I, bias=False)
        self.up_proj = nn.Linear(H, I, bias=False)
        self.down_proj = nn.Linear(I, H, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-conditioning residual MLP.

        Args:
            x: Canvas embeddings from previous diffusion step [B, canvas_len, H].

        Returns:
            Updated canvas embeddings [B, canvas_len, H].
        """
        h = self.pre_norm(x)
        return x + self.down_proj(
            F.gelu(self.gate_proj(h), approximate="tanh") * self.up_proj(h)
        )


# ---------------------------------------------------------------------------
# Transformer layer
# ---------------------------------------------------------------------------


class DiffusionGemmaLayer(nn.Module):
    """One transformer block with parallel dense MLP + sparse MoE feedforward paths.

    Forward pass:

        residual = x
        h        = input_layernorm(x)
        h        = self_attn(h, ...)
        h        = post_attention_layernorm(h)
        h        = residual + h                       # first residual

        residual = h                                  # NEW residual = post-attn out
        h        = pre_feedforward_layernorm(h)
        h        = mlp(h)
        h1       = post_feedforward_layernorm_1(h)    # dense path

        # MoE path operates on `residual` (the post-attn output):
        routing_input = residual                      # router applies its own norm
        experts_input = pre_feedforward_layernorm_2(residual)
        h2 = moe(routing_input, experts_input)
        h2 = post_feedforward_layernorm_2(h2)

        h   = h1 + h2                                  # combine dense + moe
        h   = post_feedforward_layernorm(h)           # OUTER norm on the SUM
        out = residual + h                            # second residual
        out = out * layer_scalar
    """

    def __init__(self, config: DiffusionGemmaTextConfig, layer_idx: int, is_encoder: bool = True) -> None:
        super().__init__()
        H = config.hidden_size
        eps = config.rms_norm_eps

        self.self_attn = DiffusionGemmaAttention(config, layer_idx, is_encoder=is_encoder)
        self.mlp = DiffusionGemmaDenseMLP(config)
        self.moe = DiffusionGemmaMoEMLP(config)

        # Norms as named in the HF checkpoint.
        self.input_layernorm = DiffusionGemmaRMSNorm(H, eps=eps)
        self.post_attention_layernorm = DiffusionGemmaRMSNorm(H, eps=eps)
        self.pre_feedforward_layernorm = DiffusionGemmaRMSNorm(H, eps=eps)
        self.post_feedforward_layernorm = DiffusionGemmaRMSNorm(H, eps=eps)
        self.pre_feedforward_layernorm_2 = DiffusionGemmaRMSNorm(H, eps=eps)
        self.post_feedforward_layernorm_1 = DiffusionGemmaRMSNorm(H, eps=eps)
        self.post_feedforward_layernorm_2 = DiffusionGemmaRMSNorm(H, eps=eps)

        # Per-layer scalars for encoder vs. decoder mode (separate checkpoint values).
        self.decoder_scalar = nn.Parameter(torch.ones(1))
        self.encoder_scalar = nn.Parameter(torch.ones(1))

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
        encoder_k: torch.Tensor | None = None,
        encoder_v: torch.Tensor | None = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """Compute one transformer block with dual MLP paths."""
        # -- Attention sub-layer ---------------------------------------------
        residual = x
        h = self.input_layernorm(x)
        h = self.self_attn(h, position_ids, cache, encoder_k, encoder_v, is_causal)
        h = self.post_attention_layernorm(h)
        h = residual + h                                       # first residual

        # -- Feedforward sub-layer (dense + MoE in parallel) -----------------
        residual = h                                           # post-attn output
        dense_in = self.pre_feedforward_layernorm(h)
        h1 = self.post_feedforward_layernorm_1(self.mlp(dense_in))   # dense path

        # MoE: router sees the RAW residual, experts see the _2 norm of residual.
        experts_in = self.pre_feedforward_layernorm_2(residual)
        h2 = self.moe(residual, experts_in)
        h2 = self.post_feedforward_layernorm_2(h2)

        h = h1 + h2                                            # combine dense + moe
        h = self.post_feedforward_layernorm(h)                # OUTER norm on the sum
        out = residual + h                                    # second residual

        # Scale output by the mode-appropriate per-layer scalar.
        layer_scalar = self.encoder_scalar if is_causal else self.decoder_scalar
        return out * layer_scalar


# ---------------------------------------------------------------------------
# Shared transformer backbone
# ---------------------------------------------------------------------------


class DiffusionGemmaSharedTransformer(nn.Module):
    """30-layer shared transformer backbone used by both encoder and decoder passes.

    The self_conditioning module lives in DiffusionGemmaDecoderForCoreAI (not here)
    so that the encoder export graph does not carry unused submodules.
    """

    def __init__(self, config: DiffusionGemmaTextConfig, is_encoder: bool = True) -> None:
        super().__init__()
        self.embed_tokens = ScaledEmbedding(
            config.vocab_size, config.hidden_size,
            embed_scale=config.hidden_size ** 0.5,
        )
        self.layers = nn.ModuleList(
            [DiffusionGemmaLayer(config, i, is_encoder=is_encoder) for i in range(config.num_hidden_layers)]
        )
        self.norm = DiffusionGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids_or_embeds: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
        encoder_k_cache: torch.Tensor | None = None,
        encoder_v_cache: torch.Tensor | None = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """Run the full transformer stack.

        Args:
            input_ids_or_embeds: Either integer token ids [B, T] (encoder pass)
                or pre-computed float embeddings [B, T, H] (decoder canvas pass).
            position_ids:        Absolute positions [B, seq_len].
            cache:               KVCache for encoder pass; None for decoder pass.
            encoder_k_cache:     Stacked encoder K slices [n_layers, 1, n_kv, enc_len, hd]
                                 for decoder cross-attention. None during encoder pass.
            encoder_v_cache:     Same shape as encoder_k_cache.
            is_causal:           True = encoder (causal + KV cache),
                                 False = decoder (bidirectional + encoder K/V).

        Returns:
            Normalized hidden states [B, T, hidden_size].
        """
        if input_ids_or_embeds.dtype in (torch.int32, torch.int64):
            h = self.embed_tokens(input_ids_or_embeds)
        else:
            h = input_ids_or_embeds  # decoder canvas pass: already embedded

        n_layers = len(self.layers)
        for i, layer in enumerate(self.layers):
            torch._check_is_size(i)
            torch._check(i < n_layers)
            enc_k = (
                encoder_k_cache.narrow(0, i, 1).squeeze(0)
                if encoder_k_cache is not None
                else None
            )
            enc_v = (
                encoder_v_cache.narrow(0, i, 1).squeeze(0)
                if encoder_v_cache is not None
                else None
            )
            h = layer(h, position_ids, cache, enc_k, enc_v, is_causal)

        return self.norm(h)


# ---------------------------------------------------------------------------
# Encoder export model  (autoregressive, builds KV cache from prompt)
# ---------------------------------------------------------------------------


class DiffusionGemmaEncoderForCoreAI(BaseForCausalLM):
    """Autoregressive encoder pass: processes prompt+image tokens, fills KV cache.

    This is the standard LM prefill/decode model exported for Core AI. Its
    forward pass is causal and uses the encoder_scalar at each layer.

    Exported forward signature:
        (input_ids[B, T], position_ids[B, seq_len],
         k_cache[n_layers, 1, n_kv, max_ctx, hd],
         v_cache[n_layers, 1, n_kv, max_ctx, hd])
        -> logits[B, T, vocab_size]
    """

    _HF_MODEL_CLASS = None

    @override
    def _init_model(self, config: DiffusionGemmaTextConfig) -> None:
        self.model = DiffusionGemmaSharedTransformer(config, is_encoder=True)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        # Expose cache-sizing attributes so KVCache.create_cache_tensors works.
        self.num_hidden_layers = config.num_hidden_layers
        self.num_key_value_heads = config.cache_num_key_value_heads
        self.head_dim = config.cache_head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.vocab_size = config.vocab_size
        self.num_attention_heads = config.num_attention_heads

    @BaseForCausalLM.cast_logits_bfloat16_to_float16
    def forward(
        self,
        input_ids: torch.Tensor,     # [B, T]  int32
        position_ids: torch.IntTensor,  # [B, seq_len]
        k_cache: torch.Tensor,     # [n_layers, 1, n_kv, max_ctx, hd]
        v_cache: torch.Tensor,     # [n_layers, 1, n_kv, max_ctx, hd]
    ) -> torch.Tensor:
        """Encoder forward pass: causal LM with KV cache.

        Returns logits [B, T, vocab_size] with final softcapping applied.
        """
        cache = KVCache(k_cache, v_cache)
        out = self.model(input_ids, position_ids, cache=cache, is_causal=True)
        logits = self.lm_head(out)
        cap = getattr(self.config, "final_logit_softcapping", 30.0)
        if cap and cap > 0.0:
            logits = torch.tanh(logits / cap) * cap
        return logits

    @override
    def _mutate_state_dict(self: Self, state_dict: dict[str, torch.Tensor]) -> None:
        """Remap HF checkpoint keys to our model layout (in-place)."""
        _mutate_diffusion_gemma_state_dict(state_dict, self)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        super().load_state_dict(state_dict, strict=strict, assign=assign)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight


# ---------------------------------------------------------------------------
# Decoder export model  (bidirectional diffusion canvas denoiser)
# ---------------------------------------------------------------------------


class DiffusionGemmaDecoderForCoreAI(BaseForCausalLM):
    """Bidirectional block-diffusion decoder over a fixed 256-token canvas.

    The canvas attends to itself without a causal mask and cross-attends to
    the encoder-populated KV cache. An optional self-conditioning MLP injects
    the previous diffusion step's prediction into the canvas embeddings.

    Exported forward signature:
        (canvas_embeds[B, 256, H], position_ids[B, 256],
         encoder_k[n_layers, 1, n_kv, enc_len, hd],
         encoder_v[n_layers, 1, n_kv, enc_len, hd])
        -> logits[B, 256, vocab_size]
    """

    _HF_MODEL_CLASS = None

    @override
    def _init_model(self, config: DiffusionGemmaTextConfig) -> None:
        self.model = DiffusionGemmaSharedTransformer(config, is_encoder=False)
        # self_conditioning NOT here — exported as DiffusionGemmaSelfConditioningForCoreAI
        # to avoid unused submodules in this graph.
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @BaseForCausalLM.cast_logits_bfloat16_to_float16
    def forward(
        self,
        canvas_embeds: torch.Tensor,     # [B, 256, H]
        position_ids: torch.IntTensor,  # [B, 256]
        encoder_k: torch.Tensor,     # [n_layers, 1, n_kv, enc_len, hd]
        encoder_v: torch.Tensor,     # [n_layers, 1, n_kv, enc_len, hd]
    ) -> torch.Tensor:
        """Decoder canvas forward pass: bidirectional attention over 256 tokens.

        Self-conditioning is a SEPARATE exported model (DiffusionGemmaSelfConditioningForCoreAI).
        The Swift runner applies it to canvas_embeds before calling this on steps > 0.
        """
        out = self.model(
            canvas_embeds,
            position_ids,
            cache=None,
            encoder_k_cache=encoder_k,
            encoder_v_cache=encoder_v,
            is_causal=False,
        )
        logits = self.lm_head(out)
        cap = getattr(self.config, "final_logit_softcapping", 30.0)
        if cap and cap > 0.0:
            logits = torch.tanh(logits / cap) * cap
        return logits

    @override
    def _mutate_state_dict(self: Self, state_dict: dict[str, torch.Tensor]) -> None:
        """Remap HF checkpoint keys to our model layout (in-place)."""
        _mutate_diffusion_gemma_state_dict(state_dict, self)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        super().load_state_dict(state_dict, strict=strict, assign=assign)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight


# Registry alias: the encoder is the standard text-decoder entry point used by
# the model registry / generic export pipeline. The diffusion-specific decoder
# and self-conditioning models are exported via export_diffusiongemma.py.
DiffusionGemmaForCoreAI = DiffusionGemmaEncoderForCoreAI


# ---------------------------------------------------------------------------
# Self-conditioning standalone export model
# ---------------------------------------------------------------------------


class DiffusionGemmaSelfConditioningForCoreAI(nn.Module):
    """Self-conditioning MLP exported as a separate .aimodel.

    The Swift diffusion runner calls this before the decoder on steps > 0
    to inject the previous prediction into canvas embeddings.

    forward(canvas_embeds[B, canvas_len, H]) -> conditioned_embeds[B, canvas_len, H]
    """

    def __init__(self, config: DiffusionGemmaTextConfig) -> None:
        super().__init__()
        self.self_conditioning = DiffusionGemmaSelfConditioning(config)

    def forward(self, canvas_embeds: torch.Tensor) -> torch.Tensor:
        return self.self_conditioning(canvas_embeds)


# ---------------------------------------------------------------------------
# Weight-key mutation helper
# ---------------------------------------------------------------------------


def _mutate_diffusion_gemma_state_dict(
    state_dict: dict[str, torch.Tensor],
    model: DiffusionGemmaEncoderForCoreAI | DiffusionGemmaDecoderForCoreAI,
) -> None:
    """Remap HF checkpoint keys to our model's parameter namespace in-place.

    Key transformations performed:

    1. Top-level renames:
         model.decoder.embed_tokens.weight -> model.embed_tokens.weight
         model.decoder.norm.weight         -> model.norm.weight

    2. Self-conditioning renames:
         model.decoder.self_conditioning.* -> model.self_conditioning.*

    3. Per-layer standard renames (norms, dense MLP, attention projections/norms):
         model.decoder.layers.{i}.*        -> model.layers.{i}.*
       Attention q_proj/k_proj/v_proj/q_norm/k_norm are kept UNFUSED, matching
       the reference. v_norm has no learnable weight (with_scale=False), so there
       is no v_norm weight in the checkpoint. Full-attention layers have no
       v_proj weight (V reuses the raw K projection).

    4. Per-layer scalar renames:
         model.decoder.layers.{i}.layer_scalar
             -> model.layers.{i}.decoder_scalar
         model.encoder.language_model.layers.{i}.layer_scalar
             -> model.layers.{i}.encoder_scalar

    5. MoE expert weight renames -> SwitchGLU SwitchLinear tensors:
         experts.gate_up_proj [E, 2*moe_int, H] -> moe.switch_glu.{gate,up}_proj.weight
         experts.down_proj    [E, H, moe_int]   -> moe.switch_glu.down_proj.weight

    6. Router weight renames:
         router.proj.weight      -> moe.router_proj.weight
         router.scale            -> moe.router_scale
         router.per_expert_scale -> moe.per_expert_scale

    Args:
        state_dict: Mutable dict of HF checkpoint tensors. Modified in-place.
        model:      Initialized encoder or decoder model.
    """
    # 0. Drop weights that belong to other sub-models.
    _has_self_cond = hasattr(model, "self_conditioning")
    for k in list(state_dict.keys()):
        if k.startswith("model.encoder.vision_tower") or k.startswith("model.encoder.embed_vision"):
            del state_dict[k]
        elif k.startswith("model.decoder.self_conditioning.") and not _has_self_cond:
            del state_dict[k]

    # 1. Top-level renames
    _top_renames = [
        ("model.decoder.embed_tokens.weight", "model.embed_tokens.weight"),
        ("model.decoder.norm.weight",         "model.norm.weight"),
    ]
    for src, dst in _top_renames:
        if src in state_dict:
            state_dict[dst] = state_dict.pop(src)

    # 2. Self-conditioning renames (decoder only)
    for k in list(state_dict.keys()):
        if k.startswith("model.decoder.self_conditioning."):
            new_k = k.replace(
                "model.decoder.self_conditioning.",
                "model.self_conditioning.",
            )
            state_dict[new_k] = state_dict.pop(k)

    n_layers = model.config.num_hidden_layers
    for i in range(n_layers):
        hf_pfx = f"model.decoder.layers.{i}"
        our_pfx = f"model.layers.{i}"

        # 3. Standard per-layer weight renames (kept UNFUSED, matching reference).
        _layer_subs = [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
            "pre_feedforward_layernorm_2",
            "post_feedforward_layernorm_1",
            "post_feedforward_layernorm_2",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "self_attn.q_norm",
            "self_attn.k_norm",
        ]
        for sub in _layer_subs:
            old_k = f"{hf_pfx}.{sub}.weight"
            new_k = f"{our_pfx}.{sub}.weight"
            if old_k in state_dict:
                state_dict[new_k] = state_dict.pop(old_k)

        # 4a. Decoder layer scalar
        old_dec = f"{hf_pfx}.layer_scalar"
        if old_dec in state_dict:
            state_dict[f"{our_pfx}.decoder_scalar"] = state_dict.pop(old_dec)

        # 4b. Encoder layer scalar (lives under a different HF prefix)
        old_enc = f"model.encoder.language_model.layers.{i}.layer_scalar"
        if old_enc in state_dict:
            state_dict[f"{our_pfx}.encoder_scalar"] = state_dict.pop(old_enc)

        # 5. MoE expert weights → SwitchGLU SwitchLinear tensors.
        gu_old = f"{hf_pfx}.experts.gate_up_proj"
        dn_old = f"{hf_pfx}.experts.down_proj"
        if gu_old in state_dict:
            gu = state_dict.pop(gu_old)                       # [E, 2*moe_int, H]
            moe_int = gu.shape[1] // 2
            gate_w = gu[:, :moe_int, :].unsqueeze(0).contiguous()  # [1, E, moe_int, H]
            up_w = gu[:, moe_int:, :].unsqueeze(0).contiguous()    # [1, E, moe_int, H]
            state_dict[f"{our_pfx}.moe.switch_glu.gate_proj.weight"] = gate_w
            state_dict[f"{our_pfx}.moe.switch_glu.up_proj.weight"] = up_w
        if dn_old in state_dict:
            dn = state_dict.pop(dn_old)                       # [E, H, moe_int]
            state_dict[f"{our_pfx}.moe.switch_glu.down_proj.weight"] = dn.unsqueeze(0).contiguous()

        # 6. Router weight renames
        rp = f"{hf_pfx}.router.proj.weight"
        rs = f"{hf_pfx}.router.scale"
        re = f"{hf_pfx}.router.per_expert_scale"
        if rp in state_dict:
            state_dict[f"{our_pfx}.moe.router_proj.weight"] = state_dict.pop(rp)
        if rs in state_dict:
            state_dict[f"{our_pfx}.moe.router_scale"] = state_dict.pop(rs)
        if re in state_dict:
            state_dict[f"{our_pfx}.moe.per_expert_scale"] = state_dict.pop(re)


# ---------------------------------------------------------------------------
# Convenience loading functions
# ---------------------------------------------------------------------------


def load_diffusion_gemma_encoder(
    hf_model_id: str,
    target_dtype: torch.dtype = torch.bfloat16,
    max_context_length: int | None = None,
    num_layers: int | None = None,
    mmap_path: str | None = None,
) -> DiffusionGemmaEncoderForCoreAI:
    """Load and return the encoder model with HF checkpoint weights.

    Downloads the checkpoint from the HuggingFace Hub if not already cached,
    remaps the keys, and loads into the encoder export model.

    Args:
        hf_model_id:        HuggingFace model identifier.
        target_dtype:       Weight dtype (default bfloat16).
        max_context_length: Override for max_position_embeddings.
        num_layers:         Truncate to this many layers for smoke tests.
        mmap_path:          If set, move weights to a disk-backed mmap file at
                            this path after loading. Keeps RAM low for the full
                            26B model (the OS pages weights in on demand instead
                            of swapping). Recommended for end-to-end generation.

    Returns:
        DiffusionGemmaEncoderForCoreAI in eval mode, weights loaded.
    """
    from coreai_models.models.base import (
        _build_safetensors_key_index,
        _load_tensors_for_keys,
        _resolve_safetensors_files,
        move_model_to_disk,
    )
    from huggingface_hub import snapshot_download

    full_cfg = DiffusionGemmaConfig.from_pretrained(hf_model_id)
    cfg = full_cfg.text_config
    if max_context_length is not None:
        cfg.max_position_embeddings = max_context_length
    if num_layers is not None:
        cfg.num_hidden_layers = num_layers

    model = DiffusionGemmaEncoderForCoreAI(cfg)
    model.to(dtype=target_dtype)

    model_dir = snapshot_download(
        hf_model_id,
        allow_patterns=["*.safetensors", "*.safetensors.index.json"],
    )
    sf_files = _resolve_safetensors_files(model_dir)
    per_layer, shared = _build_safetensors_key_index(
        sf_files, num_layers=num_layers, hf_state_dict_prefix=""
    )

    sd: dict[str, torch.Tensor] = _load_tensors_for_keys(shared, target_dtype)
    for layer_idx in sorted(per_layer.keys()):
        sd.update(_load_tensors_for_keys(per_layer[layer_idx], target_dtype))

    _mutate_diffusion_gemma_state_dict(sd, model)
    # lm_head.weight is tied to embed_tokens (re-tied below), so it won't be in
    # the checkpoint — load non-strict and re-tie.
    model.load_state_dict(sd, strict=False, assign=True)
    del sd
    import gc
    gc.collect()
    if mmap_path is not None:
        move_model_to_disk(model, path=mmap_path)
    if cfg.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    return model.eval()


def load_diffusion_gemma_decoder(
    hf_model_id: str,
    target_dtype: torch.dtype = torch.bfloat16,
    num_layers: int | None = None,
) -> DiffusionGemmaDecoderForCoreAI:
    """Load and return the canvas denoiser model with HF checkpoint weights.

    Args:
        hf_model_id:  HuggingFace model identifier.
        target_dtype: Weight dtype (default bfloat16).
        num_layers:   Truncate to this many layers for smoke tests.

    Returns:
        DiffusionGemmaDecoderForCoreAI in eval mode, weights loaded.
    """
    from coreai_models.models.base import (
        _build_safetensors_key_index,
        _load_tensors_for_keys,
        _resolve_safetensors_files,
    )
    from huggingface_hub import snapshot_download

    full_cfg = DiffusionGemmaConfig.from_pretrained(hf_model_id)
    cfg = full_cfg.text_config
    if num_layers is not None:
        cfg.num_hidden_layers = num_layers

    model = DiffusionGemmaDecoderForCoreAI(cfg)
    model.to(dtype=target_dtype)

    model_dir = snapshot_download(
        hf_model_id,
        allow_patterns=["*.safetensors", "*.safetensors.index.json"],
    )
    sf_files = _resolve_safetensors_files(model_dir)
    per_layer, shared = _build_safetensors_key_index(
        sf_files, num_layers=num_layers, hf_state_dict_prefix=""
    )

    sd: dict[str, torch.Tensor] = _load_tensors_for_keys(shared, target_dtype)
    for layer_idx in sorted(per_layer.keys()):
        sd.update(_load_tensors_for_keys(per_layer[layer_idx], target_dtype))

    _mutate_diffusion_gemma_state_dict(sd, model)
    # lm_head.weight is tied to embed_tokens (re-tied below) — load non-strict.
    model.load_state_dict(sd, strict=False, assign=True)
    if cfg.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    return model.eval()
