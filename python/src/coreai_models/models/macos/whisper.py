# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import Self

from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.sdpa import SDPA

# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class _EncoderAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.sdpa = SDPA(is_causal=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        n_heads, head_dim = self.n_heads, self.head_dim
        q = self.q_proj(x).view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
        out = self.sdpa(q, k, v).transpose(1, 2).reshape(batch_size, seq_len, n_heads * head_dim)
        return self.out_proj(out)


class _EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.self_attn = _EncoderAttention(d_model, n_heads)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ffn_dim, bias=True)
        self.fc2 = nn.Linear(ffn_dim, d_model, bias=True)
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.self_attn(self.self_attn_layer_norm(x))
        x = x + r
        r = self.fc2(F.gelu(self.fc1(self.final_layer_norm(x))))
        return x + r


class WhisperEncoder(nn.Module):
    def __init__(
        self: Self,
        num_mel_bins: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ffn_dim: int,
        max_source_positions: int,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(num_mel_bins, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(max_source_positions, d_model)
        self.layers = nn.ModuleList(
            [_EncoderLayer(d_model, n_heads, ffn_dim) for _ in range(n_layers)]
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self: Self, input_features: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.conv1(input_features))
        h = F.gelu(self.conv2(h))
        h = h.transpose(1, 2)
        h = h + self.embed_positions.weight
        for layer in self.layers:
            h = layer(h)
        return self.layer_norm(h)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class _DecoderSelfAttention(nn.Module):
    def __init__(self: Self, d_model: int, n_heads: int, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.sdpa = SDPA(is_causal=True)

    def forward(
        self: Self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        cache: KVCache | None,
    ) -> torch.Tensor:
        batch_size, query_len, _ = x.shape
        n_heads, head_dim = self.n_heads, self.head_dim
        seq_len = position_ids.shape[-1]
        torch._check_is_size(query_len)
        torch._check_is_size(seq_len)
        offset = seq_len - query_len
        torch._check_is_size(offset)
        q = self.q_proj(x).view(batch_size, query_len, n_heads, head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, query_len, n_heads, head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, query_len, n_heads, head_dim).transpose(1, 2)
        if cache is not None:
            k, v = cache.update_and_fetch(
                self.layer_idx, offset, k, v, seq_len=seq_len, query_len=query_len
            )
        out = self.sdpa(q, k, v).transpose(1, 2).reshape(batch_size, query_len, n_heads * head_dim)
        return self.out_proj(out)


class _DecoderCrossAttention(nn.Module):
    def __init__(self: Self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.sdpa = SDPA(is_causal=False)

    def forward(
        self: Self, x: torch.Tensor, encoder_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        batch_size, query_len, _ = x.shape
        enc_len = encoder_hidden_states.shape[1]
        n_heads, head_dim = self.n_heads, self.head_dim
        q = self.q_proj(x).view(batch_size, query_len, n_heads, head_dim).transpose(1, 2)
        k = (
            self.k_proj(encoder_hidden_states)
            .view(batch_size, enc_len, n_heads, head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(encoder_hidden_states)
            .view(batch_size, enc_len, n_heads, head_dim)
            .transpose(1, 2)
        )
        out = self.sdpa(q, k, v).transpose(1, 2).reshape(batch_size, query_len, n_heads * head_dim)
        return self.out_proj(out)


class _DecoderLayer(nn.Module):
    def __init__(self: Self, d_model: int, n_heads: int, ffn_dim: int, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = _DecoderSelfAttention(d_model, n_heads, layer_idx)
        self.encoder_attn = _DecoderCrossAttention(d_model, n_heads)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.encoder_attn_layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ffn_dim, bias=True)
        self.fc2 = nn.Linear(ffn_dim, d_model, bias=True)
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(
        self: Self,
        x: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cache: KVCache | None,
    ) -> torch.Tensor:
        r = self.self_attn(self.self_attn_layer_norm(x), position_ids, cache)
        x = x + r
        r = self.encoder_attn(self.encoder_attn_layer_norm(x), encoder_hidden_states)
        x = x + r
        r = self.fc2(F.gelu(self.fc1(self.final_layer_norm(x))))
        return x + r


class WhisperDecoder(nn.Module):
    def __init__(
        self: Self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ffn_dim: int,
        max_target_positions: int,
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.embed_positions = nn.Embedding(max_target_positions, d_model)
        self.layers = nn.ModuleList(
            [_DecoderLayer(d_model, n_heads, ffn_dim, layer_idx=i) for i in range(n_layers)]
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self: Self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        cache = KVCache(k_cache, v_cache)
        batch_size, query_len = input_ids.shape
        seq_len = position_ids.shape[-1]
        torch._check_is_size(query_len)
        torch._check_is_size(seq_len)
        offset = seq_len - query_len
        torch._check_is_size(offset)

        token_embeds = self.embed_tokens(input_ids)
        current_pos = position_ids.narrow(-1, offset, query_len)
        pos_embeds = self.embed_positions(current_pos.long())
        h = token_embeds + pos_embeds

        for layer in self.layers:
            h = layer(h, encoder_hidden_states, position_ids, cache)

        h = self.layer_norm(h)
        return F.linear(h, self.embed_tokens.weight)
