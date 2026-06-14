"""
dataclasses for DiffusionGemma model configuration.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Vision encoder config

@dataclass
class DiffusionGemmaVisionConfig:
    hidden_size: int = 1152
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 72
    global_head_dim: int = 72
    intermediate_size: int = 4304
    rms_norm_eps: float = 1e-6
    patch_size: int = 16
    pooling_kernel_size: int = 3
    default_output_length: int = 280
    max_position_embeddings: int = 131072
    position_embedding_size: int = 10240
    hidden_activation: str = "gelu_pytorch_tanh"
    attention_bias: bool = False
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    standardize: bool = True
    use_clipped_linears: bool = False
    rope_parameters: Dict[str, Any] = field(
        default_factory=lambda: {
            "rope_theta": 100.0,
            "rope_type": "default",
        }
    )
    dtype: str = "bfloat16"
    model_type: str = "gemma4_vision"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DiffusionGemmaVisionConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

# Text / language-model config

def _default_layer_types() -> List[str]:
    # 30-layer pattern from the 26B checkpoint:
    # 5x sliding + 1x full, repeated 5 times
    pattern = ["sliding_attention"] * 5 + ["full_attention"]
    return pattern * 5


@dataclass
class DiffusionGemmaTextConfig:
    hidden_size: int = 2816
    num_hidden_layers: int = 30
    num_attention_heads: int = 16
    head_dim: int = 256
    global_head_dim: int = 512
    num_key_value_heads: int = 8
    num_global_key_value_heads: int = 2
    intermediate_size: int = 2112
    moe_intermediate_size: int = 704
    num_experts: int = 128
    top_k_experts: int = 8
    vocab_size: int = 262144
    max_position_embeddings: int = 262144
    sliding_window: int = 1024
    rms_norm_eps: float = 1e-6
    final_logit_softcapping: float = 30.0
    hidden_activation: str = "gelu_pytorch_tanh"
    attention_bias: bool = False
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    bos_token_id: int = 2
    eos_token_id: int = 1
    pad_token_id: int = 0
    tie_word_embeddings: bool = True
    use_bidirectional_attention: str = "vision"
    layer_types: List[str] = field(default_factory=_default_layer_types)
    rope_parameters: Dict[str, Any] = field(
        default_factory=lambda: {
            "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1000000.0,
                "rope_type": "proportional",
            },
            "sliding_attention": {
                "rope_theta": 10000.0,
                "rope_type": "default",
            },
        }
    )
    dtype: str = "bfloat16"
    model_type: str = "diffusion_gemma_text"

    def is_full_attention_layer(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == "full_attention"

    @property
    def cache_head_dim(self) -> int:
        """Max head_dim across all layer types — used to size the unified KV cache."""
        return max(self.head_dim, self.global_head_dim)

    @property
    def cache_num_key_value_heads(self) -> int:
        """Max KV heads across all layer types — used to size the unified KV cache."""
        return max(self.num_key_value_heads, self.num_global_key_value_heads)

    @property
    def rope_sliding(self):
        return self.rope_parameters.get("sliding_attention", {})

    @property
    def rope_full(self):
        return self.rope_parameters.get("full_attention", {})

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DiffusionGemmaTextConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


# Top-level model config

@dataclass
class DiffusionGemmaConfig:
    text_config: DiffusionGemmaTextConfig = field(
        default_factory=DiffusionGemmaTextConfig
    )
    vision_config: DiffusionGemmaVisionConfig = field(
        default_factory=DiffusionGemmaVisionConfig
    )
    canvas_length: int = 256
    vision_soft_tokens_per_image: int = 280
    image_token_id: int = 258880
    boi_token_id: int = 255999
    eoi_token_id: int = 258882
    eos_token_id: List[int] = field(default_factory=lambda: [1, 106])
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02
    dtype: str = "bfloat16"
    model_type: str = "diffusion_gemma"

    # Constructors
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DiffusionGemmaConfig":
        text_cfg = DiffusionGemmaTextConfig.from_dict(d.get("text_config", {}))
        vision_cfg = DiffusionGemmaVisionConfig.from_dict(d.get("vision_config", {}))

        top_level_known = {
            "canvas_length",
            "vision_soft_tokens_per_image",
            "image_token_id",
            "boi_token_id",
            "eoi_token_id",
            "eos_token_id",
            "tie_word_embeddings",
            "initializer_range",
            "dtype",
            "model_type",
        }
        top_kwargs: Dict[str, Any] = {
            k: v for k, v in d.items() if k in top_level_known
        }
        return cls(text_config=text_cfg, vision_config=vision_cfg, **top_kwargs)

    @classmethod
    def from_local(cls, config_path: str) -> "DiffusionGemmaConfig":
        """Load from a local config.json file path."""
        with open(config_path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        return cls.from_dict(d)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        cache_dir: Optional[str] = None,
    ) -> "DiffusionGemmaConfig":
        """Download (or reuse cached) config.json from the Hub and parse it."""
        from huggingface_hub import snapshot_download  # lazy import

        model_dir = snapshot_download(
            model_id,
            allow_patterns=["config.json"],
            cache_dir=cache_dir,
        )
        config_path = os.path.join(model_dir, "config.json")
        return cls.from_local(config_path)


# Generation / denoising config

@dataclass
class DiffusionGemmaGenerationConfig:
    """Hyperparameters that govern the masked-diffusion denoising loop."""

    max_denoising_steps: int = 48
    canvas_length: int = 256
    confidence_threshold: float = 0.005
    stability_threshold: int = 1
    t_max: float = 0.8
    t_min: float = 0.4
    entropy_bound: float = 0.1
    # image_token_id doubles as the MASK token during diffusion
    mask_token_id: int = 258880

    @classmethod
    def from_pretrained(cls, model_id: str, cache_dir: Optional[str] = None) -> "DiffusionGemmaGenerationConfig":
        """Parse generation_config.json from the Hub (or cache)."""
        from huggingface_hub import hf_hub_download  # lazy import

        try:
            path = hf_hub_download(model_id, "generation_config.json", cache_dir=cache_dir)
        except Exception:  # noqa: BLE001 — fall back to defaults if absent
            return cls()
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        sampler = d.get("sampler_config", {}) or {}
        return cls(
            max_denoising_steps=d.get("max_denoising_steps", 48),
            canvas_length=d.get("canvas_length", 256),
            confidence_threshold=d.get("confidence_threshold", 0.005),
            stability_threshold=d.get("stability_threshold", 1),
            t_max=d.get("t_max", 0.8),
            t_min=d.get("t_min", 0.4),
            entropy_bound=sampler.get("entropy_bound", 0.1),
        )
