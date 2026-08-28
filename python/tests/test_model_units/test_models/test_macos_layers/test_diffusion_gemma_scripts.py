# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Unit tests for the DiffusionGemma CLI/driver scripts.

The scripts live at the ``python/`` root (not in the installed package), so they
are loaded by file path. Pure helpers are tested directly; the weight-loading /
export / Core AI runtime paths are exercised with mocks so no 26B checkpoint or
Core AI runtime is required.
"""

import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

import numpy as np
import torch

_PY_ROOT = Path(__file__).resolve().parents[4]


def _load_script(name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, _PY_ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


export_dg = _load_script("export_diffusion_gemma")
validate_dg = _load_script("validate_diffusion_gemma")
runtime_dg = _load_script("run_diffusion_gemma_coreai")


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_resolve_dtype() -> None:
    assert export_dg._resolve_dtype("float16") is torch.float16
    assert export_dg._resolve_dtype("bfloat16") is torch.bfloat16
    assert export_dg._resolve_dtype("float32") is torch.float32


def test_rm_removes_existing_dir_only_when_overwrite() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "asset"
        target.mkdir()
        export_dg._rm(target, overwrite=False)
        assert target.exists()  # not removed without overwrite
        export_dg._rm(target, overwrite=True)
        assert not target.exists()


def test_build_parser_parses_diffusion_args() -> None:
    args = export_dg._build_parser().parse_args(
        ["--model", "m", "--enc-len", "17", "--static-encoder", "--compression", "4bit"]
    )
    assert args.model == "m"
    assert args.enc_len == 17
    assert args.static_encoder is True
    assert args.compression == "4bit"


def test_write_bundle_metadata_contents() -> None:
    from coreai_models.models.macos.diffusion_gemma_config import (
        DiffusionGemmaConfig,
        DiffusionGemmaGenerationConfig,
    )

    full_cfg = DiffusionGemmaConfig()
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp)
        export_dg._write_bundle_metadata(
            bundle,
            "dg",
            "some/model",
            full_cfg.text_config,
            DiffusionGemmaGenerationConfig(),
            full_cfg,
            max_ctx=4096,
            canvas_length=32,
            compression="4bit",
            num_layers=None,
            encoder_only=False,
        )
        meta = json.loads((bundle / "metadata.json").read_text())
    assert meta["kind"] == "diffusion_llm"
    assert meta["assets"] == {"encoder": "encoder.aimodel", "decoder": "decoder.aimodel"}
    assert meta["diffusion"]["canvas_length"] == 32
    assert meta["language"]["vocab_size"] == full_cfg.text_config.vocab_size


def test_write_bundle_metadata_encoder_only_omits_decoder() -> None:
    from coreai_models.models.macos.diffusion_gemma_config import (
        DiffusionGemmaConfig,
        DiffusionGemmaGenerationConfig,
    )

    full_cfg = DiffusionGemmaConfig()
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp)
        export_dg._write_bundle_metadata(
            bundle,
            "dg",
            "some/model",
            full_cfg.text_config,
            DiffusionGemmaGenerationConfig(),
            full_cfg,
            max_ctx=64,
            canvas_length=32,
            compression="none",
            num_layers=2,
            encoder_only=True,
        )
        meta = json.loads((bundle / "metadata.json").read_text())
    assert "decoder" not in meta["assets"]


def test_chat_format_matches_between_scripts() -> None:
    fmt_v = validate_dg.chat_format("hi")
    fmt_r = runtime_dg.chat_format("hi")
    assert fmt_v == fmt_r
    assert "hi" in fmt_v and fmt_v.startswith("<|turn>user")


def test_token_entropy_torch() -> None:
    vocab = 8
    onehot = torch.full((3, vocab), -1e4)
    onehot[:, 0] = 1e4
    uniform = torch.zeros(3, vocab)
    ent_onehot = validate_dg._token_entropy(onehot)
    ent_uniform = validate_dg._token_entropy(uniform)
    assert torch.all(ent_onehot < 1e-2)
    assert torch.allclose(ent_uniform, torch.full((3,), float(np.log(vocab))), atol=1e-4)


def test_entropy_numpy() -> None:
    vocab = 8
    onehot = np.full((3, vocab), -1e4, dtype=np.float32)
    onehot[:, 0] = 1e4
    uniform = np.zeros((3, vocab), dtype=np.float32)
    assert np.all(runtime_dg._entropy(onehot) < 1e-2)
    assert np.allclose(runtime_dg._entropy(uniform), np.log(vocab), atol=1e-4)


# ---------------------------------------------------------------------------
# Export orchestration (mocked; no weights / Core AI compile)
# ---------------------------------------------------------------------------


def test_export_encoder_only_orchestration_mocked() -> None:
    from coreai_models.models.macos.diffusion_gemma import DiffusionGemmaEncoderForCoreAI
    from coreai_models.models.macos.diffusion_gemma_config import (
        DiffusionGemmaConfig,
        DiffusionGemmaGenerationConfig,
        DiffusionGemmaTextConfig,
    )

    tiny = DiffusionGemmaTextConfig(num_hidden_layers=2, vocab_size=100)
    encoder = DiffusionGemmaEncoderForCoreAI(tiny)
    prog = mock.Mock()  # stands in for the exported AIProgram

    with (
        tempfile.TemporaryDirectory() as tmp,
        mock.patch.object(export_dg, "load_diffusion_gemma_encoder", return_value=encoder),
        mock.patch.object(
            export_dg.DiffusionGemmaConfig,
            "from_pretrained",
            return_value=DiffusionGemmaConfig(text_config=tiny),
        ),
        mock.patch.object(
            export_dg.DiffusionGemmaGenerationConfig,
            "from_pretrained",
            return_value=DiffusionGemmaGenerationConfig(),
        ),
        mock.patch.object(export_dg, "export_macos_model", return_value=prog),
        mock.patch.object(export_dg, "build_aimodel_metadata", return_value={}),
        mock.patch.object(export_dg, "_save_tokenizer"),
    ):
        out = export_dg.export_diffusion_gemma(
            "some/model", output_dir=tmp, output_name="dg", encoder_only=True, num_layers=2
        )
        assert (Path(out) / "metadata.json").exists()
        prog.save_asset.assert_called_once()


# ---------------------------------------------------------------------------
# Runtime block-diffusion loop (mocked Core AI runtime + tokenizer)
# ---------------------------------------------------------------------------


class _FakeNDArray:
    def __init__(self, array):
        self._a = np.asarray(array)

    def numpy(self):
        return self._a


class _FakeFunction:
    """Fake inference function. The encoder is state-based (the script reads the
    KV cache back from the state NDArrays it passed in), so it returns nothing;
    the decoder emits one-hot logits on a fixed target sequence so the sampler
    converges deterministically."""

    def __init__(self, kind, target, vocab, hidden):
        self.kind = kind
        self.target = target
        self.vocab = vocab
        self.hidden = hidden

    async def __call__(self, inputs, state=None):
        if self.kind == "encoder":
            return {}  # cache is surfaced via the mutated `state` NDArrays
        canvas = inputs["decoder_input_ids"].numpy().shape[1]
        logits = np.full((1, canvas, self.vocab), -1e4, dtype=np.float32)
        for i in range(canvas):
            logits[0, i, self.target[i % len(self.target)]] = 1e4
        soft = np.zeros((1, canvas, self.hidden), dtype=np.float16)
        return {"logits": _FakeNDArray(logits), "soft_embeds": _FakeNDArray(soft)}


class _FakeModel:
    def __init__(self, fn):
        self._fn = fn
        self.function_names = ["main"]

    @classmethod
    async def _make(cls, fn):
        return cls(fn)

    def load_function(self, _name):
        return self._fn


class _FakeTokenizer:
    def __init__(self, vocab):
        self._vocab = vocab

    def encode(self, text, add_special_tokens=False):
        return types.SimpleNamespace(ids=[5, 6, 7])

    def get_vocab_size(self):
        return self._vocab

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


def test_runtime_diffusion_loop_mocked(capsys=None) -> None:

    vocab, hidden = 12, runtime_dg.__dict__.get("HIDDEN", 2816)
    target = [3, 4, 5, 0, 0, 0, 0, 0]

    enc_fn = _FakeFunction("encoder", target, vocab, hidden)
    dec_fn = _FakeFunction("decoder", target, vocab, hidden)
    models = iter([_FakeModel(enc_fn), _FakeModel(dec_fn)])

    async def fake_load(_path):
        return next(models)

    fake_runtime = types.ModuleType("coreai.runtime")
    fake_runtime.NDArray = _FakeNDArray
    fake_runtime.AIModel = types.SimpleNamespace(load=fake_load)

    fake_tokenizers = types.ModuleType("tokenizers")
    fake_tokenizers.Tokenizer = types.SimpleNamespace(from_file=lambda _p: _FakeTokenizer(vocab))

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.snapshot_download = lambda *a, **k: tempfile.mkdtemp()

    import asyncio as _aio

    import coreai

    with (
        mock.patch.object(coreai, "runtime", fake_runtime),
        mock.patch.dict(
            sys.modules,
            {
                "coreai.runtime": fake_runtime,
                "tokenizers": fake_tokenizers,
                "huggingface_hub": fake_hub,
            },
        ),
    ):
        # runtime hidden size is hardcoded in the script; keep the decoder soft
        # shape consistent with it.
        dec_fn.hidden = 2816
        _aio.run(runtime_dg.run("enc.aimodel", "dec.aimodel", "hello", canvas_len=8, max_steps=2))
    # No assertion on stdout content (decode is a fake); reaching here means the
    # encoder prefill, cache slicing, decoder loop, entropy/accept/renoise, and
    # trim/decode all executed without error.


def _tiny_text_config():
    from coreai_models.models.macos.diffusion_gemma_config import DiffusionGemmaTextConfig

    return DiffusionGemmaTextConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=8,
        global_head_dim=16,
        num_key_value_heads=2,
        num_global_key_value_heads=1,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_experts=4,
        top_k_experts=2,
        vocab_size=40,
        max_position_embeddings=64,
        sliding_window=8,
        layer_types=["sliding_attention", "full_attention"],
    )


class _FakeTok:
    def encode(self, text, add_special_tokens=False):
        return types.SimpleNamespace(ids=[5, 6, 7])

    def get_vocab_size(self):
        return 40

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


def test_validate_run_ar_mocked() -> None:
    from coreai_models.models.macos.diffusion_gemma import DiffusionGemmaEncoderForCoreAI

    enc = DiffusionGemmaEncoderForCoreAI(_tiny_text_config()).eval()
    with (
        mock.patch.object(validate_dg, "_tokenizer", return_value=_FakeTok()),
        mock.patch.object(validate_dg, "load_diffusion_gemma_encoder", return_value=enc),
    ):
        validate_dg.run_ar("hello", max_tokens=3, num_layers=2, dtype=torch.float32)


def test_validate_run_diffusion_and_load_shared_mocked() -> None:
    import coreai_models.models.macos.diffusion_gemma as dgm
    from coreai_models.models.macos.diffusion_gemma_config import DiffusionGemmaConfig

    full = DiffusionGemmaConfig(text_config=_tiny_text_config())
    gen_cfg = validate_dg.DiffusionGemmaGenerationConfig(canvas_length=8, max_denoising_steps=2)
    with (
        mock.patch.object(validate_dg, "_tokenizer", return_value=_FakeTok()),
        mock.patch.object(DiffusionGemmaConfig, "from_pretrained", return_value=full),
        mock.patch.object(dgm, "_load_state_dict_from_hub", return_value={}),
    ):
        # Exercises the real _load_shared_encoder_decoder + the diffusion loop.
        validate_dg.run_diffusion("hello", num_layers=2, dtype=torch.float32, gen_cfg=gen_cfg)


def test_validate_main_dispatches() -> None:
    with (
        mock.patch.object(validate_dg, "run_ar") as ar,
        mock.patch.object(sys, "argv", ["v", "--mode", "ar"]),
    ):
        validate_dg.main()
    ar.assert_called_once()
    with (
        mock.patch.object(validate_dg, "run_diffusion") as diff,
        mock.patch.object(sys, "argv", ["v", "--mode", "diffusion"]),
    ):
        validate_dg.main()
    diff.assert_called_once()


def test_export_full_orchestration_mocked() -> None:
    from coreai_models.models.macos.diffusion_gemma import (
        DiffusionGemmaDecoderForCoreAI,
        DiffusionGemmaEncoderForCoreAI,
    )
    from coreai_models.models.macos.diffusion_gemma_config import (
        DiffusionGemmaConfig,
        DiffusionGemmaGenerationConfig,
    )

    tiny = _tiny_text_config()
    enc = DiffusionGemmaEncoderForCoreAI(tiny)
    dec = DiffusionGemmaDecoderForCoreAI(tiny)
    prog = mock.Mock()
    with (
        tempfile.TemporaryDirectory() as tmp,
        mock.patch.object(export_dg, "load_diffusion_gemma_encoder", return_value=enc),
        mock.patch.object(export_dg, "load_diffusion_gemma_decoder", return_value=dec),
        mock.patch.object(
            export_dg.DiffusionGemmaConfig,
            "from_pretrained",
            return_value=DiffusionGemmaConfig(text_config=tiny),
        ),
        mock.patch.object(
            export_dg.DiffusionGemmaGenerationConfig,
            "from_pretrained",
            return_value=DiffusionGemmaGenerationConfig(),
        ),
        mock.patch.object(export_dg, "export_macos_model", return_value=prog),
        mock.patch.object(export_dg, "export_to_coreai", return_value=prog),
        mock.patch.object(export_dg, "build_aimodel_metadata", return_value={}),
        mock.patch.object(export_dg, "_quantize_encoder", side_effect=lambda e, *a: e),
        mock.patch.object(export_dg, "_quantize_decoder", side_effect=lambda d, *a: d),
        mock.patch.object(export_dg, "_save_tokenizer"),
    ):
        out = export_dg.export_diffusion_gemma(
            "some/model",
            output_dir=tmp,
            output_name="dg",
            compression="4bit",
            canvas_length=8,
            enc_len=4,
            num_layers=2,
            static_encoder=True,
        )
        meta = json.loads((Path(out) / "metadata.json").read_text())
    assert meta["assets"] == {"encoder": "encoder.aimodel", "decoder": "decoder.aimodel"}


def test_export_save_tokenizer_fallback_copies_files() -> None:
    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
        (Path(src) / "tokenizer.json").write_text("{}")
        fake_transformers = types.ModuleType("transformers")

        class _AutoTok:
            @staticmethod
            def from_pretrained(_m):
                raise ValueError("list has no keys")  # trigger the fallback

        fake_transformers.AutoTokenizer = _AutoTok
        fake_hub = types.ModuleType("huggingface_hub")
        fake_hub.snapshot_download = lambda *a, **k: src
        with mock.patch.dict(
            sys.modules, {"transformers": fake_transformers, "huggingface_hub": fake_hub}
        ):
            export_dg._save_tokenizer("some/model", Path(dst))
        assert (Path(dst) / "tokenizer.json").exists()


def test_export_main_mocked() -> None:
    with (
        mock.patch.object(export_dg, "export_diffusion_gemma", return_value="/tmp/out") as ex,
        mock.patch.object(sys, "argv", ["e", "--model", "m", "--num-layers", "2"]),
    ):
        export_dg.main()
    ex.assert_called_once()


def test_runtime_main_mocked() -> None:
    with (
        mock.patch.object(runtime_dg.asyncio, "run") as arun,
        mock.patch.object(sys, "argv", ["r", "--encoder", "e", "--decoder", "d"]),
    ):
        runtime_dg.main()
    arun.assert_called_once()


def test_export_quantize_encoder_decoder_mocked() -> None:
    import coreai_models.export.compression as compression
    import coreai_models.export.presets as presets
    from coreai_models.models.macos.diffusion_gemma import (
        DiffusionGemmaDecoderForCoreAI,
        DiffusionGemmaEncoderForCoreAI,
    )

    tiny = _tiny_text_config()
    enc = DiffusionGemmaEncoderForCoreAI(tiny)
    dec = DiffusionGemmaDecoderForCoreAI(tiny)
    canvas, enc_len, n_kv, hd = 8, 4, tiny.cache_num_key_value_heads, tiny.cache_head_dim
    dec_inputs = {
        "decoder_input_ids": torch.zeros(1, canvas, dtype=torch.int32),
        "prev_soft_embeds": torch.zeros(1, canvas, tiny.hidden_size),
        "position_ids": torch.arange(canvas, dtype=torch.int32).unsqueeze(0),
        "encoder_k": torch.zeros(tiny.num_hidden_layers, 1, n_kv, enc_len, hd),
        "encoder_v": torch.zeros(tiny.num_hidden_layers, 1, n_kv, enc_len, hd),
        "temperature": torch.tensor([0.8]),
    }
    with (
        mock.patch.object(presets, "get_preset", return_value={"torch_quantization_config": {}}),
        mock.patch.object(
            compression, "quantize_pytorch_model", side_effect=lambda m, *a, **k: m
        ) as quant,
    ):
        assert export_dg._quantize_encoder(enc, "4bit", torch.float32) is enc
        assert export_dg._quantize_decoder(dec, "4bit", dec_inputs) is dec

    # Both calls must pass the calibration-contract kwargs the quantizer now requires.
    enc_kwargs = quant.call_args_list[0].kwargs
    assert set(enc_kwargs) >= {"cache_seq_len", "state_indices"}
    assert enc_kwargs["state_indices"] == (2, 3)
    dec_kwargs = quant.call_args_list[1].kwargs
    assert dec_kwargs["state_indices"] == ()
    assert dec_kwargs["cache_seq_len"] == 0

    # Preset without a torch_quantization_config -> quantization is skipped.
    with mock.patch.object(presets, "get_preset", return_value={}):
        assert export_dg._quantize_encoder(enc, "weird", torch.float32) is enc
        assert export_dg._quantize_decoder(dec, "weird", dec_inputs) is dec


def test_export_save_tokenizer_success_path() -> None:
    with tempfile.TemporaryDirectory() as dst:
        saved = {}

        class _Tok:
            def save_pretrained(self, path):
                saved["path"] = path

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda _m: _Tok())
        with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
            export_dg._save_tokenizer("some/model", Path(dst))
        assert saved["path"] == dst


_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


if __name__ == "__main__":
    for fn in _TESTS:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(_TESTS)} tests passed")
