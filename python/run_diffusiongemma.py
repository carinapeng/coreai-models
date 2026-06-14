"""DiffusionGemma runner — export once, then ask prompts via the Core AI runtime.

The proven-faithful path is the in-process coreai.runtime (NOT the Swift
llm-runner GPU-pipelined engine, which diverges). Run in a NORMAL terminal
(the tool sandbox lacks Metal/compiler entitlements).

Usage:
    # 1) Export once (loads 26B weights, 4-bit quantizes, writes a static .aimodel):
    uv run python python/run_diffusiongemma.py --export

    # 2) Ask anything (loads the saved .aimodel; no PyTorch weights needed):
    uv run python python/run_diffusiongemma.py --prompt "What is the capital of France?"
    uv run python python/run_diffusiongemma.py --prompt "Name three primary colors." --max-tokens 24
"""
import argparse, asyncio, sys
import numpy as np
import ml_dtypes
import torch

sys.path.insert(0, "python/src")
import coreai.runtime as R
from pathlib import Path

HF = "google/diffusiongemma-26b-a4b-it"
AIMODEL_DIR = Path("exports/dg_runtime")          # persistent export location
MAXLEN = 48                                        # static graph length (prompt+gen)
DTYPE = torch.bfloat16

# Cache geometry (matches the model config)
NL, NKV, HD = 30, 8, 512
EOS_TOKENS = {1, 106}                              # <eos>, <turn|>


def chat_format(user_prompt: str) -> str:
    return f"<|turn>user\n{user_prompt}<turn|>\n<|turn>model\n"


def do_export() -> None:
    """Load 26B weights, 4-bit quantize, export a static-shape .aimodel."""
    import importlib.util
    from coreai_models.models.macos.diffusion_gemma import load_diffusion_gemma_encoder
    from coreai_models.export.macos import export_to_coreai
    from coreai_models.export.metadata import build_aimodel_metadata
    import shutil

    print("Loading full encoder (bf16)...", flush=True)
    enc = load_diffusion_gemma_encoder(HF, target_dtype=DTYPE, max_context_length=MAXLEN)

    spec = importlib.util.spec_from_file_location("expdg", "python/export_diffusiongemma.py")
    expdg = importlib.util.module_from_spec(spec); spec.loader.exec_module(expdg)
    print("Quantizing (4-bit)...", flush=True)
    enc = expdg._quantize_encoder(enc, HF, "4bit", DTYPE, MAXLEN)

    ids0 = torch.zeros(1, MAXLEN, dtype=torch.int32)
    pos0 = torch.arange(MAXLEN, dtype=torch.int32).unsqueeze(0)
    kc = torch.zeros(NL, 1, NKV, MAXLEN, HD, dtype=DTYPE); vc = torch.zeros_like(kc)
    print("Exporting static graph...", flush=True)
    prog = export_to_coreai(
        enc, {"input_ids": ids0, "position_ids": pos0, "k_cache": kc, "v_cache": vc},
        dynamic_shapes=None, input_names=("input_ids", "position_ids"),
        output_names=("logits",), state_names=("k_cache", "v_cache"),
    )
    if AIMODEL_DIR.exists():
        shutil.rmtree(AIMODEL_DIR)
    AIMODEL_DIR.mkdir(parents=True)
    prog.save_asset(AIMODEL_DIR / "model.aimodel", build_aimodel_metadata("diffusiongemma"))
    # copy tokenizer for the run step (resolve the HF snapshot dir portably)
    from huggingface_hub import snapshot_download
    snap = Path(snapshot_download(
        HF, allow_patterns=["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
    ))
    (AIMODEL_DIR / "tokenizer").mkdir(exist_ok=True)
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        src = snap / f
        if src.exists():
            shutil.copy(src, AIMODEL_DIR / "tokenizer" / f)
    print(f"Exported to {AIMODEL_DIR}/model.aimodel")


def do_generate(user_prompt: str, max_tokens: int) -> None:
    from tokenizers import Tokenizer
    ap = AIMODEL_DIR / "model.aimodel"
    if not ap.exists():
        sys.exit(f"No exported model at {ap}. Run with --export first.")
    tok = Tokenizer.from_file(str(AIMODEL_DIR / "tokenizer" / "tokenizer.json"))
    pids = [2] + tok.encode(chat_format(user_prompt), add_special_tokens=False).ids

    async def run():
        m = await R.AIModel.load(str(ap))
        fnames = m.function_names() if callable(m.function_names) else m.function_names
        fn = m.load_function(fnames[0])
        gen = list(pids)
        out_ids = []
        for _ in range(max_tokens):
            L = len(gen)
            if L > MAXLEN:
                break
            arr = np.zeros((1, MAXLEN), dtype=np.int32); arr[0, :L] = gen
            posa = np.arange(MAXLEN, dtype=np.int32)[None]
            kbf = np.zeros((NL, 1, NKV, MAXLEN, HD), dtype=ml_dtypes.bfloat16); vbf = np.zeros_like(kbf)
            inputs = {"input_ids": R.NDArray(arr), "position_ids": R.NDArray(posa)}
            for n in fn.desc.input_names:
                if n == "k_cache": inputs[n] = R.NDArray(kbf)
                elif n == "v_cache": inputs[n] = R.NDArray(vbf)
            out = await fn(inputs=inputs)
            lg = np.asarray(out["logits"].numpy(), dtype=np.float32)[0]
            nxt = int(lg[L - 1].argmax())
            gen.append(nxt); out_ids.append(nxt)
            if nxt in EOS_TOKENS:
                break
        return out_ids

    out_ids = asyncio.run(run())
    text = tok.decode(out_ids)
    print(f"\nPrompt: {user_prompt}")
    print(f"Answer: {text}")


def main() -> None:
    p = argparse.ArgumentParser(description="DiffusionGemma Core AI runner")
    p.add_argument("--export", action="store_true", help="Export the .aimodel (one-time)")
    p.add_argument("--prompt", type=str, help="Ask the model a question")
    p.add_argument("--max-tokens", type=int, default=32)
    args = p.parse_args()
    if args.export:
        do_export()
    if args.prompt:
        do_generate(args.prompt, args.max_tokens)
    if not args.export and not args.prompt:
        p.error("pass --export and/or --prompt")


if __name__ == "__main__":
    main()
