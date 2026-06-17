# Whisper

Automatic speech recognition (ASR) encoder-decoder model from OpenAI, trained on a large multilingual and multitask supervised dataset.[^1]

## Setup

If you haven't installed `uv`, install it by

```bash
brew install uv
```

## Export

```sh
uv run export.py
```

Saves to `<repo-root>/exports/<model>_<dtype>.aimodel`. Pass `--output-dir <path>` to override the destination.

```sh
uv run export.py --help
```

**Options:**

| Flag           | Description                      | Default                         |
| -------------- | -------------------------------- | ------------------------------- |
| `--model`      | Model variant                    | `openai/whisper-large-v3-turbo` |
| `--output-dir` | Output directory for `.aimodel`  | `<repo-root>/exports/`          |
| `--dtype`      | `float16`, `bfloat16`, `float32` | `float32`                       |
| `--overwrite`  | Overwrite existing `.aimodel`    | —                               |
| `--mode`       | `legacy` or `coreai` (see below) | `legacy`                        |

**Supported models:**

| Model                         | Parameters |
| ----------------------------- | ---------- |
| openai/whisper-large-v3-turbo | 809M       |
| openai/whisper-large-v3       | 1.54B      |

## Export modes

**`--mode legacy`** wraps the full HuggingFace model as a single `.aimodel`. Plain conversion. Encoder and decoder are fused — the encoder re-runs on every decode step.

**`--mode coreai`** exports encoder and decoder as separate `.aimodel` files, with the decoder's self-attention KV cache declared as persistent state. The encoder runs once per audio clip; only the decoder loops per token.

## Performance

`whisper-large-v3-turbo`, float32, 11s audio clip, M5 Max:

**Throughput** — audio seconds processed per wall-clock second.


| Mode | Throughput | RTF |
|---|---|---|
| `legacy` | 0.77× | 1.29 |
| `coreai` | 8.46× | 0.12 |

[^1]: [Paper](https://arxiv.org/abs/2212.04356) · [HuggingFace](https://huggingface.co/openai/whisper-large-v3)
