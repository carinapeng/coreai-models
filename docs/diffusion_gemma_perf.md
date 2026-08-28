# DiffusionGemma — performance notes

Status: preliminary single-run measurements. Not a rigorous benchmark (cold
load, no warmup averaging); intended as a first data point and a methodology
reference.

## Setup

- Model: DiffusionGemma-26B-A4B, 4-bit weight quantization (encoder + decoder).
- Path: Swift `llm-runner` block-diffusion runner, Core AI runtime, GPU backend
  (the encoder/decoder currently fall back to GPU; not running on the ANE).
- Bundle: static encoder at a fixed 17-token prompt length, decoder canvas
  length 32, 24 denoising steps.
- Prompt: a 17-token chat-formatted question.
- Apple silicon development machine; numbers are cold-cache, single run.

## Measurements

| Phase | Measured |
| --- | --- |
| Cold model load (encoder + decoder, ~27 GB on disk) | 38.9 s (one-time) |
| Encoder prefill (17 tokens) | 412 ms |
| Decoder denoising step (steady state) | ~155 ms/step |
| Decoder denoising step (first step, includes warmup) | 426 ms |
| Full decode (23 steps) | 3.88 s (169 ms/step average) |
| End-to-end generation, excluding load | ~4.3 s |

## Cost model

Generation cost is:

```
prefill + denoising_steps × (one full decoder forward over the canvas)
```

Each denoising step is a complete 30-layer forward with the 128-expert top-8 MoE
and the dense MLP, run over the full 32-token canvas plus cross-attention to the
encoder KV cache. Unlike autoregressive decoding (one KV-cached token per
forward), block diffusion runs many full-canvas forwards per generated block.

For the short answer measured here (~8 content tokens over ~24 forwards) this is
roughly 0.35 useful tokens per forward. The ratio improves for longer answers
that fill more of the canvas, and with early stopping (the answer stabilized
several steps before the final step under the fixed schedule).

## Caveats

- Cold load dominates wall-clock; weights load from disk uncached.
- Runs on GPU via fallback; an ANE-targeted path is not yet available (the ANE
  compile step fails and falls back).
- Single run, no warmup averaging; step-to-step variance is not characterized.
- 4-bit weights; higher-precision numbers are not measured (full precision is
  impractical to export at this size).

## Comparison and ceiling — methodology

Two follow-ups to turn these into comparative and ceiling-relative numbers:

1. Backbone throughput reference. The decoder forward is a Gemma-4-26B-A4B
   forward over the canvas; comparing its per-forward latency against the same
   backbone run through another framework (e.g. an MLX Gemma-4 forward over an
   equal token count) gives a per-forward reference. Per-token throughput is not
   directly comparable across autoregressive and block-diffusion generation.

2. Roofline ceiling from op shapes. The decoder's per-step op list (attention,
   cross-attention, dense MLP, MoE gather-matmul, 4-bit dequantization) is a
   fixed graph. Running that op list through the from-shapes GPU performance
   model yields a per-step GPU ceiling; scaling by `steps × canvases + prefill`
   gives a full-generation ceiling, and the measured 155 ms/step versus that
   ceiling quantifies the achieved-vs-ceiling gap. The MoE gather-matmul and the
   4-bit dequantization are expected to dominate.
