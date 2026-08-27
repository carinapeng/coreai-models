# DiffusionGemma — Foundation Models API integration plan

Status: design / not yet implemented.

## Context

DiffusionGemma is a block-diffusion language model: rather than generating tokens
left-to-right, it denoises a fixed-length token *canvas* over several steps, and
produces output block-by-block (an outer autoregressive loop over canvases, with
an inner diffusion loop per canvas). It exports as two Core AI components — a
causal encoder that prefills a KV cache and a bidirectional decoder that denoises
the canvas while cross-attending to that cache.

The shared inference stack (`InferenceEngine` / `EngineFactory` / the
Foundation Models generation surface) is autoregressive-token-streaming oriented:
`generate(input, samplingConfiguration) -> OutputSequence`, where `OutputSequence`
is an `AsyncSequence<InferenceOutput>` terminated by a `StopReason`, and
`EngineFactory` selects an engine variant from the model's structure.

The current DiffusionGemma path is a self-contained runner that loads the two
components by path and drives the diffusion loop directly. This document describes
how to make DiffusionGemma a first-class citizen of the shared stack so that
callers use the standard generation API unchanged.

The design goal: adapt block diffusion to the streaming `InferenceEngine`
contract (streaming committed tokens as each canvas finalizes) rather than
special-casing callers.

## Phase 0 — Bundle schema

- Add a `diffusionLLM` case to `ModelBundle.Kind`, plus a diffusion language config
  parsed from the metadata `diffusion` block (canvas length, max denoising steps,
  entropy bound, temperature bounds, stability/confidence thresholds) and the
  `encoder` / `decoder` component keys.
- Extend the language bundle loader (or add a dedicated diffusion bundle type) so
  the new kind is accepted and `loadTokenizer()`, config access, and component URL
  resolution work through the standard path. This removes the current
  load-by-path handling.

## Phase 1 — Diffusion inference engine

- Add a `DiffusionEngine` conforming to `InferenceEngine` that holds the encoder
  and decoder inference functions and the sampler (entropy-bound accept/renoise,
  linear temperature schedule, self-conditioning).
- Map block diffusion onto `generate() -> OutputSequence`: run the outer
  canvas loop; as each canvas finalizes, yield its committed tokens as
  `InferenceOutput`s, and set the `StopReason` on end-of-sequence or when the
  token budget is reached. Optionally emit per-step argmax canvases as draft
  outputs for responsive UIs.
- Implement the remaining protocol surface: an `InferenceConfiguration` carrying
  the diffusion parameters from metadata, `processedTokenCount`, `reset(to:)`,
  `warmup`, and `cancel` (checked between denoising steps).

## Phase 2 — Factory and model/session wiring

- Add a diffusion variant to `EngineFactory`, auto-selected when the bundle kind
  is `diffusionLLM` (structure-based selection, consistent with the existing
  variant resolution).
- Construct the diffusion engine from the language-model / runner layer for
  diffusion bundles, so the existing session and generation API works unchanged:
  a caller issues a normal generate request and receives streamed tokens without
  needing to know the model is diffusion-based.

## Phase 3 — Sampling configuration mapping

- Define how the autoregressive-oriented `SamplingConfiguration` maps onto
  diffusion parameters: temperature scales the temperature schedule; the maximum
  token count maps to a number of canvases; top-k / top-p / repetition penalty do
  not apply and are documented and either ignored or rejected rather than
  silently mis-applied.

## Phase 4 — Variable-length prompts

The encoder is currently exported with a fixed prompt length (a static-shape
export), because the dynamic-shape encoder graph is not usable through the
current shape-inference path. Supporting arbitrary prompt lengths requires one of:

1. Bucketed static encoders — export the encoder at a set of prompt-length buckets
   and pad to the next bucket at runtime with an attention mask, reusing the
   existing query-length bucketing used by the static-shape engine. Lowest risk.
2. Resolving the dynamic-shape encoder lowering so a single dynamic export can be
   used. Cleaner, but depends on a runtime/compiler fix.

Only the encoder-context length needs bucketing; the decoder canvas length is
fixed by the algorithm.

## Phase 5 — Consolidation and tests

- Once the engine is auto-selected through the factory, remove the standalone
  diffusion path from the CLI runner (which then just resolves an engine and
  calls `generate`); retain the standalone runner only as a debugging aid.
- Tests: engine unit tests with mocked inference functions (a one-hot decoder
  yields deterministic convergence), a session-level integration test, and a
  golden end-to-end generation check.

## Sequencing

- Phase 0 and Phase 4 (option 1) are prerequisites (bundle schema + variable-length
  support).
- Phases 1–2 are the bulk of the work (engine, factory, session wiring).
- Phases 3 and 5 are mapping and consolidation.
- Suggested landing order: bundle schema first (small change), then bucketed
  encoder support, then the engine and wiring as the main change, keeping each
  review scoped.

## Open questions

1. Bucketed static encoders versus resolving the dynamic-shape encoder lowering.
2. Whether to stream per-denoising-step drafts or only finalized blocks.
3. Whether the sampling configuration needs a diffusion-specific subtype or
   field-level mapping is sufficient.
