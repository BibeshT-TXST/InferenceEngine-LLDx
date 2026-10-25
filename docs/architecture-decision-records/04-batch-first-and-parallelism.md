# ADR-04: Batch-first tensors and the parallelism model

- **Status:** Accepted
- **Date:** 2026-10-25
- **Module:** #0

## Context
Decode is memory-bound: each step reads every weight to produce one token per sequence. Running several sequences in the same forward pass reuses that one weight read. The harness wants parallelism (subagents exploring several hypotheses over the same logs). Adding a batch dimension to a model that was written without one means touching every layer, every cache index and every test.

## Decision
1. **Batch-first tensors.** Every model tensor carries a leading batch dimension from the first layer module (#11) onward, including when the batch is 1.
2. **Per-sequence cache slots.** KV cache v1 (#16) stores state per sequence, so sequences of different lengths can coexist and be forked later.
3. **Thread-safe handles.** Engine and cache handles are `Send + Sync`, with compile-time assertions in tests.
4. **Parallelism levels, landed in order:**

| Level | Module | Gate |
|---|---|---|
| Prefill: whole prompt in one forward pass | #15 | logits match Transformers |
| Batched decode: N sequences per forward pass | #30a | each sequence's greedy output matches its sequential run |
| Shared-prefix subagents: fork a cached prefix copy-on-write, decode branches concurrently | #30b (needs #26) | no cross-talk between forks under concurrent use |
| CPU/GPU overlap: tokenization, sampling and detokenization on CPU while the GPU runs the next step | #30c | identical output to the non-overlapped path |

5. **Metrics:** aggregate tokens/sec at batch 1/2/4/8, per-sequence latency, and memory saved by prefix sharing.

## Alternatives considered
- **Batch-1 first, batching later:** simpler early layers, but batching then becomes a rewrite of attention, the cache and every gate.

## Consequences
- Batching is never a retrofit. Every gate from #11 on runs with batch > 1 (the reference fixtures use a batch of 2).
- Slightly more shape bookkeeping in early modules.
- Padding and attention masks for unequal sequence lengths arrive with batched decode at #30a. The Phase 0–2 fixtures use equal-length sequences.

## Mitigations
- Shape assertions at layer boundaries, and parity gates against Transformers at batch 2.
