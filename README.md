# InferenceEngine-LLDx

A custom on-device inference engine for Apple Silicon, built to power the LLDx harness.

## Stack
- **Language:** Rust
- **GPU:** Apple Silicon via Metal
- **Tensor library:** TBD (Candle / MLX bindings / raw Metal)
- **Kernels:** Metal Shading Language

## Components
1. Model loading
2. Tokenization
3. Forward pass
4. Generation
5. Sampling
6. KV cache
7. Constrained decoding
8. Memory and context management
9. Quantization
10. Apple Silicon GPU execution
11. Kernels
12. Parallelism
13. Speculative decoding *(stretch goal)*
14. Correctness
15. Profiling
16. Benchmarks and observability

## Roadmap
1. **Engine:** loading, tokenization, forward pass, generation, sampling, KV cache, correctness tests
2. **Optimization:** quantization, Metal kernels, profiling, benchmarks
3. **Harness:** LLDx read-only tools, log intelligence, agent loop *(separate repo that imports this engine)*

