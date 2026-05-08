# Changelog

## [0.3.0-alpha] - 2026-05-07
### Added
- **Genomic Neural Architecture:** Experimental support for 2-bit weight inference.
- **LUT-ADC Forward Pass:** Ultra-fast linear layer execution in Rust.
- **Deep Stabilization:** Implementation of RMSNorm and RoPE (Rotary Embeddings) for deep signal propagation.
- **GGUF Genomizer:** Script to convert standard LLM models (Qwen2, Llama) to 2-bit DNA format.
- **Genomic Tokenizer:** Frontend for high-fidelity text-to-DNA conversion (98% accuracy).
- **MLP Demo:** Native Multi-Layer Perceptron running on 100% genomic weights.

### Changed
- Refactored `GajeIndex` to use Flat Storage buffer, reducing fragmentation.
- Optimized Python-Rust bindings using `PyBytes` for 16x memory efficiency during ingestion.

### Fixed
- HNSW result heap ordering (Max-Heap vs Min-Heap consistency).
- Out-of-memory crashes in extreme-scale benchmarks (10M+).

- **Phase 4 SBERT Validation**: Successfully implemented and verified high-density vector search (768 dimensions) using real-world text data (Shakespeare).
- **ADC Search Protocol**: Enabled Asymmetric Distance Computation (ADC) in the benchmark suite for cosine-equivalent search in DNA space.
- **Recall@10 Metric**: Integrated structured accuracy reporting for Phase 4, reaching **85.40%** precision.

### Fixed
- **Termux/Android Compatibility**: 
    - Implemented a `scipy` monkeypatch/bypass to resolve `dlopen` symbol errors (`__emutls_get_address`) specific to Python 3.13 on Termux.
    - Switched Ground Truth calculation to `torch`-based cosine similarity to bypass broken `scipy` extension modules.
- **SSL/Connectivity Resilience**: Updated data fetchers to handle network timeouts and certificate issues by allowing fallback patterns and model introspection bypass.
- **Build Process**: Fixed library import path issues by enforcing `pip install .` for system-wide accessibility of the Rust core.

### Changed
- **Benchmark Suite**: Updated `benchmarks/phase4_sbert.py` with improved stability features and support for high-dimensional vector distributions.
- **Codebook Training**: Optimized `train_genomic_codebook` for SBERT manifold shapes.

## [0.1.0] - 2026-05-01
- Initial release of the Genomic Semantic Compression Protocol (GAJE).
- Basic PQ quantization and DNA strand encoding.
