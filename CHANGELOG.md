# Changelog

## [0.4.0-alpha] - 2026-05-07
### Added
- **Advanced Validation Suite:** Integrated JSD (Jensen-Shannon), Top-k Overlap, and Activation Drift metrics for deep semantic monitoring.
- **Genomic Attention Kernel (Rust):** High-performance implementation of Multi-Head Attention with native **GQA (Grouped-Query Attention)** and **RoPE** support.
- **Internal KV-Cache:** 16x compressed key-value storage directly in Rust for massive context efficiency.
- **Block-Quant Implementation:** Row-wise adaptive quantization for near-lossless signal reconstruction at 2-bit density.
- **Repetition Penalty:** Native sampling mechanism to prevent autoregressive loops in genomic space.
- **End-to-End Chat Loop:** First successful text generation using 100% 2-bit genomic weights (Qwen2).

### Changed
- Refactored `src/lib.rs` for modularity and SIMD-ready attention score calculation.
- Improved GGUF ingestion with native Q8_0 de-quantization layer.

## [0.3.0-alpha] - 2026-05-07

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
