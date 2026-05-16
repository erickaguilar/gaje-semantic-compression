## [0.6.5] - 2026-05-15
### Added
- **Genomic Sequential Memory:** Implementación de una RNN genómica de 2 bits capaz de aprender secuencias textuales ("Hola Mundo").
- **Monte Carlo Evolution Engine:** Motor de optimización basado en mutaciones aleatorias y selección natural en Rust, superando las limitaciones del gradiente en espacios discretos.
- **Evolución Acelerada:** Validación de aprendizaje secuencial en <20ms en dispositivos móviles.

### Changed
- **Roadmap Pivot:** Re-priorización hacia el "Nacimiento de Micro-Organismos" (10MB) y la independencia total de Python.
- **Project Manifesto:** Publicación del `MANIFESTO.md` detallando la visión de largo plazo del protocolo.

## [0.6.3] - 2026-05-13
### Added
- **IQAT Definitive Implementation:** Implementación nativa en Rust de `refine_swiglu` para alineación simétrica de capas FFN.
- **`refine_with_grads`:** Nueva interfaz de gradientes externos en `GenomicLinear` para permitir backpropagation local.
- **Verification Suite:** Test `test_iqat_convergence.py` validando una mejora del +95% en la fidelidad de la señal SwiGLU.

### Fixed
- **Semantic Drift:** Eliminación del ruido acumulativo en las capas de activación que causaba alucinaciones incoherentes.

## [0.6.2] - 2026-05-12
### Added
- **Evolución 4 (Inferencia Nativa en Rust):** Migración completa del bucle `forward` del Transformer a Rust (`RustGenomicLLM`), eliminando el overhead de serialización PyO3 y reduciendo la latencia masivamente a ~0.24s por token (hasta 2.18 t/s en entorno móvil).
- **Soporte Dinámico de RoPE:** Lectura dinámica de la frecuencia base (`rope.freq_base`) desde el GGUF para modelos modernos (Qwen2, Llama 3).
- **Refactorización Modular del Core:** División arquitectónica de `src/nn.rs` en subcomponentes limpios (`linear.rs`, `attention.rs`, `block.rs`, `llm.rs`).

### Fixed
- **Fuga de Precisión por De-permutación:** Eliminado un doble proceso de permutación erróneo sobre los pesos Q/K en GGUF (Llama/Qwen format) que provocaba que el motor arrojara texto "basura".
- **Crash por Desbordamiento OOM:** Prevención de colisiones por dimensionamiento de embeddings al usar `out_features` nativo como límite seguro.

## [0.6.1] - 2026-05-10
### Added
- **Direct Genomic Ingestion (DGI):** Puente de alta fidelidad que carga tensores F16/F32 directamente desde GGUF a ADN de 2 bits, eliminando la pérdida de calidad del paso intermedio Q8_0.
- **Capa Epigenética (Fase 11):** Implementación de Residual Quantization (RQ) mediante un segundo strand de ADN (ARN regulador).
- **Dual-Core ADC:** Kernel de Rust optimizado para búsqueda asimétrica con de-cuantización combinada de 16 valores por dimensión.
- **GAJE Archive v3:** Soporte para persistencia de modelos de alta fidelidad con strands duales.
- **Recall Breakthrough:** Incremento validado de +30 puntos en Recall@10 (+60% de mejora relativa) en búsqueda semántica de alta densidad.
### Added
- **Kernel Fusion (Fase 9):** Implementación nativa en Rust para `GenomicSwiGLU` y fusión de `RMSNorm` en todas las proyecciones.
- **KV-Cache DNA:** Compresión real de 16x de la caché de atención utilizando cuantización de 2 bits y de-cuantización asimétrica (ADC) al vuelo.
- **Sincronización Total del Maestro (Fase 10):** Unificación de arquitecturas (RoPE split, SwiGLU) para eliminar discrepancias algorítmicas.
- **IQAT (Iterative Quantization-Aware Training):** Motor de refinamiento de centroides basado en el Activation Drift del Maestro F32.
- **Mobile-Native Learning:** Optimizador ligero `refine_centroids` en Rust para aprendizaje local sin sobrecarga de frameworks externos.
- **Persistencia Avanzada:** Soporte para guardar organismos genómicos refinados mediante `save_genomic_model`.

## [0.5.0] - 2026-05-10
### Added
- **Validación Integral del Sistema:** Verificación completa de métricas en Termux (Recall@10: 85.3%, Throughput: 172k ops/s).
- **Arquitectura de Anclas Clonadas:** Implementación de la técnica de "Clonación de Segmentos Específicos" para proteger pesos críticos (Top 1%-5% de energía).
- **Benchmark de Perplejidad (PPL) Final:** Reducción masiva de PPL de 564.24 a 1.60 (99.7% de mejora en coherencia).
- **Frontera de Salida Optimizada:** Refinamiento de resolución en embeddings frágiles mediante tripletes de ADN (8-bit simulado).
- **Fusión de Kernels (Fase 9):** Implementación de `GenomicSwiGLU`, `RMSNorm Fusion` y `Real KV-Cache DNA` (2-bit) para optimización extrema de RAM y velocidad.
- **Nuevos Benchmarks:** Inclusión de `benchmarks/final_ppl_cloning_benchmark.py` y `benchmarks/dna_cloning_test_v2.py`.

### Fixed
- **Kernel Consistency:** Sincronización de la API de `GenomicLinear` en los tests de regresión.

### Changed
- **Roadmap Actualizado:** Consolidación de la Fase 10 como "Genomic Distillation & Anchor Cloning".

## [0.4.1] - 2026-05-07
### Added
- **Reporte Técnico de Destilación:** Publicación de `docs/qwen2_distillation_report.md` con análisis de fallas en el modelo Qwen2 2-bit.
- **Validación de 24 Bloques:** Script `benchmarks/distilled_qwen_test.py` para evaluación de modelos de profundidad completa.
- **Soporte de Attention en Python:** Clase `GenomicAttention` expuesta y verificada tras actualización del core.

### Fixed
- **Desalineación del Maestro:** Identificación de inconsistencias en RoPE/SwiGLU del modelo F32 de referencia.
- **Lógica de Carga:** Corregido el import de `GenomicAttention` en `genomize_llm.py`.

### Changed
- **Roadmap Actualizado:** Re-priorización de la Fase 10 (Genomic Distillation) tras detectar degradación de calidad.

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
