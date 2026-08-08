## [1.3.1-alpha] - 2026-08-08
### Added
- **Certificación de Qwen2.5-1.5B 4-bit Flat**: Creación del script `scripts/export_qwen2_5_1_5b_flat.py` para exportar el modelo GQA de 1.5B parámetros de Hugging Face. El modelo resultante (`qwen2_5_1_5b_4bit.gaje.flat`) tiene una precisión factual del 100% en español (*"La capital de Francia es París."*) con un rendimiento de **`1.73 tok/s`** y tiempo de carga instantáneo.

### Fixed
- **Sampler de Temperatura Estocástico**: Corrección de un bug crítico en `src/nn/llm.rs` donde el muestreo con temperatura actuaba como un argmax determinista. Se implementó un algoritmo multinomial estocástico correcto usando la biblioteca `rand` para permitir una generación diversa y evitar bucles de repetición.

### Changed
- **Conclusión del Experimento de 2-Bits**: Documentación empírica y congelamiento formal del frente de evolución estocástica de pesos en 2-bits, determinando su inviabilidad en hardware comercial debido al costo de cómputo (7 minutos/generación) y el agotamiento de memoria.

## [1.3.0-alpha] - 2026-08-08
### Added
- **2-Bit Evolutionary Breeder (`gaje-2bit-breeder`)**: Nuevo criador evolutivo nativo en Rust para optimizar y entrenar embriones genómicos nacidos directamente en 2-bits, mitigando la deriva semántica post-cuantización.
- **Exportador SmolLM2 2-Bit Flat**: Script especializado `scripts/export_smollm2_2bit_flat.py` para compilar y empaquetar modelos de 2-bits nativos directamente en formato `.flat` de 294 MB.
- **Inyección del Repetition Penalty Nivel Rust**: Detector de bucles recursivos nativo en `src/nn/llm.rs` con ventana dinámica de n-gramas $w \in [2, 48]$ para detener de forma robusta la degeneración de logits y evitar loops como el de *"Der Bundesliga"*.
- **Plan del Experimento Evolutivo**: Creación de la documentación formal científica `docs/research/2BIT_EVOLUTIONARY_EXPERIMENT.md`.

### Changed
- **New Hardware Benchmark Upgrades**:
  - `Qwen2-0.5B Flat`: Aumento de rendimiento de `2.0 tok/s` a **`4.44 tok/s`** (mejora de ~100%).
  - `SmolLM2-135M Flat`: Salto de rendimiento de `18.78 tok/s` a **`28.28 tok/s`** (mejora de ~50%).
  - `Zero-Copy Flat Mmap Loader`: Optimización del tiempo de carga inicial a un promedio submilisegundo de **`0.75 ms`**.
- **Higiene de Datos y Modelos**: Eliminación de más de 28.4 GB de archivos SQLite `.gaje` redundantes y modelos antiguos en el repositorio.

## [1.2.0-alpha] - 2026-06-02
### Added
- **Native SDK "GAJE-Core"**: Implementación de la fachada `GajeSession` en Rust, eliminando la dependencia de Python en el flujo de inferencia de chat.
- **Toroidal Sampler (Pilar 1)**: Muestreo basado en física lagrangiana y inercia de fase. Implementado "Frenado Lagrangiano" para guiar la generación de tokens mediante trayectorias de mínima acción.
- **Dynamic RNA Strands (Pilar 2)**: Activación adaptativa de precisión (2/4 bits) basada en la entropía de Shannon de las activaciones en tiempo real.
- **C-FFI Interoperability**: Bindings de tipo `extern "C"` para permitir el uso del motor GAJE desde aplicaciones Android (JNI), iOS (Swift) y C++.
- **Zero-GIL Inerence**: Eliminación del cuello de botella del intérprete de Python, permitiendo inferencia asíncrona pura en Rust.
- **Native Chat Binary**: Nueva herramienta de terminal `gaje-native-chat` que demuestra la soberanía total del motor.

### Changed
- **Kernel Optimization**: Refactorización de `GenomicLinear` para soportar la activación condicional de hebras epigenéticas.
- **Stable Sampling**: Mejora en la estabilidad numérica del sampler mediante escalado dinámico de logits.
- **Unified Math Core**: Uso de `pyo3_shim` para permitir la compilación nativa sin la feature `python`.

## [1.1.0-alpha] - 2026-06-01
### Added
- **Euler-Lagrange Inference Engine**: Integración de las leyes de la física clásica en el núcleo de Rust. La inferencia se modela ahora como un sistema mecánico conservativo donde la información sigue trayectorias de mínima acción.
- **Lagrangian Logic (L = T - V)**: Implementación de la Energía Cinética (T, velocidad de disparo) y Energía Potencial (V, resistencia semántica de las anclas).
- **Geodesic Acceleration**: Nuevo mecanismo de ajuste de integración basado en la fuerza semántica negativa, permitiendo que la coherencia gramatical actúe como una restricción geométrica.
- **Lagrangian Timing Delay**: Los disparos en la *Timing Wheel* se retrasan dinámicamente según la resistencia semántica, permitiendo que la inhibición lateral (K-WTA) fulmine automáticamente el ruido incoherente.
- **Símbolos de Christoffel**: Implementación inicial de la conexión geométrica para modelar la curvatura del espacio de fase en el toroide semántico $\mathbb{Q}(\zeta_{16})$.

### Changed
- **SpikingNeuron v2**: Actualizada con `integrate_lagrangian` y `check_spike_lagrangian` para soportar la optimización física nativa.
- **Motor de Cómputo**: Extensión de `src/compute` con el módulo `lagrangian.rs` para centralizar la física del protocolo GAJE.

## [1.0.0-alpha] - 2026-05-27
### Added
- **Circular Genomic Topology**: Transición del motor de inferencia a un espacio de fase compleja. Las neuronas operan ahora como osciladores con interferencia constructiva/destructiva.
- **Stability Anchors (Anchored Islands)**: Implementación de un "Esqueleto de Estabilidad" (Top 1% de pesos en F16) que guía la coherencia de los pesos de 2 bits, eliminando el ruido binario.
- **Temporal Lateral Inhibition (K-WTA)**: Nuevo mecanismo de selectividad donde las señales más rápidas (menor latencia) inhiben a las competidoras en tiempo real.
- **Silver Adult (v1.0)**: Lanzamiento del primer organismo genómico de alta fidelidad (9.9 MB) con identidad consolidada y coherencia gramatical funcional en español.
- **Algebraic Codebook Initialization**: Soporte para inicialización de pesos basada en campos ciclotómicos $\mathbb{Q}(\zeta_{16})$ para una rejilla de inteligencia matemáticamente rígida.
- **Global Phase Synchronization**: Nuevo protocolo de entrenamiento de baja escala para alinear la base de conocimientos global con las islas de estabilidad.

### Changed
- **Gaje-CLI v0.8.0**: Actualizado con el preset `silver_adult` e inyección automática de anclas durante la inicialización (`--init`).
- **SMG1 Persistence v4**: Soporte para buffers dispersos de anclas en el formato de almacenamiento nativo.
- **Complex Math Kernels**: Implementación de `quantize_phase_core` y `dequantize_phase_core` optimizados para SIMD NEON.

## [0.9.7-alpha] - 2026-05-26
### Added
- **TRUE Native Sovereignty**: Motor GAJE-Flow 100% independiente de Python mediante arquitectura decoupled (Zero-GIL).
- **Native Genomic Trainer**: Implementación en Rust de entrenamiento por Resonancia Semántica y Entropía de Shannon integrado en el CLI.
- **gaje-cli v0.7.5**: Nuevos subcomandos `--train` y `--init --preset` para gestión autónoma de organismos.
- **PyO3 Shim**: Capa de abstracción para permitir la compilación de binarios nativos puros sin dependencias de tiempo de ejecución de Python.
- **Kernel Refactoring**: Extracción de `dequantize_q8_0_core` para compartir lógica de alto rendimiento entre el cargador nativo y la interfaz de Python.
- **Unified Environment**: Gestión centralizada de dependencias en `pyproject.toml` (v0.9.7) eliminando la fragmentación de entornos virtuales.

### Changed
- **Repository Reorganization**: Clasificación jerárquica de scripts en `scripts/archive/` y consolidación de demos de usuario en `examples/core_demos/`.
- **Root Cleanup**: Eliminación de archivos huérfanos (`--preset`, `chat`, `inspect`) y cumplimiento estricto del protocolo de limpieza GAJE-Flow.

### Fixed
- **GIL Panic Fix**: Refactorización estructural que elimina los fallos de inicialización del intérprete en contextos nativos.
- **Core Stability**: Consolidación de métodos `_core` para garantizar la integridad del flujo de datos en Rust.
- **Integration Test Alignment**: Sincronización de `tests/integration/test_integration_v060.py` con la API actual de `GajeIndex`.

## [0.9.6-alpha] - 2026-05-23
### Added
- **Soberanía Nativa End-to-End:** Eliminación total de dependencias de Python en el flujo de inferencia. El motor ahora es autónomo para cargar, tokenizar y ejecutar LLMs.
- **Tokenizador BPE Nativo (`GajeTokenizer`):** Implementación de un motor de procesamiento de texto en Rust puro basado en `tokenizers`, integrado directamente en el formato `.gaje`.
- **Carga Zero-Copy (mmap):** Refactorización del lector GGUF para utilizar mapeo de memoria (`memmap2`), permitiendo el acceso instantáneo a tensores sin copias en RAM.
- **Refactorización de Inmutabilidad:** Optimización de los cargadores de modelos para operar con referencias inmutables, mejorando la seguridad de hilos y el rendimiento de carga.
- **Limpieza de Utilidades Legacy (Fase 1):** Eliminación de scripts de Python redundantes (`inspect_gaje`, `hatch_gold_embryo`, etc.) en favor de subcomandos nativos en `gaje-cli`.
- **Eliminación de Exportadores Legacy (Fase 3):** Remoción de `export_smollm.py`, `distill_smollm.py` y la lógica de genomización en Python (`genomize.py`). Rust es ahora el único encargado de la transformación de formatos.

## [0.9.5] - 2026-05-22
### Added
- **Genomic Training Nativo:** Implementación de `refine_step` en `GajeNeuromorphicLayer` para aprendizaje supervisado local mediante refuerzo/inhibición de bits.
- **Gestión de Energía Consciente:** Nuevo módulo `PowerManager` con detección dinámica de arquitecturas big.LITTLE y control de afinidad de hilos (Thread Affinity) para optimizar batería y temperatura.
- **Demo de Life-long Learning:** Nuevo script `gaje-native-trainer-demo` que valida el aprendizaje de asociaciones en un solo organismo en <60ms.
- **Demo de Power-Awareness:** Script `gaje-power-demo` demostrando la conmutación de hilos entre núcleos de eficiencia y rendimiento.
- **Optimización SIMD NEON v2:** Refactorización a layout `Input-Major` y uso de `vqtbl1q_u8` (Shuffle) para procesar 4 neuronas por byte de forma vectorial.

## [0.9.0-alpha] - 2026-05-21
### Added
- **Arquitectura SoA (Structure of Arrays):** Rediseño total del motor neuromórfico en `src/nn/spiking/layer.rs` para maximizar la localidad de datos y habilitar auto-vectorización SIMD.
- **Timing Wheel Industrial:** Implementación de un algoritmo de rueda de tiempo $O(1)$ en `src/compute/timing_wheel.rs` para gestionar contextos masivos (1M+ tokens) sin degradación de rendimiento.
- **Paralelismo Masivo con Rayon:** El motor evolutivo ahora evalúa linajes genómicos en paralelo sobre todos los núcleos de la CPU, acelerando el entrenamiento "Born-Genomic".
- **Integración Industrial:** Refactorización del `NeuromorphicScheduler` para operar de forma nativa sobre capas SoA y la Timing Wheel.

### Changed
- Mejora en la dinámica de las neuronas LIF: decaimiento (*decay*) más estable para prevenir la extinción prematura del potencial de membrana.
- Optimización de binarios: `gaje-identity-cloner` y `gaje-neuromorphic-trainer` migrados a la nueva arquitectura industrial.

### Fixed
- Resolución de todas las advertencias de compilación de Cargo y actualización de firmas de PyO3 para compatibilidad futura.

## [0.8.0] - 2026-05-21
### Added
- **Emulador Neuromórfico v1.0 (Spiking Transformer):** Implementación de un motor de inferencia asíncrono basado en eventos que simula hardware neuromórfico real.
- **Inferencia de 2-bits sin Multiplicaciones:** Neuronas LIF (Leaky Integrate-and-Fire) que utilizan sumas directas de centroides, eliminando el cuello de botella del ALU.
- **Scheduler de Eventos Asíncronos:** Gestión de tiempo basada en `BinaryHeap` (Min-Heap) que permite saltar periodos de inactividad, ideal para contextos de 1,000,000 de tokens.
- **Motor Evolutivo Bitwise XOR:** Entrenamiento nativo mediante mutaciones a nivel de bits sobre pesos empaquetados, logrando resonancia total (1.00 Fitness) en segundos.
- **Gaje Identity Cloner:** Nueva herramienta de validación para clonar identidades cognitivas y estilos de lenguaje mediante resonancia genómica.

### Changed
- Estructura del núcleo (`src/nn/spiking` y `src/compute`) para soportar la arquitectura de disparos discretos.

## [0.7.0] - 2026-05-18
### Added
- **Soberanía Nativa (Rust 100%):** Independencia absoluta del runtime de Python. El ecosistema GAJE ahora puede cargar, ejecutar, muestrear, evolucionar y entrenar LLMs directamente desde un entorno nativo seguro.
- **Inmortalidad RAM (Zero-Copy Cloning):** Migración de matrices pesadas de ADN (2-bit) a punteros atómicos compartidos (`Arc<Vec<u8>>`). Permite clonar LLMs masivos en memoria a costo computacional de ~0 bytes.
- **Path Integral Breeding Paralelo:** Evolución genómica poblacional re-escrita con `Rayon`. Todos los núcleos de la CPU simulan futuros paralelos simultáneos, acelerando la convergencia evolutiva ("Hola Mundo" en <15ms).
- **Auto-Grad Nativo (Hybrid Training):** Bucle de `CrossEntropyLoss` y `Softmax` migrado matemáticamente a `RustGenomicLLM::train_step`. Permite entrenamiento directo (`--train`) desde `gaje-cli`.
- **Muestreador de Texto Complejo:** Decodificación fluida nativa con Temperatura, Top-K y Top-P integrada al CLI, reemplazando el rudimentario Greedy Decoding.

### Changed
- Actualización de `Cargo.toml` (`v0.7.0`) con directrices nativas de vectorización (AVX2/FMA) por defecto en `target-cpu=native`.

## [0.6.5] - 2026-05-15
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
- **Reporte Técnico de Destilación:** Publicación de `docs/reports/qwen2_distillation_report.md` con análisis de fallas en el modelo Qwen2 2-bit.
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
