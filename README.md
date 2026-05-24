# 🧬 Protocolo GAJE: Inteligencia Genómica Evolutiva

[![Version](https://img.shields.io/badge/version-0.9.6--alpha-purple)](CHANGELOG.md)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**GAJE (Genomic Adaptive Joint Embedding)** es un protocolo de computación genómica de alta densidad que permite ejecutar modelos de lenguaje masivos (LLMs) con solo **2 bits por peso**, utilizando un alfabeto de 4 bases nitrogenadas digitales (A, C, G, T).

## 🚀 Soberanía Nativa v0.9.6-alpha
GAJE v0.9.6-alpha alcanza la **Soberanía de Extremo a Extremo**, eliminando completamente la dependencia de Python en el flujo de inferencia. El motor ahora es un sistema autónomo, ultra-eficiente y portátil.

| Característica | Impacto | Estado |
| :--- | :---: | :--- |
| **Soberanía Nativa** | **Independencia de Python** | ✅ **Alcanzada** |
| **Tokenizador Nativo** | **Procesamiento de Texto Local** | ✅ Implementado |
| **Carga Zero-Copy (mmap)**| **Carga Instantánea** | ✅ Implementado |
| **Arquitectura SoA** | **Optimización SIMD/Caché** | ✅ Implementado |

## 🛠️ Innovaciones Tecnológicas
- **Soberanía Total (Rust 100%)**: El ecosistema GAJE es ahora un binario único que integra cargador, tokenizador BPE nativo y motor de inferencia, sin dependencias externas.
- **Carga Zero-Copy (memmap2)**: Uso de archivos mapeados en memoria para acceder a los tensores GGUF instantáneamente, eliminando el overhead de copia en la RAM.
- **Diseño SoA (Structure of Arrays)**: Los datos de las neuronas se almacenan en vectores planos contiguos, permitiendo que la CPU procese múltiples potenciales de membrana en un solo ciclo (NEON/AVX2).
- **Timing Wheel Industrial**: Buffer circular para la gestión de eventos neuromórficos con costo estrictamente constante $O(1)$, ideal para contextos masivos de 1M+ tokens.

## 📈 Benchmarks Industriales
| Métrica | Inferencia Densa (f16) | GAJE Industrial (2-bit) | Ganancia |
| :--- | :--- | :--- | :--- |
| **Localidad de Datos** | Dispersa | **SoA Contigua** | **Max Cache Hit** |
| **Gestión Eventos** | O(log N) Heap | **O(1) Timing Wheel** | **Contexto 1M+** |
| **Throughput** | Limitado por ALU | **>1.1M eventos/seg** | **Escalabilidad SIMD** |
| **Entrenamiento** | Horas (Gradiente) | **Segundos (Parallel XOR)**| **>100x Velocidad** |

## 📁 Estructura del Proyecto
```
dna-semantic-compression/
├── src/                    # Núcleo Rust (Spiking Engine, Scheduler, Kernels)
├── src/nn/spiking/         # Arquitectura Neuromórfica (LIF, Attention, FFN)
├── src/compute/            # Gestión de Eventos y Programación Asíncrona
├── src/bin/                # CLI, Identity Cloner y Entrenadores Nativos
├── docs/plans/             # Planes de Destilación y Entrenamiento Nativo
└── docs/reports/           # Reportes de Resonancia y Benchmarks
```

## 📚 Documentación Adicional
- [**Emulador Neuromórfico**](docs/research/SPIKING_NEUROMORPHIC_EMULATOR.md): La ciencia detrás de los spikes.
- [**Plan de Entrenamiento Nativo**](docs/plans/NATIVE_GAJE_TRAINING_PLAN.md): El fin de la dependencia de Python.
- [**Visión Estratégica v0.8.0**](docs/plans/NEUROMORPHIC_STRATEGIC_VISION.md): El futuro de la IA de borde.
- [**Reporte de Resonancia**](docs/reports/NEUROMORPHIC_RESONANCE_TEST_20260521.md): Validación del 1.00 de fitness.

## 🚀 Instalación y Ejecución (Modo Neuromórfico)
```bash
# Ejecutar el benchmark neuromórfico
cargo test nn::spiking::benchmark -- --nocapture

# Ejecutar el clonador de identidad (Identity Cloner)
cargo run --bin gaje-identity-cloner
```
