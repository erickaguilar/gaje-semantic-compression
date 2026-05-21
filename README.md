# 🧬 Protocolo GAJE: Inteligencia Genómica Evolutiva

[![Version](https://img.shields.io/badge/version-0.9.0--alpha-purple)](CHANGELOG.md)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**GAJE (Genomic Adaptive Joint Embedding)** es un protocolo de computación genómica de alta densidad que permite ejecutar modelos de lenguaje masivos (LLMs) con solo **2 bits por peso**, utilizando un alfabeto de 4 bases nitrogenadas digitales (A, C, G, T).

## 🚀 Industrialización v0.9.0-alpha: Alto Rendimiento
GAJE v0.9.0-alpha introduce la arquitectura **SoA (Structure of Arrays)** y el algoritmo de **Timing Wheel**, transformando el prototipo en un motor industrial capaz de procesar contextos masivos con eficiencia SIMD.

| Característica | Impacto | Estado |
| :--- | :---: | :--- |
| **Arquitectura SoA** | **Optimización SIMD/Caché** | ✅ Implementado |
| **Timing Wheel O(1)** | **1M+ Context Support** | ✅ Implementado |
| **Rayon Parallelism** | **Massive Training** | ✅ Implementado |
| **Zero-Mult Engine** | **Eficiencia Energética** | ✅ Validado |

## 🛠️ Innovaciones Tecnológicas
- **Diseño SoA (Structure of Arrays)**: Los datos de las neuronas se almacenan en vectores planos contiguos, eliminando la dispersión de memoria y permitiendo que la CPU procese múltiples potenciales de membrana en un solo ciclo (AVX2/NEON).
- **Timing Wheel Industrial**: Buffer circular para la gestión de eventos neuromórficos con costo estrictamente constante $O(1)$, eliminando el overhead de las colas de prioridad tradicionales.
- **Bitwise Evolution (Rayon)**: Motor evolutivo paralelo que opera directamente sobre el ADN de 2-bits, permitiendo el entrenamiento de modelos en milisegundos mediante paralelismo masivo.
- **Emulador Neuromórfico**: Motor nativo en Rust que utiliza neuronas **Leaky Integrate-and-Fire (LIF)** para eliminar las multiplicaciones en favor de sumas directas de centroides.

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
