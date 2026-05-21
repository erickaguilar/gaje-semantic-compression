# 🧬 Protocolo GAJE: Inteligencia Genómica Evolutiva

[![Version](https://img.shields.io/badge/version-0.8.0-purple)](CHANGELOG.md)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**GAJE (Genomic Adaptive Joint Embedding)** es un protocolo de computación genómica de alta densidad que permite ejecutar modelos de lenguaje masivos (LLMs) con solo **2 bits por peso**, utilizando un alfabeto de 4 bases nitrogenadas digitales (A, C, G, T).

## 🚀 Breakthrough v0.8.0: Inferencia Neuromórfica Asíncrona
GAJE v0.8.0 introduce el **Emulador de Spiking Transformer**, un motor de inferencia basado en eventos que simula el comportamiento biológico de las neuronas para procesar contextos masivos con consumo energético mínimo.

| Característica | Impacto | Estado |
| :--- | :---: | :--- |
| **Spiking Transformer** | **Zero-Mult Inferencia** | ✅ 330k eventos/seg |
| **Event-Driven Scheduler**| **1M Context Support** | ✅ Inferencia Asíncrona |
| **Bitwise Evolution** | **Real-time Training** | ✅ 1.00 Fitness (SFA) |
| **Identity Cloner** | **Personalización** | ✅ Clonación de Estilo |

## 🛠️ Innovaciones Tecnológicas
- **Emulador Neuromórfico (Spiking Engine)**: Motor nativo en Rust que utiliza neuronas **Leaky Integrate-and-Fire (LIF)**. Elimina las multiplicaciones de matrices en favor de sumas directas de centroides de 2-bits, permitiendo una eficiencia térmica sin precedentes.
- **Asincronía Basada en Eventos**: Implementación de una cola de prioridad (`BinaryHeap`) que permite al sistema procesar solo la actividad eléctrica relevante (spikes), saltando periodos de inactividad en contextos de hasta 1,000,000 de tokens.
- **Bitwise Evolution (XOR Mutation)**: Motor evolutivo que opera directamente sobre el ADN de 2-bits de los pesos, permitiendo el entrenamiento y ajuste de modelos en milisegundos mediante paralelismo masivo.
- **Path Integral Breeding**: Técnica inspirada en Richard Feynman que evoluciona múltiples poblaciones de pesos en paralelo para restaurar la inteligencia tras la compresión extrema.

## 📈 Benchmarks de Nueva Generación (Neuromorphic Mode)
| Métrica | Inferencia Densa (f16) | GAJE Spiking (2-bit) | Ganancia |
| :--- | :--- | :--- | :--- |
| **Operaciones** | Multiplicación de Matrices | **Sumas de Centroides** | **Consumo ~0 ALU** |
| **Soporte Contexto** | O(N²) Memoria | **O(E) Eventos** | **Contexto 1M+** |
| **Velocidad de Ajuste**| Horas (Gradiente) | **Segundos (Bitwise)** | **>100x Entrenamiento** |
| **Eficiencia Edge** | Alta Carga CPU | **Event-Driven (Idle CPU)**| **Soberanía Energética** |

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
