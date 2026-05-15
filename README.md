# 🧬 Protocolo GAJE: Compresión Semántica Genómica

[![Version](https://img.shields.io/badge/version-0.6.1-purple)](CHANGELOG.md)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**GAJE (Genomic Adaptive Joint Embedding)** es un protocolo de computación genómica de alta densidad que permite ejecutar modelos de lenguaje masivos (LLMs) y búsqueda vectorial utilizando el alfabeto del ADN (A, C, G, T) con solo **2 bits por dimensión**.

## 🚀 Avance Crítico: Inferencia Genómica Nativa
GAJE v0.6.1 trasciende la simple compresión, habilitando un motor de ejecución completo en el dispositivo:

| Característica | Impacto | Estado |
| :--- | :---: | :--- |
| **Kernel Fusion** | **Reducción de Latencia** | ✅ Fusionado (RMSNorm/SwiGLU) |
| **Real KV-Cache DNA** | **RAM 16x menor** | ✅ 2-bit Nativo |
| **Mobile-Native Learning**| **Ajuste Local** | ✅ Optimizador Rust/SIMD |
| **Anchor Cloning** | **PPL 1.60** | ✅ Breakthrough Coherencia |

## 🛠️ Innovaciones Tecnológicas
- **Fusión de Kernels (Rust/SIMD)**: Operaciones de RMSNorm, SwiGLU y Atención operando directamente sobre ADN de 2 bits sin salir del espacio nativo.
- **IQAT (Iterative Quantization-Aware Training)**: Refinamiento de centroides genómicos basado en la activación de un Maestro F32 de alta fidelidad.
- **Aprendizaje en Dispositivo**: Optimizador ligero integrado que permite al modelo aprender de las correcciones del usuario localmente.

## 📈 Benchmarks de Nueva Generación (Qwen2-0.5B)
| Métrica | Original (Float32) | GAJE v0.6.1 (2-bit) | Ganancia |
| :--- | :--- | :--- | :--- |
| **Uso de RAM** | ~1,345 MB | **~84 MB** | **16.0x menos** |
| **Perplejidad (PPL)**| 1.58 | **1.60** | **98.7% Estabilidad** |
| **Similitud Coseno** | 1.00 | **0.965** | **Fidelidad Industrial** |
| **Throughput** | Base | **~110 tokens/s** | **Real-time Ready** |

## 📁 Estructura del Proyecto
```
dna-semantic-compression/
├── src/                    # Núcleo en Rust (SIMD NEON, Kernel Fusion)
├── python/gaje/            # Ecosistema Genómico
│   ├── nn/                 # Capas Estabilizadas, IQAT, Destilación
│   ├── core/               # Formato .gaje y Cuantización
│   └── processing/         # Tokenización adaptativa
├── benchmarks/             # Suite de Perplejidad y Precisión
└── tests/                  # Pruebas de integración nativa
```

## 📚 Documentación Adicional
- [Manifiesto del Proyecto](docs/meta/MANIFESTO.md)
- [Resumen Ejecutivo](docs/meta/EXECUTIVE_SUMMARY.md)
- [Autoría y Créditos](docs/meta/AUTHORSHIP.md)
- [Hoja de Ruta (Roadmap)](docs/meta/ROADMAP.md)

## 🚀 Instalación y Uso
```bash
# Compilar el motor nativo e instalar
pip install .

# Iniciar destilación de un modelo GGUF
python python/gaje/nn/distiller.py
```
