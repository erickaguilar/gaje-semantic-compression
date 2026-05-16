# 🧬 Protocolo GAJE: Inteligencia Genómica Evolutiva

[![Version](https://img.shields.io/badge/version-0.6.5-purple)](CHANGELOG.md)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**GAJE (Genomic Adaptive Joint Embedding)** es un protocolo de computación genómica de alta densidad que permite ejecutar modelos de lenguaje masivos (LLMs) con solo **2 bits por peso**, utilizando un alfabeto de 4 bases nitrogenadas digitales (A, C, G, T).

## 🚀 Breakthrough v0.6.5: Crianza por Integración de Caminos
GAJE v0.6.5 marca el fin de la era de la compresión pasiva y el inicio de la **Inteligencia Evolutiva Local**. Hemos superado la barrera de fidelidad de los 2 bits mediante la fusión de múltiples historias evolutivas (Path Integrals).

| Característica | Impacto | Estado |
| :--- | :---: | :--- |
| **Path Integral Breeding** | **Coherencia Cuántica** | ✅ Fidelidad > 84% |
| **Native GGUF Ingestor** | **Zero-Python Loading** | ✅ Carga en < 12s |
| **Sequential Memory** | **Lógica Temporal** | ✅ Validación "Hola Mundo" |
| **Monte Carlo Engine** | **Gradiente Discreto** | ✅ Optimización sin Derivadas |

## 🛠️ Innovaciones Tecnológicas
- **Path Integral Breeding (Crianza Poblacional)**: Técnica inspirada en Richard Feynman que evoluciona múltiples poblaciones de pesos en paralelo, integrando los caminos más exitosos para restaurar la inteligencia tras la compresión extrema.
- **Native GGUF Ingestion**: Parser binario 100% Rust que lee y genomiza modelos GGUF directamente del disco, eliminando el overhead de Python y permitiendo el uso en dispositivos con RAM mínima.
- **Monte Carlo Optimization**: Motor de mutación y selección natural que permite entrenar modelos en espacios de pesos discretos donde el descenso de gradiente tradicional (Backpropagation) falla.

## 📈 Benchmarks de Nueva Generación (SmolLM2-135M)
| Métrica | Original (Float16) | GAJE v0.6.5 (2-bit) | Ganancia |
| :--- | :--- | :--- | :--- |
| **Tamaño en Disco** | ~270 MB | **~37 MB** | **7.3x Compresión** |
| **Prob. Token Promedio**| > 99% | **~84% (Criado)** | **Coherencia Real** |
| **Tiempo de Carga** | ~45s (Python) | **~11s (Rust)** | **4x más rápido** |
| **Crianza de Memoria** | N/A | **18ms** | **Evolución Instantánea** |

## 📁 Estructura del Proyecto
```
dna-semantic-compression/
├── src/                    # Núcleo Rust (Monte Carlo, GGUF Parser, Kernels)
├── src/bin/                # CLI Evolutivo y Micro-organismos
├── docs/                   # Hallazgos, Manifiesto y Roadmap
├── scripts/                # Herramientas de investigación y simulación
└── tests/                  # Validación de integridad binaria
```

## 📚 Documentación Adicional
- [**Manifiesto del Proyecto**](docs/meta/MANIFESTO.md): Nuestra visión de la inteligencia ligera.
- [**Memoria Secuencial Genómica**](docs/GENOMIC_SEQUENTIAL_MEMORY.md): El hito del nacimiento desde cero.
- [**Hoja de Ruta (Roadmap)**](docs/meta/ROADMAP.md): El camino hacia los modelos de 10 MB.

## 🚀 Instalación y Ejecución (CLI Nativo)
```bash
# Compilar el runtime nativo
cargo build --release --bin gaje-cli

# Ejecutar inferencia desde un GGUF (Genomización al vuelo)
./target/release/gaje-cli <modelo.gguf> --prompt "Hola"

# Iniciar Crianza Evolutiva por Integración de Caminos
./target/release/gaje-cli <modelo.gguf> --evolve "Objetivo de coherencia" --gens 300
```
