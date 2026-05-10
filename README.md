# 🧬 Protocolo GAJE: Compresión Semántica Genómica

[![Version](https://img.shields.io/badge/version-0.5.0-blue)](CHANGELOG.md)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**GAJE (Genomic Adaptive Joint Embedding)** es un protocolo de compresión de alta densidad diseñado para almacenar embeddings vectoriales de gran escala utilizando el alfabeto genómico (A, C, G, T).

## 🚀 Innovación: El "Punto Dulce" de la Compresión
GAJE resuelve el compromiso entre precisión y espacio, superando los estándares industriales en alta dimensionalidad (768d+):

| Método | Recall@10 | Bits/Dim | Eficiencia |
| :--- | :---: | :---: | :--- |
| **Scalar Quant (SQ8)** | 99% | 8.0 | Baja |
| **GAJE Protocol (DNA)** | **85.3%** | **2.0** | **Alta (Ganador)** |
| **Binary Flat (1-bit)** | 64% | 1.0 | Degradada |

## ⚡ Características Principales (v0.5.0)
...
- **Native Rust Core**: Procesamiento paralelo masivo con Rayon y aceleración SIMD (NEON).

## 📁 Estructura del Proyecto
```
dna-semantic-compression/
├── src/                    # Núcleo en Rust (High-Performance)
├── python/gaje/            # Paquete principal de Python
│   ├── core/               # Cuantización y lógica de archivos .gaje
│   ├── nn/                 # Capas genómicas (Linear, Attention) y Modelos
│   ├── processing/         # Tokenización y pipelines de inferencia
│   └── utils/              # Entrenamiento de codebooks y métricas
├── benchmarks/             # Suite de pruebas (Accuracy, Speed, PPL)
├── examples/               # Demos y ejemplos de uso
├── models/                 # Artefactos de modelos genómicos
└── tests/                  # Tests automatizados (Pytest)
```

## 🛠️ Instalación y Uso
```bash
# Compilar el núcleo de Rust e instalar la librería
pip install .

# Ejecutar el demo de búsqueda HNSW
python python/hnsw_demo.py
```

## 📈 Benchmarks Reales
Para ver el análisis comparativo completo contra FAISS y las pruebas de velocidad, consulta [BENCHMARKS.md](benchmarks/BENCHMARKS.md).

---
*Desarrollado para la frontera del almacenamiento de datos en ADN.*
