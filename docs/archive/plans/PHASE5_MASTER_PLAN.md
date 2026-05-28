# 🚀 Phase 5: Master Plan - Silver Fetus (10MB High-Fidelity)

**Versión:** 1.0 (2026-05-25)
**Estado:** Activo
**Objetivo:** Evolucionar del *Gold Embryo* (4MB) al *Silver Fetus* (10MB) utilizando soberanía algebraica y destilación por consenso para alcanzar coherencia gramatical total en 2 bits.

## 1. Arquitectura del Sistema (v2.0)
| Parámetro | Valor |
| :--- | :--- |
| **Tamaño en Disco** | ~9.8 MB |
| **Cuantización** | 2-bit Genomic (Algebraic) |
| **Capas** | 12 Bloques Genomic Attention |
| **Dimensión Oculta** | 512 |
| **Cabezas GQA** | 8 |
| **Vocabulario** | 32,768 (Silver Tokenizer) |

## 2. Etapas de Ejecución

### 🏁 Etapa 1: Cimentación Algebraica (Soberanía)
*   **Tarea 1.1:** Ejecutar `scripts/research/generate_algebraic_codebook.py` para generar los centroides basados en el campo ciclotómico $\mathbb{Q}(\zeta_{16})$.
*   **Tarea 1.2:** Modificar el motor de inicialización en Rust (`src/core/model.rs`) para que el "nacimiento" del modelo use estos pesos algebraicos en lugar de inicialización aleatoria.

### 🧪 Etapa 2: Tokenización Silver (Poda de Vocabulario)
*   **Tarea 2.1:** Analizar `data/datasets/dataset_es_ext.txt` para identificar los 32k tokens más relevantes.
*   **Tarea 2.2:** Generar el `silver_tokenizer.json` podando el vocabulario de SmolLM2-135M.

### 🧠 Etapa 3: Council of Teachers (Destilación CoT)
*   **Tarea 3.1:** Configurar el entorno de destilación con dos maestros: `Qwen2-1.5B` y `SmolLM2-135M`.
*   **Tarea 3.2:** Implementar el "Entrenamiento por Resonancia" donde el Estudiante (Silver Fetus) debe predecir no solo el token, sino la topología de activación sugerida por el consenso de los maestros.

### 🏋️ Etapa 4: Crianza Evolutiva (10MB Training)
*   **Tarea 4.1:** Lanzar el entrenamiento masivo sobre el dataset extendido.
*   **Tarea 4.2:** Aplicar el *Relational Bias* de la CAM (Centroid Adjacency Matrix) extraída en la Fase 4.0 para estabilizar la atención.

### ⚖️ Etapa 5: Validación de Alta Fidelidad
*   **Tarea 5.1:** Re-ejecutar el test *Needle in a Haystack*. Meta: > 80% de recuperación.
*   **Tarea 5.2:** Benchmarking de Perplejidad. Meta: PPL < 200 en español.

## 3. Datos Existentes y Progreso Actual
- [x] **Investigación Topológica:** Documentada en `docs/research/UNIT_DISTANCE_TOPOLOGY_INSIGHTS.md`.
- [x] **Generador de Centroides:** `scripts/research/generate_algebraic_codebook.py` ya implementado.
- [x] **Scaffolding Inicial:** Estructura base definida en `docs/plans/SILVER_FETUS_SCAFFOLDING_PLAN.md`.
- [x] **Tokenizador Silver:** Poda a 32k completada.
- [x] **Modelo 10MB:** `SilverFetus-v1.gaje` (12MB) inicializado exitosamente.

---
*Este plan es la hoja de ruta definitiva para superar la barrera de la compresión extrema.*
