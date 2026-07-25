# 🧬 Fase 5.0: Scaffolding del Feto de Plata (10 MB High-Fidelity)

**Estado:** En Planificación (Pivot Estratégico)
**Objetivo:** Inicializar una arquitectura base desde cero que integre los hallazgos de la Topología Algebraica y resuelva la inestabilidad del modelo de 4MB.

## 1. Justificación del Reinicio (Clean Slate)
Tras validar que la compresión extrema a 4MB genera un *Semantic Drift* inmanejable para tareas de coherencia compleja, la Fase 5.0 propone una arquitectura de **10 MB** diseñada con **Soberanía Algebraica**. Partir desde cero elimina el ruido acumulado en experimentos previos y garantiza una simetría perfecta entre el vocabulario, las dimensiones y la profundidad del modelo.

**Referencia Matemática:**
[Remarks on the Disproof of the Unit Distance Conjecture (OpenAI)](https://cdn.openai.com/pdf/74c24085-19b0-4534-9c90-465b8e29ad73/unit-distance-remarks.pdf) - Este paper justifica el uso de campos de números algebraicos para maximizar la densidad relacional en espacios de alta dimensión.

## 2. Especificaciones de la Nueva Estructura (v2.0)
*   **Nombre en Clave:** Silver Fetus (Feto de Plata)
*   **Capacidad:** 12.5 Millones de Parámetros (2-bit).
*   **Embedding (`n_embd`):** 512.
*   **Profundidad:** 12 Bloques (Transformers Genómicos).
*   **Atención:** 8 Cabezas (GQA Nativo).
*   **Vocabulario:** 32,768 (Tokenizador Silver).
*   **Meta de Tamaño:** **9.8 MB** (incluyendo el 5% de Anclas de Oro).

## 3. Pilares de Implementación

### A. Inicialización Algebraica (OpenAI Insight)
A diferencia de las versiones anteriores, los centroides NO serán aleatorios.
*   **Mecánica:** Los 4 estados de los 2 bits se mapearán a raíces de la unidad de un campo ciclotómico específico.
*   **Impacto:** El modelo nace con un "esqueleto" de distancias semánticas pre-calculadas, reduciendo la entropía inicial y acelerando la convergencia.

### B. Simetría de Capas (Torres de Campos)
Cada bloque Transformer se inicializará como una extensión relacional del anterior.
*   **Mecánica:** Aplicación de factores de escala basados en el discriminante de la raíz para mantener el flujo de señal estable a través de los 12 bloques.

### C. Tokenizador Silver (BPE Optimizado)
*   **Mecánica:** Poda del vocabulario de SmolLM2/Qwen2 para extraer solo los 32k tokens con mayor frecuencia en el `dataset_es_ext.txt`.

## 4. Hoja de Ruta Inmediata

1.  **[Tooling]** Crear `scripts/research/generate_algebraic_codebook.py` para calcular los centroides basados en campos CM.
2.  **[Rust Core]** Actualizar `src/io/loader.rs` y `init_born_genomic_model` para soportar la arquitectura v2.0.
3.  **[Nacimiento]** Ejecutar el comando de inicialización para generar `SilverFetus-v1.gaje`.
4.  **[Validación]** Test de "Balbuceo Algebraico" para confirmar que la PPL inicial es más baja que en el Gold Embryo.

---
*Este documento marca el inicio de la Fase 5.0, sustituyendo el enfoque de optimización por el de construcción arquitectónica de alta fidelidad.*
