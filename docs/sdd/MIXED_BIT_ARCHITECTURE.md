# 🧬 SDD: Arquitectura Mixed-Bit Genomic (v1.5)

**Estatus:** Diseño de Implementación para Certificación Nivel 2
**Objetivo:** Evolucionar el genoma de 2-bits hacia un esquema híbrido que permita la resolución semántica necesaria en Atención mientras se mantiene la eficiencia en FFN.

---

## 1. Diseño de la Jerarquía Genómica
El modelo ya no utiliza una cuantización uniforme. Se segmenta la precisión basándose en la sensibilidad métrica de Fisher:

### A. Capas de Atención (4-bit: "Genoma Denso")
*   **Capas:** `attn_q`, `attn_k`, `attn_v`, `attn_output`.
*   **Formato:** 4 bits por parámetro (16 centroides).
*   **Densidad:** 2 parámetros por byte.
*   **Justificación:** Estas capas actúan como los "ojos" del modelo. La ε-net de 2 bits era incapaz de mapear el manifold de relaciones semánticas, causando el colapso de PPL.

### B. Capas FFN y Embeddings (2-bit: "Genoma Sparse")
*   **Capas:** `ffn_gate`, `ffn_up`, `ffn_down`, `token_embd`.
*   **Formato:** 2 bits por parámetro (4 centroides).
*   **Densidad:** 4 parámetros por byte.
*   **Justificación:** La redundancia en las capas feed-forward permite una compresión extrema. Las anclas F16 corrigen las desviaciones críticas en estas capas.

---

## 2. Nuevo Formato de Almacenamiento (.gaje)
Los tensores en el archivo de base de datos ahora incluyen una cabecera de bits:

*   `0x02`: 2-bit Genomic.
*   `0x04`: 4-bit Genomic.
*   `0x10`: 16-bit Anchor (F16).

---

## 3. Kernels SIMD Híbridos (Roadmap Rust)
Se requiere la creación de nuevos kernels en `src/compute/math.rs` y `src/nn/linear.rs`:

1.  **`genomic_dot_product_4bit`**: Kernel optimizado para desempacar 4 bits y multiplicar por centroides (16 polos).
2.  **`MixedLinear`**: Una nueva estructura en Rust que puede cargar dinámicamente el bit-depth basándose en los metadatos de la capa.

---

## 4. Impacto en Memoria y Performance
*   **Tamaño del Modelo:** El modelo "Silver Adult" (135M) pasará de ~35MB a ~55MB. Sigue estando por debajo del límite de 100MB para "On-Device Sovereignty".
*   **Latencia:** Se espera una penalización del 15-20% en las capas de atención, compensada por la estabilidad de la PPL.

---
*Este diseño sustituye la sección 2 del ARCHITECTURE_CORE.md previo.*
