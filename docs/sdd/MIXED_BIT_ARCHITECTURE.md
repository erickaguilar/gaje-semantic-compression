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

## 3. Implementación: Dispatch Estático y Traits
Para evitar el overhead de branching en los loops de inferencia, utilizaremos un patrón de **Static Dispatch** en Rust.

### A. Estructura de Datos
```rust
pub enum WeightDatabase {
    Genomic2Bit(Arc<Vec<u8>>), // 4 pesos/byte, 4 centroides/bloque
    Genomic4Bit(Arc<Vec<u8>>), // 2 pesos/byte, 16 centroides/bloque
}

pub struct GenomicLinear {
    pub weights: WeightDatabase,
    pub centroids: Vec<f32>,
    // ... otros campos
}
```

### B. El Trait `GenomicKernel`
Definiremos un contrato para los kernels de cómputo que el compilador podrá monomorfizar:

```rust
pub trait GenomicKernel {
    fn dot_product(
        &self,
        weights: &[u8],
        input: &[f32],
        centroids: &[f32],
        stride: usize,
        n_blocks: usize,
    ) -> f32;
}
```

### C. Optimización NEON (Android)
En arquitecturas ARM, el kernel de 4-bits aprovechará que 16 centroides caben exactamente en un registro `uint8x16_t`. Utilizaremos la instrucción `vqtbl1q_u8` para realizar el lookup de los 16 polos de resonancia en un solo ciclo SIMD, manteniendo la paridad de rendimiento con el kernel de 2-bits.

---

## 4. Protocolo de Validación (Baseline PPL)
Antes de proceder con la re-calibración de centroides, ejecutaremos el cambio arquitectural manteniendo el corpus de calibración original para aislar el efecto de la **resolución de la ε-net**.

**Baseline Actual (Silver Adult v1.0):**
*   **Simple:** 23,651.23
*   **Medio:** 32,803.82
*   **Técnico:** 47,549.09

---
*Este diseño sustituye la sección 2 del ARCHITECTURE_CORE.md previo.*
