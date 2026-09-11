# 🧬 Especificación Técnica: Formato Binario `Q3_0Block` y Pipeline de Exportación Asimétrica

**Estado:** Especificación Técnica Archivada como Referencia (No Implementada en Producción)  
**Fecha:** 2026-09-11  
**Autores:** Erick Aguilar & Antigravity (Google DeepMind Pair Programming)  
**Veredicto Oficial:** **RECHAZADO POR ARBITRAJE EMPÍRICO** ([`docs/research/bilingual_sustained_generation_matrix.json`](bilingual_sustained_generation_matrix.json))  
**Módulos Evaluados:** `src/io/header/blocks.rs`, `src/io/header/types.rs`, `src/nn/linear/storage.rs`, `src/compute/kernels/`, `src/cli/tools.rs`

> [!CAUTION]
> **ESTADO DE LA ESPECIFICACIÓN: ARCHIVADA / PISO CONFIRMADO EN Q4_0**  
> El arnés de generación bilingüe sostenida de 200 tokens (`tests/test_bilingual_sustained_harness.rs`) determinó formalmente que la cuantización FFN a 3-bit no sostiene la coherencia en autoregresión prolongada (50% de concordancia en Token-1, colapso en bucle cíclico `ZH-2` y mezcla lingüística en `EN-2`).  
> **Decisión de Ingeniería:** `Q4_0` se certifica como el piso mínimo inviolable para modelos de 0.5B. Esta especificación se preserva intacta como referencia de diseño binario para futuros experimentos en modelos de gran escala ($7\text{B}+$ con mayor capacidad latente), pero **NO se implementará en el motor de producción `gaje-core`**.

---

## 1. Motivación y Justificación Empírica


El análisis ortogonal de ablación en 202 tokens continuos ([`ASYMMETRIC_3BIT_FFN_ATTENTION_ABLATION_AND_TOKENIZATION_FINDINGS.md`](file:///data/data/com.termux/files/home/develop/gaje-semantic-compression/docs/research/ASYMMETRIC_3BIT_FFN_ATTENTION_ABLATION_AND_TOKENIZATION_FINDINGS.md)) demostró que:

1. **La Atención en 2-bit colapsa el modelo:** Con FFN en Q4_0, la atención en Q2_0 produce una caída a $S_c \approx 0.2369$ en el cuerpo del transformer. Por tanto, la atención debe preservarse en **Q4_0** (4 bits).
2. **El FFN tolera 3 bits con alta fidelidad:** Cuantizar el FFN (87.7% de los parámetros del bloque) a 3 bits sostiene un **$S_c = 0.9437$** en el cuerpo (L10) y **$S_c = 0.9667$** en la salida (L23), sin degradación angular ni colapso de fase.
3. **Economía de Parámetros:** Una arquitectura asimétrica con Atención en 4-bit y FFN en 3-bit entrega **$3.123\text{ bits/peso efectivo}$**, logrando una **reducción neta del 21.92%** en los pesos del cuerpo respecto a Q4_0 uniforme.

Para hacer operativo este hallazgo en producción, este documento define la especificación canónica del nuevo bloque binario **`Q3_0Block`** y la arquitectura de exportación asimétrica en `gaje-cli`.

---

## 2. Layout Binario de `Q3_0Block`

Un bloque `Q3_0Block` empaqueta un grupo de **32 pesos contiguos** en una estructura compacta de exactamente **16 bytes**:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        ESTRUCTURA DE Q3_0Block (16 Bytes)              │
├──────────────────┬──────────────────┬──────────────────────────────────┤
│   scale (2 B)    │     min (2 B)    │         qs: [u8; 12] (12 B)      │
│     half::f16    │     half::f16    │   32 pesos × 3 bits = 96 bits    │
└──────────────────┴──────────────────┴──────────────────────────────────┘
```

### Definición en Rust (`src/io/header/blocks.rs`)

```rust
/// Bloque de cuantización group-wise Q3_0 con scale + min
/// 32 pesos -> 16 bytes (escala f16, mínimo f16, y 12 bytes de pesos de 3-bits)
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct Q3_0Block {
    pub scale: half::f16,
    pub min: half::f16,
    pub qs: [u8; 12],
}
```

### Propiedades de Hardware y Memoria:
* **Tamaño exacto:** $2 + 2 + 12 = \mathbf{16\text{ bytes}}$ (128 bits).
* **Alineación:** Potencia exacta de 2, permitiendo cargas alineadas de 128 bits en registros SIMD (`uint8x16_t` en ARM NEON o `__m128i` en x86 AVX).
* **Densidad Bruta:** $16\text{ bytes} / 32\text{ pesos} = \mathbf{4.0\text{ bits/peso}}$ a nivel de contenedor de bloque (incluyendo escala y min), mientras que el payload de información pura es de **$3.0\text{ bits}$**.

---

## 3. Algoritmo de Empaquetado (*Bitpacking*) de 3 Bits

Dado que 3 bits no dividen de forma entera a un byte de 8 bits, el empaquetado se estructura en **cuatro cuartetos de 8 pesos**:
$$8\text{ pesos} \times 3\text{ bits} = 24\text{ bits} = \mathbf{3\text{ bytes}}$$

Por consiguiente, 32 pesos se distribuyen en $4 \times 3 = \mathbf{12\text{ bytes}}$.

### Mapeo Canónico de Bits (8 pesos en 3 bytes):

Sea $w_0, w_1, \dots, w_7 \in [0, 7]$ los 8 valores cuantizados de 3 bits, y $b_0, b_1, b_2$ los 3 bytes empaquetados:

| Peso | Bits | Byte Fuente | Máscara y Desplazamiento |
| :---: | :---: | :---: | :--- |
| **$w_0$** | 3 | $b_0$ | `b0 & 0x07` (bits 0..2) |
| **$w_1$** | 3 | $b_0$ | `(b0 >> 3) & 0x07` (bits 3..5) |
| **$w_2$** | 2 + 1 | $b_0, b_1$ | `((b0 >> 6) & 0x03) | ((b1 & 0x01) << 2)` (bits 6..7 de $b_0$ + bit 0 de $b_1$) |
| **$w_3$** | 3 | $b_1$ | `(b1 >> 1) & 0x07` (bits 1..3) |
| **$w_4$** | 3 | $b_1$ | `(b1 >> 4) & 0x07` (bits 4..6) |
| **$w_5$** | 1 + 2 | $b_1, b_2$ | `((b1 >> 7) & 0x01) | ((b2 & 0x03) << 1)` (bit 7 de $b_1$ + bits 0..1 de $b_2$) |
| **$w_6$** | 3 | $b_2$ | `(b2 >> 2) & 0x07` (bits 2..4) |
| **$w_7$** | 3 | $b_2$ | `(b2 >> 5) & 0x07` (bits 5..7) |

---

## 4. Implementación en Rust (`impl Q3_0Block`)

```rust
impl Q3_0Block {
    /// Extrae el código cuantizado (0..7) del peso `idx` (0..31)
    #[inline(always)]
    pub fn q_value(&self, idx: usize) -> u8 {
        let group = idx / 8;        // Grupo de 8 pesos (0..3)
        let sub_idx = idx % 8;      // Índice dentro del grupo (0..7)
        let b_off = group * 3;      // Offset base en self.qs (0, 3, 6, 9)

        let b0 = self.qs[b_off] as u32;
        let b1 = self.qs[b_off + 1] as u32;
        let b2 = self.qs[b_off + 2] as u32;

        let packed24 = b0 | (b1 << 8) | (b2 << 16);
        ((packed24 >> (sub_idx * 3)) & 0x07) as u8
    }

    /// Escribe el código cuantizado (0..7) del peso `idx` (0..31)
    #[inline(always)]
    pub fn set_q_value(&mut self, idx: usize, val: u8) {
        let group = idx / 8;
        let sub_idx = idx % 8;
        let b_off = group * 3;

        let b0 = self.qs[b_off] as u32;
        let b1 = self.qs[b_off + 1] as u32;
        let b2 = self.qs[b_off + 2] as u32;

        let mut packed24 = b0 | (b1 << 8) | (b2 << 16);
        let shift = sub_idx * 3;
        let mask = !(0x07 << shift);

        packed24 = (packed24 & mask) | (((val & 0x07) as u32) << shift);

        self.qs[b_off] = (packed24 & 0xFF) as u8;
        self.qs[b_off + 1] = ((packed24 >> 8) & 0xFF) as u8;
        self.qs[b_off + 2] = ((packed24 >> 16) & 0xFF) as u8;
    }

    /// Dequantiza un peso individual a f32
    #[inline(always)]
    pub fn dequantize_weight(&self, idx: usize) -> f32 {
        let q = self.q_value(idx) as f32;
        let scale = self.scale.to_f32();
        let min = self.min.to_f32();
        q * scale + min
    }
}
```

---

## 5. Kernel de Producto Punto Vectorizado: `genomic_dot_product_q3_0`

Para inferencia en CPU (ARM NEON / AVX2), el desempaquetado de 24 bits se ejecuta sin bifurcaciones condicionales (*branchless*) procesando cuartetos de 8 elementos:

```rust
#[inline(always)]
pub unsafe fn genomic_dot_product_q3_0(
    blocks: &[Q3_0Block],
    input: &[f32],
    n_blocks: usize,
) -> f32 {
    let mut total_sum = 0.0f32;

    for b in 0..n_blocks {
        let block = &blocks[b];
        let in_base = b * 32;
        let scale = block.scale.to_f32();
        let min = block.min.to_f32();

        let mut block_sum = 0.0f32;
        for g in 0..4 {
            let b_off = g * 3;
            let b0 = block.qs[b_off] as u32;
            let b1 = block.qs[b_off + 1] as u32;
            let b2 = block.qs[b_off + 2] as u32;
            let packed24 = b0 | (b1 << 8) | (b2 << 16);

            let g_in = in_base + g * 8;
            for i in 0..8 {
                let q = ((packed24 >> (i * 3)) & 0x07) as f32;
                let w = q * scale + min;
                block_sum += w * *input.get_unchecked(g_in + i);
            }
        }
        total_sum += block_sum;
    }

    total_sum
}
```

---

## 6. Extensiones en la Jerarquía de Tipos del Motor

### A. Cabecera (`src/io/header/types.rs`)
```rust
pub enum QuantFormat {
    LegacyCentroids = 0,
    Q4_0 = 1,
    Q8_0 = 2,
    Q2_0 = 3,
    Q3_0 = 4,   // <-- Nuevo formato estandarizado
    Unknown,
}
```

### B. Base de Datos de Pesos (`src/nn/linear/storage.rs`)
```rust
pub enum WeightStorage {
    Genomic2Bit(Arc<Vec<u8>>),
    Genomic4Bit(Arc<Vec<u8>>),
    GenomicQ4_0(Arc<Vec<Q4_0Block>>),
    GenomicQ8_0(Arc<Vec<Q8_0Block>>),
    GenomicQ2_0(Arc<Vec<Q2_0Block>>),
    GenomicQ3_0(Arc<Vec<Q3_0Block>>), // <-- Contenedor de 3-bit
    GenomicF32(Arc<Vec<f32>>),
}
```

### C. Despacho en `compute_single_row` (`src/nn/linear/forward.rs`)
```rust
WeightDatabase::GenomicQ3_0(db) => {
    let row_off = i * n_blocks;
    let db_slice = db.get(row_off..row_off + n_blocks).unwrap_or(&[]);
    if !db_slice.is_empty() {
        sum = unsafe {
            crate::compute::kernels::genomic_dot_product_q3_0(db_slice, input, n_blocks)
        };
    }
}
```

---

## 7. Pipeline de Exportación Asimétrica (`gaje-cli export-flat`)

La CLI permitirá compresión asimétrica dirigida por tipo de tensor:

```bash
cargo run --release --bin gaje-cli -- export-flat \
  --model models/qwen2.5-0.5b-instruct-q4_0.gguf \
  --quant-attn q4_0 \
  --quant-ffn q3_0 \
  --out models/qwen2_5_0_5b_q3_a.flat
```

### Lógica de Despacho de Cuantización por Tensor:

```rust
let is_attention = name.contains("attn_q")
    || name.contains("attn_k")
    || name.contains("attn_v")
    || name.contains("attn_output");

let is_ffn = name.contains("ffn_gate")
    || name.contains("ffn_up")
    || name.contains("ffn_down");

let target_format = if is_attention {
    QuantFormat::Q4_0
} else if is_ffn {
    QuantFormat::Q3_0
} else {
    QuantFormat::Q8_0 // embeddings y lm_head
};
```

---

## 8. Verificación de Integridad Matemática (Unit Tests Previos)

Se diseñan los siguientes tests unitarios para certificar el empaquetado antes de la integración:
1. **`test_q3_0_bitpacking_roundtrip`:** Escribir valores aleatorios $0..7$ en los 32 slots de un `Q3_0Block`, leerlos con `q_value(i)` y verificar concordancia del 100%.
2. **`test_q3_0_dequantize_precision`:** Contrastar el error cuadrático medio (MSE) de dequantización de un tensor sintético frente a una cuantización directa en punto flotante.
3. **`test_q3_0_dot_product_parity`:** Verificar que `genomic_dot_product_q3_0` produzca el mismo resultado numérico (dentro de $\epsilon = 10^{-5}$) que el producto punto matricial FP32 equivalente.
