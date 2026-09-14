//! # 🧬 BF2-Complex: Micro-Kernel Aritmético de Fase en $\mathbb{C}$ (EXPERIMENTAL)
//!
//! > ⚠️ **ESTADO DEL MÓDULO: ARTEFACTO EXPERIMENTAL AISLADO — NO APTO PARA PRODUCCIÓN**
//! >
//! > 1. **Sin Infraestructura de Exportación:** Este archivo implementa únicamente un micro-kernel
//! >    aritmético aislado. No cuenta con integración en `FlatHeaderV2`, `flat_writer`, `flat_reader`
//! >    ni en la arquitectura del transformer (`LlmModel` / `TransformerBlock`).
//! > 2. **Invariancia del Presupuesto de Shannon:** 4 fases cuaternarias en $\mathbb{C}$ equivalen a
//! >    exactamente 2.0 bits por peso ($\log_2 4$). La investigación empírica previa (Q2_0, Q3_0) demostró
//! >    que cuantizaciones sub-4 bits colapsan la similitud angular ($S_c \approx 0.18 - 0.23$) en
//! >    contexto profundo debido a la amplificación multiplicativa de error a través de 24 capas.
//! >    Rotar la constelación a $\mathbb{C}$ no incrementa los bits de información representacional.
//! > 3. **Dispersión Semántica de Born vs Softmax:** Reemplazar $\exp(z)$ por $|\Psi|^2$ altera drásticamente
//! >    la distribución de probabilidades (función par, pérdida de contraste exponencial), requiriendo
//! >    re-entrenamiento completo para no colapsar la perplejidad.
//!
//! ## Fundamentos Matemáticos:
//! - Cada peso se representa en 2 bits sobre el círculo unitario en $\mathbb{C}$:
//!   - `00` (A): $\frac{+1 + i}{\sqrt{2}}$ (Fase $\theta = 45^\circ$, Cuadrante I)
//!   - `01` (C): $\frac{-1 + i}{\sqrt{2}}$ (Fase $\theta = 135^\circ$, Cuadrante II)
//!   - `11` (G): $\frac{-1 - i}{\sqrt{2}}$ (Fase $\theta = 225^\circ$, Cuadrante III)
//!   - `10` (T): $\frac{+1 - i}{\sqrt{2}}$ (Fase $\theta = 315^\circ$, Cuadrante IV)
//!
//! - Para cualquier activación compleja $x = (x_r + i \cdot x_i)$, definimos:
//!   - $s = x_r + x_i$ (suma)
//!   - $d = x_r - x_i$ (diferencia)
//!
//! - La multiplicación compleja $\mathbf{w} \cdot x$ se reduce a:
//!   - `00` (A): $(+d, \,\, +s)$
//!   - `01` (C): $(-s, \,\, +d)$
//!   - `11` (G): $(-d, \,\, -s)$
//!   - `10` (T): $(+s, \,\, -d)$
//!
//! **CERO MULTIPLICACIONES EN EL BUCLE INTERNO.** Toda la acumulación se realiza con
//! inversiones de signo y sumas de $s$ y $d$. Al final del bloque de 32 pesos, se aplica
//! la escala $\alpha \cdot \frac{1}{\sqrt{2}}$.

use rayon::prelude::*;

pub const BF2_BLOCK_SIZE: usize = 32;
pub const INV_SQRT_2: f32 = 0.7071067811865475;

/// Bloque cuantizado BF2 de 32 pesos (8 bytes empaquetados + escala FP32)
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bf2Block {
    /// 32 pesos de 2 bits empaquetados en 8 bytes (4 pesos por byte)
    pub weights: [u8; 8],
    /// Factor de escala macro/micro alpha del bloque
    pub scale: f32,
}

impl Default for Bf2Block {
    fn default() -> Self {
        Self {
            weights: [0u8; 8],
            scale: 1.0,
        }
    }
}

/// Matriz de pesos en formato BF2-Complex ($M \times K$)
#[derive(Clone, Debug)]
pub struct Bf2Matrix {
    pub rows: usize,
    pub cols: usize,
    /// Bloques organizados por filas. Cada fila tiene `cols / 32` bloques.
    pub blocks: Vec<Bf2Block>,
}

impl Bf2Matrix {
    /// Crea una matriz BF2 a partir de pesos complejos reales e imaginarios.
    /// Si `weights_imag` es None, se asume entrada puramente real y se infiere la fase.
    pub fn from_f32(
        weights_real: &[f32],
        weights_imag: Option<&[f32]>,
        rows: usize,
        cols: usize,
    ) -> Self {
        assert_eq!(
            cols % BF2_BLOCK_SIZE,
            0,
            "cols ({}) debe ser múltiplo de BF2_BLOCK_SIZE ({})",
            cols,
            BF2_BLOCK_SIZE
        );
        assert_eq!(
            weights_real.len(),
            rows * cols,
            "Longitud de weights_real no coincide con rows * cols"
        );

        let blocks_per_row = cols / BF2_BLOCK_SIZE;
        let mut blocks = Vec::with_capacity(rows * blocks_per_row);

        for r in 0..rows {
            let row_offset = r * cols;
            for b in 0..blocks_per_row {
                let block_offset = row_offset + b * BF2_BLOCK_SIZE;

                // 1. Calcular escala del bloque (amplitud media L1)
                let mut sum_amp = 0.0f32;
                for i in 0..BF2_BLOCK_SIZE {
                    let idx = block_offset + i;
                    let re = weights_real[idx];
                    let im = weights_imag.map_or(0.0, |im_slice| im_slice[idx]);
                    sum_amp += (re * re + im * im).sqrt();
                }
                let scale = sum_amp / (BF2_BLOCK_SIZE as f32);

                // 2. Empaquetar 32 pesos en 8 bytes (2 bits cada uno)
                let mut packed_weights = [0u8; 8];
                for byte_idx in 0..8 {
                    let mut byte_val = 0u8;
                    for p in 0..4 {
                        let i = byte_idx * 4 + p;
                        let idx = block_offset + i;
                        let re = weights_real[idx];
                        let im = weights_imag.map_or(0.0, |im_slice| im_slice[idx]);

                        // Determinar el cuadrante de fase
                        let angle = im.atan2(re);
                        let bits = if (0.0..std::f32::consts::FRAC_PI_2).contains(&angle) {
                            0b00 // Quadrant I: A (+1 + i)
                        } else if (std::f32::consts::FRAC_PI_2..=std::f32::consts::PI).contains(&angle) {
                            0b01 // Quadrant II: C (-1 + i)
                        } else if (-std::f32::consts::PI..-std::f32::consts::FRAC_PI_2).contains(&angle) {
                            0b11 // Quadrant III: G (-1 - i)
                        } else {
                            0b10 // Quadrant IV: T (+1 - i)
                        };

                        byte_val = (byte_val << 2) | bits;
                    }
                    packed_weights[byte_idx] = byte_val;
                }

                blocks.push(Bf2Block {
                    weights: packed_weights,
                    scale,
                });
            }
        }

        Self { rows, cols, blocks }
    }

    /// Tamaño en bytes en memoria de la matriz cuantizada
    pub fn memory_bytes(&self) -> usize {
        self.blocks.len() * std::mem::size_of::<Bf2Block>()
    }

    /// Tasa de compresión contra FP32 denso (ambas componentes real + imag)
    pub fn compression_ratio(&self) -> f32 {
        let dense_bytes = (self.rows * self.cols * std::mem::size_of::<f32>() * 2) as f32;
        dense_bytes / (self.memory_bytes() as f32)
    }
}

/// Prepara las sumas y diferencias del vector de entrada complejo:
/// $s_k = x_{r,k} + x_{i,k}$
/// $d_k = x_{r,k} - x_{i,k}$
///
/// Se ejecuta una sola vez para el vector de entrada con coste $O(K)$ sumas/restas.
#[inline(always)]
pub fn prepare_complex_input(
    real: &[f32],
    imag: &[f32],
    s_out: &mut [f32],
    d_out: &mut [f32],
) {
    let k = real.len();
    assert_eq!(imag.len(), k);
    assert_eq!(s_out.len(), k);
    assert_eq!(d_out.len(), k);

    for i in 0..k {
        let r = real[i];
        let im = imag[i];
        s_out[i] = r + im;
        d_out[i] = r - im;
    }
}

/// Realiza la multiplicación matriz-vector BF2-Complex sin multiplicadores de hardware
/// en el bucle interno de acumulación.
///
/// $Y = W_{\text{BF2}} X \in \mathbb{C}^M$
#[inline]
pub fn bf2_gemv_scalar(
    matrix: &Bf2Matrix,
    s_vec: &[f32],
    d_vec: &[f32],
    out_real: &mut [f32],
    out_imag: &mut [f32],
) {
    let rows = matrix.rows;
    let cols = matrix.cols;
    let blocks_per_row = cols / BF2_BLOCK_SIZE;

    assert_eq!(s_vec.len(), cols);
    assert_eq!(d_vec.len(), cols);
    assert_eq!(out_real.len(), rows);
    assert_eq!(out_imag.len(), rows);

    for r in 0..rows {
        let mut row_acc_r = 0.0f32;
        let mut row_acc_i = 0.0f32;
        let row_blocks = &matrix.blocks[r * blocks_per_row..(r + 1) * blocks_per_row];

        for (b_idx, block) in row_blocks.iter().enumerate() {
            let base_k = b_idx * BF2_BLOCK_SIZE;
            let mut block_acc_r = 0.0f32;
            let mut block_acc_i = 0.0f32;

            for byte_idx in 0..8 {
                let byte = block.weights[byte_idx];
                let k0 = base_k + byte_idx * 4;

                // Desempaquetar 4 pesos del byte
                let w0 = (byte >> 6) & 0b11;
                let w1 = (byte >> 4) & 0b11;
                let w2 = (byte >> 2) & 0b11;
                let w3 = byte & 0b11;

                // 1er peso
                let (r0, i0) = match w0 {
                    0b00 => (d_vec[k0], s_vec[k0]),
                    0b01 => (-s_vec[k0], d_vec[k0]),
                    0b10 => (s_vec[k0], -d_vec[k0]),
                    0b11 => (-d_vec[k0], -s_vec[k0]),
                    _ => unreachable!(),
                };
                block_acc_r += r0;
                block_acc_i += i0;

                // 2do peso
                let k1 = k0 + 1;
                let (r1, i1) = match w1 {
                    0b00 => (d_vec[k1], s_vec[k1]),
                    0b01 => (-s_vec[k1], d_vec[k1]),
                    0b10 => (s_vec[k1], -d_vec[k1]),
                    0b11 => (-d_vec[k1], -s_vec[k1]),
                    _ => unreachable!(),
                };
                block_acc_r += r1;
                block_acc_i += i1;

                // 3er peso
                let k2 = k0 + 2;
                let (r2, i2) = match w2 {
                    0b00 => (d_vec[k2], s_vec[k2]),
                    0b01 => (-s_vec[k2], d_vec[k2]),
                    0b10 => (s_vec[k2], -d_vec[k2]),
                    0b11 => (-d_vec[k2], -s_vec[k2]),
                    _ => unreachable!(),
                };
                block_acc_r += r2;
                block_acc_i += i2;

                // 4to peso
                let k3 = k0 + 3;
                let (r3, i3) = match w3 {
                    0b00 => (d_vec[k3], s_vec[k3]),
                    0b01 => (-s_vec[k3], d_vec[k3]),
                    0b10 => (s_vec[k3], -d_vec[k3]),
                    0b11 => (-d_vec[k3], -s_vec[k3]),
                    _ => unreachable!(),
                };
                block_acc_r += r3;
                block_acc_i += i3;
            }

            // Escalar el bloque completo con 1 sola multiplicación por componente
            let effective_scale = block.scale * INV_SQRT_2;
            row_acc_r += block_acc_r * effective_scale;
            row_acc_i += block_acc_i * effective_scale;
        }

        out_real[r] = row_acc_r;
        out_imag[r] = row_acc_i;
    }
}

/// Multiplicación matriz-vector BF2-Complex paralelizada con Rayon.
pub fn bf2_gemv_parallel(
    matrix: &Bf2Matrix,
    s_vec: &[f32],
    d_vec: &[f32],
    out_real: &mut [f32],
    out_imag: &mut [f32],
) {
    let cols = matrix.cols;
    let blocks_per_row = cols / BF2_BLOCK_SIZE;

    out_real
        .par_iter_mut()
        .zip(out_imag.par_iter_mut())
        .enumerate()
        .for_each(|(r, (out_r, out_i))| {
            let mut row_acc_r = 0.0f32;
            let mut row_acc_i = 0.0f32;
            let row_blocks = &matrix.blocks[r * blocks_per_row..(r + 1) * blocks_per_row];

            for (b_idx, block) in row_blocks.iter().enumerate() {
                let base_k = b_idx * BF2_BLOCK_SIZE;
                let mut block_acc_r = 0.0f32;
                let mut block_acc_i = 0.0f32;

                for byte_idx in 0..8 {
                    let byte = block.weights[byte_idx];
                    let k0 = base_k + byte_idx * 4;

                    let w0 = (byte >> 6) & 0b11;
                    let w1 = (byte >> 4) & 0b11;
                    let w2 = (byte >> 2) & 0b11;
                    let w3 = byte & 0b11;

                    let (r0, i0) = match w0 {
                        0b00 => (d_vec[k0], s_vec[k0]),
                        0b01 => (-s_vec[k0], d_vec[k0]),
                        0b10 => (s_vec[k0], -d_vec[k0]),
                        0b11 => (-d_vec[k0], -s_vec[k0]),
                        _ => unreachable!(),
                    };
                    block_acc_r += r0;
                    block_acc_i += i0;

                    let k1 = k0 + 1;
                    let (r1, i1) = match w1 {
                        0b00 => (d_vec[k1], s_vec[k1]),
                        0b01 => (-s_vec[k1], d_vec[k1]),
                        0b10 => (s_vec[k1], -d_vec[k1]),
                        0b11 => (-d_vec[k1], -s_vec[k1]),
                        _ => unreachable!(),
                    };
                    block_acc_r += r1;
                    block_acc_i += i1;

                    let k2 = k0 + 2;
                    let (r2, i2) = match w2 {
                        0b00 => (d_vec[k2], s_vec[k2]),
                        0b01 => (-s_vec[k2], d_vec[k2]),
                        0b10 => (s_vec[k2], -d_vec[k2]),
                        0b11 => (-d_vec[k2], -s_vec[k2]),
                        _ => unreachable!(),
                    };
                    block_acc_r += r2;
                    block_acc_i += i2;

                    let k3 = k0 + 3;
                    let (r3, i3) = match w3 {
                        0b00 => (d_vec[k3], s_vec[k3]),
                        0b01 => (-s_vec[k3], d_vec[k3]),
                        0b10 => (s_vec[k3], -d_vec[k3]),
                        0b11 => (-d_vec[k3], -s_vec[k3]),
                        _ => unreachable!(),
                    };
                    block_acc_r += r3;
                    block_acc_i += i3;
                }

                let effective_scale = block.scale * INV_SQRT_2;
                row_acc_r += block_acc_r * effective_scale;
                row_acc_i += block_acc_i * effective_scale;
            }

            *out_r = row_acc_r;
            *out_i = row_acc_i;
        });
}

/// Regla de Born para proyección de probabilidad de tokens:
/// $P(k) = \frac{|\Psi_k|^2}{\sum_{j} |\Psi_j|^2} = \frac{x_{r,k}^2 + x_{i,k}^2}{\sum_j (x_{r,j}^2 + x_{i,j}^2)}$
///
/// Reemplaza la exponencial de Softmax $(\exp(z_k))$ eliminando el cálculo trascendental
/// en la capa final del vocabulario.
pub fn born_density(real: &[f32], imag: &[f32], probs_out: &mut [f32]) {
    let n = real.len();
    assert_eq!(imag.len(), n);
    assert_eq!(probs_out.len(), n);

    let mut sum_density = 0.0f32;
    for i in 0..n {
        let r = real[i];
        let im = imag[i];
        let density = r * r + im * im;
        probs_out[i] = density;
        sum_density += density;
    }

    let inv_sum = if sum_density > 1e-12 {
        1.0 / sum_density
    } else {
        1.0 / (n as f32)
    };

    for p in probs_out.iter_mut() {
        *p *= inv_sum;
    }
}

/// Activación no lineal ModReLU:
/// $\sigma(z) = \text{ReLU}(|z| + b) \cdot \frac{z}{|z|}$
///
/// Preserva intacta la fase compleja $\theta = \text{arg}(z)$ mientras modula la amplitud
/// y filtra el ruido cuántico por debajo del umbral $|b|$.
pub fn mod_relu(real: &mut [f32], imag: &mut [f32], bias: f32) {
    let n = real.len();
    assert_eq!(imag.len(), n);

    for i in 0..n {
        let r = real[i];
        let im = imag[i];
        let mag = (r * r + im * im).sqrt();
        let new_mag = (mag + bias).max(0.0);

        if mag > 1e-9 {
            let factor = new_mag / mag;
            real[i] = r * factor;
            imag[i] = im * factor;
        } else {
            real[i] = 0.0;
            imag[i] = 0.0;
        }
    }
}
