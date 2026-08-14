// =============================================================================
// lut.rs — Distancia LUT, tabla de shuffle y el filtro del "Río Semántico"
// =============================================================================

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// =============================================================================
// Tabla de shuffle para decodificación de bases genómicas (2-bit → índice)
// =============================================================================

static mut SHUFFLE_MASK_TABLE: [[u8; 16]; 256] = [[0; 16]; 256];
static mut SHUFFLE_TABLE_INITIALIZED: bool = false;

/// # Safety
/// Esta función accede y modifica variables estáticas globales mutables sin sincronización.
/// Debe ser llamada una sola vez durante la inicialización del programa o garantizando
/// que no haya condiciones de carrera.
pub unsafe fn init_shuffle_table() {
    if SHUFFLE_TABLE_INITIALIZED {
        return;
    }
    for b in 0..256usize {
        for i in 0..4 {
            let shift = (3 - i) * 2;
            let bits = (b >> shift) & 0b11;
            let idx = (bits ^ (bits >> 1)) as u8;
            for j in 0..4 {
                SHUFFLE_MASK_TABLE[b][(i * 4 + j) as usize] = idx * 4 + j as u8;
            }
        }
    }
    SHUFFLE_TABLE_INITIALIZED = true;
}

// =============================================================================
// lateral_inhibition_kwta — El Filtro del "Río Semántico"
// =============================================================================

/// Implementa la Inhibición Lateral (K-Winners-Take-All).
///
/// Este kernel simula cómo las "Islas" de cristalización inhiben el ruido
/// de la "Materia Oscura" circundante, forzando a la señal a fluir por los
/// canales de máxima resonancia (El Río Semántico).
pub fn lateral_inhibition_kwta(scores: &mut [f32], k: usize) {
    if scores.len() <= k {
        return;
    }

    // Revertido para diagnóstico de NaN
    let mut sorted_scores = scores.to_vec();
    sorted_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = sorted_scores[k - 1];

    // Inhibición: las señales por debajo del umbral se extinguen (Materia Oscura)
    for s in scores.iter_mut() {
        if *s < threshold {
            *s = -1e9; // Silencio inhibitorio
        }
    }
}

// =============================================================================
// calculate_distance_lut — Distancia LUT universal
// =============================================================================

/// # Safety
/// Esta función realiza accesos directos a memoria mediante `get_unchecked` (implícito en la lógica nativa)
/// y asume que todos los strands y máscaras tienen longitudes coherentes con `n_dims`.
#[inline(always)]
pub unsafe fn calculate_distance_lut(
    lut_base: &[f32],
    lut_epi: &[f32],
    lut_tri: &[f32],
    strand: &[u8],
    epi_strand: &[u8],
    tri_strand: &[u8],
    mask: &[u8],
    n_dims: usize,
) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        let mut sum_v = vdupq_n_f32(0.0);
        let mut dims = 0;
        let n_blocks = n_dims / 4;
        for i in 0..n_blocks {
            let mode = *mask.get(i).unwrap_or(&0);
            let b_byte = *strand.get(i).unwrap_or(&0);
            let mut d_v = [0.0f32; 4];
            for j in 0..4 {
                let shift = (3 - j) * 2;
                let bb = (b_byte >> shift) & 0b11;
                let b_idx = (bb ^ (bb >> 1)) as usize;
                if mode == 0 {
                    d_v[j] = *lut_base.get(dims * 4 + b_idx).unwrap_or(&0.0);
                } else if mode == 1 {
                    let eb = (*epi_strand.get(i).unwrap_or(&0) >> shift) & 0b11;
                    d_v[j] = *lut_epi
                        .get(dims * 16 + (b_idx << 2 | (eb ^ (eb >> 1)) as usize))
                        .unwrap_or(&0.0);
                } else {
                    let eb = (*epi_strand.get(i).unwrap_or(&0) >> shift) & 0b11;
                    let tb = (*tri_strand.get(i).unwrap_or(&0) >> shift) & 0b11;
                    d_v[j] = *lut_tri
                        .get(
                            dims * 64
                                + (b_idx << 4
                                    | ((eb ^ (eb >> 1)) as usize) << 2
                                    | (tb ^ (tb >> 1)) as usize),
                        )
                        .unwrap_or(&0.0);
                }
                dims += 1;
            }
            sum_v = vaddq_f32(sum_v, vld1q_f32(d_v.as_ptr()));
        }
        let mut total = vaddvq_f32(sum_v);
        while dims < n_dims {
            let i = dims / 4;
            let mode = *mask.get(i).unwrap_or(&0);
            let shift = (3 - (dims % 4)) * 2;
            let bb = (*strand.get(i).unwrap_or(&0) >> shift) & 0b11;
            let b_idx = (bb ^ (bb >> 1)) as usize;
            if mode == 0 {
                total += *lut_base.get(dims * 4 + b_idx).unwrap_or(&0.0);
            } else if mode == 1 {
                let eb = (*epi_strand.get(i).unwrap_or(&0) >> shift) & 0b11;
                total += *lut_epi
                    .get(dims * 16 + (b_idx << 2 | (eb ^ (eb >> 1)) as usize))
                    .unwrap_or(&0.0);
            } else {
                let eb = (*epi_strand.get(i).unwrap_or(&0) >> shift) & 0b11;
                let tb = (*tri_strand.get(i).unwrap_or(&0) >> shift) & 0b11;
                total += *lut_tri
                    .get(
                        dims * 64
                            + (b_idx << 4
                                | ((eb ^ (eb >> 1)) as usize) << 2
                                | (tb ^ (tb >> 1)) as usize),
                    )
                    .unwrap_or(&0.0);
            }
            dims += 1;
        }
        total.sqrt()
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut total = 0.0f32;
        for dims in 0..n_dims {
            let i = dims / 4;
            let mode = *mask.get(i).unwrap_or(&0);
            let shift = (3 - (dims % 4)) * 2;
            let bb = (*strand.get(i).unwrap_or(&0) >> shift) & 0b11;
            let b_idx = (bb ^ (bb >> 1)) as usize;
            if mode == 0 {
                total += *lut_base.get(dims * 4 + b_idx).unwrap_or(&0.0);
            } else if mode == 1 {
                let eb = (*epi_strand.get(i).unwrap_or(&0) >> shift) & 0b11;
                total += *lut_epi
                    .get(dims * 16 + (b_idx << 2 | (eb ^ (eb >> 1)) as usize))
                    .unwrap_or(&0.0);
            } else {
                let eb = (*epi_strand.get(i).unwrap_or(&0) >> shift) & 0b11;
                let tb = (*tri_strand.get(i).unwrap_or(&0) >> shift) & 0b11;
                total += *lut_tri
                    .get(
                        dims * 64
                            + (b_idx << 4
                                | ((eb ^ (eb >> 1)) as usize) << 2
                                | (tb ^ (tb >> 1)) as usize),
                    )
                    .unwrap_or(&0.0);
            }
        }
        total.sqrt()
    }
}

// Alias para compatibilidad con rama windows
/// # Safety
/// Ver `calculate_distance_lut`.
#[inline(always)]
pub unsafe fn calculate_distance_lut_neon(
    lut_base: &[f32],
    lut_epi: &[f32],
    lut_tri: &[f32],
    strand: &[u8],
    epi_strand: &[u8],
    tri_strand: &[u8],
    mask: &[u8],
    n_dims: usize,
) -> f32 {
    calculate_distance_lut(
        lut_base, lut_epi, lut_tri, strand, epi_strand, tri_strand, mask, n_dims,
    )
}