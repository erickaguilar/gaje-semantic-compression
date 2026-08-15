// =============================================================================
// blocks — Bloques de cuantización group-wise Q4_0/Q8_0 y su dequantización
// =============================================================================

/// Bloque de cuantización group-wise q4_0 con scale + min (Variant B)
/// 32 pesos -> 20 bytes (escala f16, mínimo f16, y 16 bytes de pesos de 4-bits)
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct Q4_0Block {
    pub scale: half::f16,
    pub min: half::f16,
    pub qs: [u8; 16],
}

impl Q4_0Block {
    /// Dequantiza un peso individual del bloque
    #[inline(always)]
    pub fn dequantize_weight(&self, idx: usize) -> f32 {
        let byte_idx = idx / 2;
        let q4_value = if idx % 2 == 0 {
            self.qs[byte_idx] & 0x0F
        } else {
            self.qs[byte_idx] >> 4
        };

        let scale = self.scale.to_f32();
        let min = self.min.to_f32();

        (q4_value as f32) * scale + min
    }
}

/// Bloque de cuantización group-wise q8_0 con scale
/// 32 pesos -> 34 bytes (escala f16, y 32 bytes de pesos de 8-bits con signo)
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct Q8_0Block {
    pub scale: half::f16,
    pub qs: [i8; 32],
}

impl Q8_0Block {
    /// Dequantiza un peso individual del bloque
    #[inline(always)]
    pub fn dequantize_weight(&self, idx: usize) -> f32 {
        let scale = self.scale.to_f32();
        let q8_value = self.qs[idx] as f32;
        q8_value * scale
    }
}
