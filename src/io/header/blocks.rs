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
    /// Devuelve el valor cuantizado (nibble) del peso `idx` dentro del bloque
    #[inline(always)]
    pub fn q_value(&self, idx: usize) -> u8 {
        let byte_idx = idx / 2;
        if idx % 2 == 0 {
            self.qs[byte_idx] & 0x0F
        } else {
            self.qs[byte_idx] >> 4
        }
    }

    /// Dequantiza un peso individual del bloque
    #[inline(always)]
    pub fn dequantize_weight(&self, idx: usize) -> f32 {
        let q4_value = self.q_value(idx);
        let scale = self.scale.to_f32();
        let min = self.min.to_f32();

        (q4_value as f32) * scale + min
    }

    /// Escribe el valor cuantizado (nibble) del peso `idx` dentro del bloque.
    /// Layout inverso a `q_value`: par = nibble bajo, impar = nibble alto.
    #[inline(always)]
    pub fn set_q_value(&mut self, idx: usize, q: u8) {
        let byte_idx = idx / 2;
        let qv = q & 0x0F;
        if idx % 2 == 0 {
            self.qs[byte_idx] = (self.qs[byte_idx] & 0xF0) | qv;
        } else {
            self.qs[byte_idx] = (self.qs[byte_idx] & 0x0F) | (qv << 4);
        }
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

/// Bloque de cuantización group-wise q2_0 con scale + min (2 bits por peso)
/// 32 pesos -> 12 bytes (escala f16, mínimo f16, y 8 bytes de pesos de 2-bits)
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct Q2_0Block {
    pub scale: half::f16,
    pub min: half::f16,
    pub qs: [u8; 8],
}

impl Q2_0Block {
    /// Devuelve el valor cuantizado (2 bits, 0..3) del peso `idx` dentro del bloque.
    /// Cada byte empaqueta 4 códigos (bits 0-1, 2-3, 4-5, 6-7).
    #[inline(always)]
    pub fn q_value(&self, idx: usize) -> u8 {
        let byte_idx = idx / 4;
        let shift = (idx % 4) * 2;
        (self.qs[byte_idx] >> shift) & 0b11
    }

    /// Dequantiza un peso individual del bloque
    #[inline(always)]
    pub fn dequantize_weight(&self, idx: usize) -> f32 {
        let q2_value = self.q_value(idx);
        let scale = self.scale.to_f32();
        let min = self.min.to_f32();

        (q2_value as f32) * scale + min
    }

    /// Asigna el valor cuantizado (2 bits, 0..3) al peso `idx` dentro del bloque.
    #[inline(always)]
    pub fn set_q_value(&mut self, idx: usize, val: u8) {
        let byte_idx = idx / 4;
        let shift = (idx % 4) * 2;
        let mask = !(0b11 << shift);
        let clean = self.qs[byte_idx] & mask;
        self.qs[byte_idx] = clean | ((val & 0b11) << shift);
    }
}
