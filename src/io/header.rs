use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantFormat {
    LegacyCentroids = 0,
    Q4_0 = 1,
    Q8_0 = 2,
    Unknown,
}

#[derive(Debug)]
pub enum HeaderError {
    TooShort,
    InvalidMagic,
}

impl fmt::Display for HeaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooShort => write!(f, "Header buffer is less than 4096 bytes"),
            Self::InvalidMagic => write!(f, "Invalid magic bytes (expected 'GAJE')"),
        }
    }
}

impl std::error::Error for HeaderError {}

/// Header del formato .flat v2
/// 
/// Cambios vs v1:
/// - Bytes 48-51: group_size (antes reservado/relleno de ceros)
/// - Bytes 52-55: quant_format (antes reservado/relleno de ceros)
/// - Compatibilidad garantizada: modelos v1 (ceros en bytes 48-55) se interpretan como quant_format=0, group_size=16
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FlatHeaderV2 {
    // === IDENTIFICACIÓN (12 bytes) ===
    pub magic: [u8; 4],
    pub version: u32,
    pub flags: u32,
    pub num_tensors: u32,
    
    // === METADATOS Y DICT (24 bytes) ===
    pub meta_len: u64,
    pub dir_len: u64,
    pub weights_offset: u64,
    pub weights_len: u64,
    
    // === CUANTIZACIÓN (8 bytes) ===
    pub group_size: u32,
    pub quant_format: u32,
    
    // === RESERVA (4040 bytes) ===
    pub reserved: [u8; 4040],
}

impl FlatHeaderV2 {
    pub const SIZE: usize = 4096;
    
    /// Lee el header desde un slice de bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, HeaderError> {
        if bytes.len() < Self::SIZE {
            return Err(HeaderError::TooShort);
        }
        
        let header = unsafe { 
            std::ptr::read(bytes.as_ptr() as *const Self) 
        };
        
        // Validar magic bytes
        if &header.magic != b"GAJE" {
            return Err(HeaderError::InvalidMagic);
        }
        
        Ok(header)
    }
    
    /// Determina el formato de cuantización
    pub fn quantization_type(&self) -> QuantFormat {
        match self.quant_format {
            0 => QuantFormat::LegacyCentroids,
            1 => QuantFormat::Q4_0,
            2 => QuantFormat::Q8_0,
            _ => QuantFormat::Unknown,
        }
    }
    
    /// Obtiene el tamaño del grupo de cuantización efectivo
    pub fn effective_group_size(&self) -> usize {
        if self.group_size == 0 {
            16 // Legacy block size
        } else {
            self.group_size as usize
        }
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_v1_backward_compatibility() {
        let mut header_bytes = [0u8; 4096];
        header_bytes[0..4].copy_from_slice(b"GAJE");
        // version 0x000907
        header_bytes[4..8].copy_from_slice(&0x000907u32.to_le_bytes());
        // flags, num_tensors
        header_bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
        header_bytes[12..16].copy_from_slice(&12u32.to_le_bytes());
        // rest of layout fields are zero

        let header = FlatHeaderV2::from_bytes(&header_bytes).unwrap();
        assert_eq!(header.quantization_type(), QuantFormat::LegacyCentroids);
        assert_eq!(header.effective_group_size(), 16);
        assert_eq!(header.num_tensors, 12);
        assert_eq!(header.version, 0x000907);
    }

    #[test]
    fn test_header_v2_q4_0() {
        let mut header_bytes = [0u8; 4096];
        header_bytes[0..4].copy_from_slice(b"GAJE");
        header_bytes[4..8].copy_from_slice(&0x000908u32.to_le_bytes());
        // group_size = 32
        header_bytes[48..52].copy_from_slice(&32u32.to_le_bytes());
        // quant_format = 1 (Q4_0)
        header_bytes[52..56].copy_from_slice(&1u32.to_le_bytes());

        let header = FlatHeaderV2::from_bytes(&header_bytes).unwrap();
        assert_eq!(header.quantization_type(), QuantFormat::Q4_0);
        assert_eq!(header.effective_group_size(), 32);
        assert_eq!(header.version, 0x000908);
    }

    #[test]
    fn test_invalid_magic_bytes() {
        let mut header_bytes = [0u8; 4096];
        header_bytes[0..4].copy_from_slice(b"XGAJ");
        
        let res = FlatHeaderV2::from_bytes(&header_bytes);
        assert!(res.is_err());
        assert!(matches!(res.err().unwrap(), HeaderError::InvalidMagic));
    }

    #[test]
    fn test_q4_0_block_roundtrip() {
        let f32_weights: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect(); // 0.0 to 3.1
        let min_val = 0.0f32;
        let max_val = 3.1f32;
        let scale = (max_val - min_val) / 15.0;
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

        let mut qs = [0u8; 16];
        for k in 0..16 {
            let q0 = (((f32_weights[k * 2] - min_val) * inv_scale).round().clamp(0.0, 15.0)) as u8;
            let q1 = (((f32_weights[k * 2 + 1] - min_val) * inv_scale).round().clamp(0.0, 15.0)) as u8;
            qs[k] = q0 | (q1 << 4);
        }

        let block = Q4_0Block {
            scale: half::f16::from_f32(scale),
            min: half::f16::from_f32(min_val),
            qs,
        };

        // Dequantize and check error bounds
        for i in 0..32 {
            let original = f32_weights[i];
            let dequantized = block.dequantize_weight(i);
            let err = (original - dequantized).abs();
            assert!(err <= 0.11, "Original {} vs dequantized {} error {} above step/2 limit", original, dequantized, err);
        }
    }
}

