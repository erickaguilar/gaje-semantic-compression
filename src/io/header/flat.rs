// =============================================================================
// flat — FlatHeaderV2: cabecera binaria de 4096 bytes del formato .flat v2
// =============================================================================
use crate::io::header::types::{HeaderError, QuantFormat};

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

    // === ARCHITECTURE DESCRIPTOR (24 bytes) ===
    pub arch_family: u32,
    pub arch_n_embd: u32,
    pub arch_n_head: u32,
    pub arch_n_head_kv: u32,
    pub arch_n_blocks: u32,
    pub arch_qk_permute: u32,

    // === GTOK EMBEDDED TOKENIZER (16 bytes) ===
    pub gtok_offset: u64,
    pub gtok_len: u64,

    // === SECCIÓN DE ADAPTACIÓN GENÓMICA Y DELTAS (48 bytes) ===
    pub adapt_offset: u64,
    pub adapt_len: u64,
    pub num_overrides: u32,
    pub num_mutations: u32,
    pub lineage_parent_hash: u64,
    pub lineage_current_hash: u64,
    pub adapt_flags: u32,
    pub _pad_adapt: u32,

    // === RESERVA (3952 bytes) ===
    pub reserved: [u8; 3952],
}

pub type FlatHeaderV3 = FlatHeaderV2;

impl FlatHeaderV2 {
    pub const SIZE: usize = 4096;

    /// Lee el header desde un slice de bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, HeaderError> {
        if bytes.len() < Self::SIZE {
            return Err(HeaderError::TooShort);
        }

        let header = unsafe { std::ptr::read(bytes.as_ptr() as *const Self) };

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
            3 => QuantFormat::Q2_0,
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

    /// Indica si el modelo contiene una sección de adaptación genómica activa
    pub fn has_adaptive_section(&self) -> bool {
        self.adapt_len > 0 && self.adapt_offset > 0
    }

    /// Devuelve el descriptor de arquitectura si está presente en el header
    pub fn architecture_descriptor(&self) -> Option<crate::io::arch::ArchitectureDescriptor> {
        if self.arch_family == 0 {
            return None;
        }

        let family = match self.arch_family {
            1 => crate::io::arch::ModelFamily::Llama,
            2 => crate::io::arch::ModelFamily::SmolLM,
            3 => crate::io::arch::ModelFamily::Qwen2,
            4 => crate::io::arch::ModelFamily::Qwen2_5,
            5 => crate::io::arch::ModelFamily::Gemma,
            _ => crate::io::arch::ModelFamily::Unknown,
        };

        let head_dim = if self.arch_n_head > 0 {
            self.arch_n_embd as usize / self.arch_n_head as usize
        } else {
            0
        };

        let (rope_base, rope_style, ffn_act, chat_template) = match family {
            crate::io::arch::ModelFamily::Llama => (
                10000.0f32,
                "split".to_string(),
                "swiglu".to_string(),
                "llama".to_string(),
            ),
            crate::io::arch::ModelFamily::SmolLM => (
                100000.0f32,
                "split".to_string(),
                "swiglu".to_string(),
                "chatml".to_string(),
            ),
            crate::io::arch::ModelFamily::Qwen2 | crate::io::arch::ModelFamily::Qwen2_5 => (
                1000000.0f32,
                "split".to_string(),
                "swiglu".to_string(),
                "chatml".to_string(),
            ),
            crate::io::arch::ModelFamily::Gemma => (
                10000.0f32,
                "interleaved".to_string(),
                "geglu".to_string(),
                "gemma".to_string(),
            ),
            _ => (
                10000.0f32,
                "split".to_string(),
                "silu".to_string(),
                "standard".to_string(),
            ),
        };

        Some(crate::io::arch::ArchitectureDescriptor {
            family,
            n_embd: self.arch_n_embd as usize,
            n_head: self.arch_n_head as usize,
            n_head_kv: self.arch_n_head_kv as usize,
            n_blocks: self.arch_n_blocks as usize,
            head_dim,
            rope_base,
            rope_style,
            ffn_act,
            qk_permute: self.arch_qk_permute != 0,
            chat_template,
        })
    }
}
