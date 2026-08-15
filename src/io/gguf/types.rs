// =============================================================================
// types — Tipos básicos del formato GGUF
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GGUFValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

#[derive(Debug, Clone)]
pub enum GGUFValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GGUFValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl GGUFValue {
    /// Tipo GGUF (id numérico) correspondiente a esta variante. Inverso de la
    /// lectura de `val_type` en el reader.
    pub fn value_type(&self) -> u32 {
        match self {
            GGUFValue::Uint8(_) => GGUFValueType::Uint8 as u32,
            GGUFValue::Int8(_) => GGUFValueType::Int8 as u32,
            GGUFValue::Uint16(_) => GGUFValueType::Uint16 as u32,
            GGUFValue::Int16(_) => GGUFValueType::Int16 as u32,
            GGUFValue::Uint32(_) => GGUFValueType::Uint32 as u32,
            GGUFValue::Int32(_) => GGUFValueType::Int32 as u32,
            GGUFValue::Float32(_) => GGUFValueType::Float32 as u32,
            GGUFValue::Bool(_) => GGUFValueType::Bool as u32,
            GGUFValue::String(_) => GGUFValueType::String as u32,
            GGUFValue::Array(_) => GGUFValueType::Array as u32,
            GGUFValue::Uint64(_) => GGUFValueType::Uint64 as u32,
            GGUFValue::Int64(_) => GGUFValueType::Int64 as u32,
            GGUFValue::Float64(_) => GGUFValueType::Float64 as u32,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(non_camel_case_types)]
pub enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
}

#[derive(Debug, Clone)]
pub struct GGUFTensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub shape: Vec<u64>,
    pub tensor_type: GGMLType,
    pub offset: u64,
}
