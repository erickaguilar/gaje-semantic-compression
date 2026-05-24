use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, Read, Seek, SeekFrom};
use memmap2::Mmap;

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

pub struct GGUFReader {
    pub metadata: HashMap<String, GGUFValue>,
    pub tensors: HashMap<String, GGUFTensorInfo>,
    mmap: Mmap,
    data_offset: u64,
}

impl GGUFReader {
    pub fn open(path: &str) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mut reader = Cursor::new(&mmap);
        
        // 1. Magic
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != b"GGUF" {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Not a GGUF file"));
        }

        // 2. Version
        let version = Self::read_u32(&mut reader)?;
        if version != 2 && version != 3 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("Unsupported GGUF version: {}", version)));
        }

        // 3. Counts
        let tensor_count = Self::read_u64(&mut reader)?;
        let metadata_kv_count = Self::read_u64(&mut reader)?;

        // 4. Metadata
        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let key = Self::read_string(&mut reader)?;
            let val_type = Self::read_u32(&mut reader)?;
            let value = Self::read_value(&mut reader, val_type)?;
            metadata.insert(key, value);
        }

        // 5. Tensor Infos
        let mut tensors = HashMap::new();
        for _ in 0..tensor_count {
            let name = Self::read_string(&mut reader)?;
            let n_dims = Self::read_u32(&mut reader)?;
            let mut shape = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                shape.push(Self::read_u64(&mut reader)?);
            }
            let tensor_type_id = Self::read_u32(&mut reader)?;
            let tensor_type = match tensor_type_id {
                0 => GGMLType::F32,
                1 => GGMLType::F16,
                8 => GGMLType::Q8_0,
                // Add more as needed
                _ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("Unsupported tensor type: {}", tensor_type_id))),
            };
            let offset = Self::read_u64(&mut reader)?;
            
            tensors.insert(name.clone(), GGUFTensorInfo {
                name,
                n_dims,
                shape,
                tensor_type,
                offset,
            });
        }

        let current_pos = reader.position();
        // GGUF alignment (usually 32 bytes)
        let alignment = if let Some(GGUFValue::Uint32(a)) = metadata.get("general.alignment") {
            *a as u64
        } else {
            32u64
        };
        
        let data_offset = (current_pos + alignment - 1) / alignment * alignment;

        Ok(GGUFReader {
            metadata,
            tensors,
            mmap,
            data_offset,
        })
    }

    pub fn get_tensor_data(&self, name: &str) -> std::io::Result<&[u8]> {
        let info = self.tensors.get(name)
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, format!("Tensor not found: {}", name)))?;
        
        let size = self.get_tensor_size_bytes(info);
        let start = (self.data_offset + info.offset) as usize;
        let end = start + size;
        
        if end > self.mmap.len() {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Tensor offset out of bounds"));
        }
        
        Ok(&self.mmap[start..end])
    }

    fn get_tensor_size_bytes(&self, info: &GGUFTensorInfo) -> usize {
        let n_elements: u64 = info.shape.iter().product();
        match info.tensor_type {
            GGMLType::F32 => (n_elements * 4) as usize,
            GGMLType::F16 => (n_elements * 2) as usize,
            GGMLType::Q8_0 => {
                // Q8_0: blocks of 32 elements. Each block: 1 f16 delta + 32 i8 weights = 34 bytes.
                ((n_elements + 31) / 32 * 34) as usize
            }
            _ => 0, // Should be handled during loading
        }
    }

    // Helper functions for reading binary types
    fn read_u32<R: Read>(r: &mut R) -> std::io::Result<u32> {
        let mut buf = [0u8; 4];
        r.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_u64<R: Read>(r: &mut R) -> std::io::Result<u64> {
        let mut buf = [0u8; 8];
        r.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    fn read_string<R: Read>(r: &mut R) -> std::io::Result<String> {
        let len = Self::read_u64(r)?;
        let mut buf = vec![0u8; len as usize];
        r.read_exact(&mut buf)?;
        String::from_utf8(buf).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    fn read_value<R: Read>(r: &mut R, val_type: u32) -> std::io::Result<GGUFValue> {
        match val_type {
            0 => { // Uint8
                let mut buf = [0u8; 1];
                r.read_exact(&mut buf)?;
                Ok(GGUFValue::Uint8(buf[0]))
            }
            1 => { // Int8
                let mut buf = [0u8; 1];
                r.read_exact(&mut buf)?;
                Ok(GGUFValue::Int8(buf[0] as i8))
            }
            2 => { // Uint16
                let mut buf = [0u8; 2];
                r.read_exact(&mut buf)?;
                Ok(GGUFValue::Uint16(u16::from_le_bytes(buf)))
            }
            3 => { // Int16
                let mut buf = [0u8; 2];
                r.read_exact(&mut buf)?;
                Ok(GGUFValue::Int16(i16::from_le_bytes(buf)))
            }
            4 => Ok(GGUFValue::Uint32(Self::read_u32(r)?)),
            5 => { // Int32
                let mut buf = [0u8; 4];
                r.read_exact(&mut buf)?;
                Ok(GGUFValue::Int32(i32::from_le_bytes(buf)))
            }
            6 => { // Float32
                let mut buf = [0u8; 4];
                r.read_exact(&mut buf)?;
                Ok(GGUFValue::Float32(f32::from_le_bytes(buf)))
            }
            7 => { // Bool
                let mut buf = [0u8; 1];
                r.read_exact(&mut buf)?;
                Ok(GGUFValue::Bool(buf[0] != 0))
            }
            8 => Ok(GGUFValue::String(Self::read_string(r)?)),
            9 => { // Array
                let item_type = Self::read_u32(r)?;
                let len = Self::read_u64(r)?;
                let mut items = Vec::with_capacity(len as usize);
                for _ in 0..len {
                    items.push(Self::read_value(r, item_type)?);
                }
                Ok(GGUFValue::Array(items))
            }
            10 => Ok(GGUFValue::Uint64(Self::read_u64(r)?)),
            11 => Ok(GGUFValue::Int64(Self::read_u64(r)? as i64)),
            12 => { // Float64
                let mut buf = [0u8; 8];
                r.read_exact(&mut buf)?;
                Ok(GGUFValue::Float64(f64::from_le_bytes(buf)))
            }
            _ => Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("Unsupported metadata value type: {}", val_type))),
        }
    }
}
