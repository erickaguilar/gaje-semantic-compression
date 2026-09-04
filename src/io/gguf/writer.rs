// =============================================================================
// writer — GGUFWriter: serialización binaria del formato GGUF (v3)
// =============================================================================
//
// Inverso de `io/gguf/reader.rs`: escribe cabecera, metadatos key-value, infos
// de tensores y datos crudos con alineación correcta (por defecto 32 bytes,
// según `general.alignment`).
//
// Reutiliza los tipos de `io/gguf/types.rs` sin modificarlos.
use std::io::Write;

use crate::io::gguf::reader::tensor_size_bytes;
use crate::io::gguf::types::{GGMLType, GGUFTensorInfo, GGUFValue};

pub struct GGUFWriter {
    metadata: Vec<(String, GGUFValue)>,
    tensors: Vec<GGUFTensorInfo>,
    tensor_data: Vec<Vec<u8>>,
    #[allow(dead_code)]
    alignment: u64,
}

impl GGUFWriter {
    pub fn new() -> Self {
        Self {
            metadata: Vec::new(),
            tensors: Vec::new(),
            tensor_data: Vec::new(),
            alignment: 32,
        }
    }

    pub fn add_metadata(&mut self, key: impl Into<String>, value: GGUFValue) {
        self.metadata.push((key.into(), value));
    }

    pub fn add_tensor(
        &mut self,
        name: impl Into<String>,
        shape: Vec<u64>,
        tensor_type: GGMLType,
        data: Vec<u8>,
    ) -> Result<(), String> {
        if !matches!(tensor_type, GGMLType::F32 | GGMLType::F16 | GGMLType::Q8_0) {
            return Err(format!(
                "Unsupported tensor type for writer: {:?}",
                tensor_type
            ));
        }
        if shape.is_empty() {
            return Err("Tensor shape must not be empty".into());
        }
        let info = GGUFTensorInfo {
            name: name.into(),
            n_dims: shape.len() as u32,
            shape,
            tensor_type,
            offset: 0,
        };
        let expected = tensor_size_bytes(&info);
        if data.len() != expected {
            return Err(format!(
                "Tensor '{}' data size {} bytes does not match expected {} for {:?}",
                info.name,
                data.len(),
                expected,
                tensor_type
            ));
        }
        self.tensors.push(info);
        self.tensor_data.push(data);
        Ok(())
    }

    pub fn write<W: Write>(&self, mut w: W) -> std::io::Result<()> {
        let alignment = self.alignment();
        let data_offset = self.compute_data_offset(alignment);
        let tensor_offsets = self.compute_tensor_offsets(alignment);

        // 1. Cabecera
        w.write_all(b"GGUF")?;
        Self::write_u32(&mut w, 3)?;
        Self::write_u64(&mut w, self.tensors.len() as u64)?;
        Self::write_u64(&mut w, self.metadata.len() as u64)?;

        // 2. Metadatos key-value
        for (key, value) in &self.metadata {
            Self::write_string(&mut w, key)?;
            Self::write_u32(&mut w, value.value_type())?;
            Self::write_value(&mut w, value)?;
        }

        // 3. Infos de tensores (con offset calculado)
        for (info, offset) in self.tensors.iter().zip(tensor_offsets.iter()) {
            Self::write_string(&mut w, &info.name)?;
            Self::write_u32(&mut w, info.n_dims)?;
            for dim in &info.shape {
                Self::write_u64(&mut w, *dim)?;
            }
            Self::write_u32(&mut w, info.tensor_type as u32)?;
            Self::write_u64(&mut w, *offset)?;
        }

        // 4. Datos con padding a `alignment` desde data_offset
        let header_bytes = data_offset - self.full_header_size();
        w.write_all(&vec![0u8; header_bytes as usize])?;

        let mut pos = data_offset;
        for data in self.tensor_data.iter() {
            let pad = (alignment - (pos % alignment)) % alignment;
            w.write_all(&vec![0u8; pad as usize])?;
            pos += pad;
            w.write_all(data)?;
            pos += data.len() as u64;
        }

        Ok(())
    }

    fn alignment(&self) -> u64 {
        for (key, value) in &self.metadata {
            if key == "general.alignment" {
                if let GGUFValue::Uint32(a) = value {
                    return (*a as u64).max(1);
                }
            }
        }
        32
    }

    /// Offset (relativo a `data_offset`) de cada tensor, acumulando tamaño +
    /// padding de alineación.
    fn compute_tensor_offsets(&self, alignment: u64) -> Vec<u64> {
        let mut offsets = Vec::with_capacity(self.tensors.len());
        let mut cur = 0u64;
        for data in self.tensor_data.iter() {
            cur = (cur + alignment - 1) / alignment * alignment;
            offsets.push(cur);
            cur += data.len() as u64;
        }
        offsets
    }

    fn compute_data_offset(&self, alignment: u64) -> u64 {
        let header = self.full_header_size();
        (header + alignment - 1) / alignment * alignment
    }

    /// Tamaño completo de la cabecera (magic + versión + counts + metadatos + infos).
    fn full_header_size(&self) -> u64 {
        4 + 4 + 8 + 8 + self.metadata_and_tensor_info_size()
    }

    fn metadata_and_tensor_info_size(&self) -> u64 {
        let meta: u64 = self
            .metadata
            .iter()
            .map(|(k, v)| 8 + k.len() as u64 + 4 + value_size(v))
            .sum();
        let tensors: u64 = self
            .tensors
            .iter()
            .map(|t| 8 + t.name.len() as u64 + 4 + 8 * t.n_dims as u64 + 4 + 8)
            .sum();
        meta + tensors
    }

    // Helpers de escritura (inversos de los privados del reader)
    fn write_u32<W: Write>(w: &mut W, v: u32) -> std::io::Result<()> {
        w.write_all(&v.to_le_bytes())
    }

    fn write_u64<W: Write>(w: &mut W, v: u64) -> std::io::Result<()> {
        w.write_all(&v.to_le_bytes())
    }

    fn write_string<W: Write>(w: &mut W, s: &str) -> std::io::Result<()> {
        Self::write_u64(w, s.len() as u64)?;
        w.write_all(s.as_bytes())
    }

    fn write_value<W: Write>(w: &mut W, value: &GGUFValue) -> std::io::Result<()> {
        match value {
            GGUFValue::Uint8(v) => w.write_all(&[*v]),
            GGUFValue::Int8(v) => w.write_all(&[*v as u8]),
            GGUFValue::Uint16(v) => w.write_all(&v.to_le_bytes()),
            GGUFValue::Int16(v) => w.write_all(&v.to_le_bytes()),
            GGUFValue::Uint32(v) => Self::write_u32(w, *v),
            GGUFValue::Int32(v) => Self::write_u32(w, *v as u32),
            GGUFValue::Float32(v) => w.write_all(&v.to_le_bytes()),
            GGUFValue::Bool(v) => w.write_all(&[*v as u8]),
            GGUFValue::String(s) => Self::write_string(w, s),
            GGUFValue::Uint64(v) => Self::write_u64(w, *v),
            GGUFValue::Int64(v) => Self::write_u64(w, *v as u64),
            GGUFValue::Float64(v) => w.write_all(&v.to_le_bytes()),
            GGUFValue::Array(items) => {
                let item_type = items.first().map(|i| i.value_type()).unwrap_or(0);
                Self::write_u32(w, item_type)?;
                Self::write_u64(w, items.len() as u64)?;
                for item in items {
                    Self::write_value(w, item)?;
                }
                Ok(())
            }
        }
    }
}

impl Default for GGUFWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Tamaño en bytes que `write_value` emite para un valor (sin el id de tipo).
fn value_size(v: &GGUFValue) -> u64 {
    match v {
        GGUFValue::Uint8(_) | GGUFValue::Int8(_) | GGUFValue::Bool(_) => 1,
        GGUFValue::Uint16(_) | GGUFValue::Int16(_) => 2,
        GGUFValue::Uint32(_) | GGUFValue::Int32(_) | GGUFValue::Float32(_) => 4,
        GGUFValue::Uint64(_) | GGUFValue::Int64(_) | GGUFValue::Float64(_) => 8,
        GGUFValue::String(s) => 8 + s.len() as u64,
        GGUFValue::Array(items) => 12 + items.iter().map(value_size).sum::<u64>(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::gguf::reader::GGUFReader;

    #[test]
    fn roundtrip_write_then_read() {
        let mut writer = GGUFWriter::new();
        writer.add_metadata("general.name", GGUFValue::String("test".into()));
        writer.add_metadata("general.alignment", GGUFValue::Uint32(32));

        // Tensor F32 2x3 = 6 elementos = 24 bytes
        let f32_data: Vec<u8> = (0..6).map(|i| (i as f32).to_le_bytes()).flatten().collect();
        writer
            .add_tensor("tensor.f32", vec![2, 3], GGMLType::F32, f32_data)
            .unwrap();

        // Tensor Q8_0: 40 elementos -> 2 bloques de 32 = 68 bytes
        let q8_data = vec![7u8; 68];
        writer
            .add_tensor("tensor.q8", vec![40], GGMLType::Q8_0, q8_data)
            .unwrap();

        let mut buf = Vec::new();
        writer.write(&mut buf).unwrap();

        let reader = GGUFReader::open_from_bytes(&buf).unwrap();
        assert!(matches!(
            reader.metadata.get("general.name"),
            Some(GGUFValue::String(s)) if s == "test"
        ));
        assert!(matches!(
            reader.metadata.get("general.alignment"),
            Some(GGUFValue::Uint32(32))
        ));

        assert_eq!(reader.tensors.len(), 2);
        let t32 = reader.tensors.get("tensor.f32").unwrap();
        assert_eq!(t32.shape, vec![2, 3]);
        assert_eq!(t32.tensor_type, GGMLType::F32);
        let tq8 = reader.tensors.get("tensor.q8").unwrap();
        assert_eq!(tq8.tensor_type, GGMLType::Q8_0);

        assert_eq!(reader.get_tensor_data("tensor.f32").unwrap().len(), 24);
        assert_eq!(reader.get_tensor_data("tensor.q8").unwrap().len(), 68);
        assert_eq!(
            reader.get_tensor_data("tensor.f32").unwrap(),
            &(0..6)
                .map(|i| (i as f32).to_le_bytes())
                .flatten()
                .collect::<Vec<u8>>()[..]
        );
        assert_eq!(reader.get_tensor_data("tensor.q8").unwrap(), &[7u8; 68][..]);
    }
}
