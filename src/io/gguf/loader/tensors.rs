// =============================================================================
// tensors — lectura y genomización de tensores (F32/F16/Q8_0) + unpermute
// =============================================================================
use super::GGUFLoader;
use crate::io::gguf::GGMLType;
use crate::nn::linear::GenomicLinear;

impl GGUFLoader {
    pub(crate) fn load_f32_tensor_optional(&self, name: &str) -> Option<Vec<f32>> {
        if !self.reader.tensors.contains_key(name) {
            return None;
        }
        self.load_f32_tensor(name).ok()
    }

    pub(crate) fn load_f32_tensor(&self, name: &str) -> std::io::Result<Vec<f32>> {
        let data = self.reader.get_tensor_data(name)?;
        let info = self.reader.tensors.get(name).unwrap();
        match info.tensor_type {
            GGMLType::F32 => {
                let count = data.len() / 4;
                let mut res = vec![0.0f32; count];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr(),
                        res.as_mut_ptr() as *mut u8,
                        data.len(),
                    );
                }
                Ok(res)
            }
            GGMLType::F16 => {
                let count = data.len() / 2;
                let mut res = vec![0.0f32; count];
                let f16_ptr = data.as_ptr() as *const half::f16;
                for i in 0..count {
                    unsafe {
                        res[i] = (*f16_ptr.add(i)).to_f32();
                    }
                }
                Ok(res)
            }
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Tensor {} must be F32 or F16", name),
            )),
        }
    }

    pub(crate) fn genomize_tensor(
        &self,
        name: &str,
        block_size: usize,
        anchor_threshold: f32,
        unpermute: bool,
        n_head: usize,
        head_dim: usize,
        bias: Option<Vec<f32>>,
    ) -> std::io::Result<GenomicLinear> {
        let data = self.reader.get_tensor_data(name)?;
        let info = self.reader.tensors.get(name).unwrap();
        let out_features = info.shape[info.n_dims as usize - 1] as usize;
        let in_features = info.shape[0] as usize;
        let mut f32_data: Vec<f32> = match info.tensor_type {
            GGMLType::F32 => data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            GGMLType::F16 => {
                let count = data.len() / 2;
                let mut res = vec![0.0f32; count];
                let ptr = data.as_ptr() as *const half::f16;
                for i in 0..count {
                    unsafe {
                        res[i] = (*ptr.add(i)).to_f32();
                    }
                }
                res
            }
            GGMLType::Q8_0 => {
                crate::compute::math::dequantize_q8_0_core(data, out_features, in_features)
            }
            GGMLType::Q4_0 => {
                crate::compute::math::dequantize_q4_0_core(data, out_features, in_features)
            }
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Unsupported tensor type: {:?}", info.tensor_type),
                ))
            }
        };

        if unpermute && n_head > 0 && head_dim > 0 {
            unpermute_f32(&mut f32_data, n_head, head_dim, out_features, in_features);
        }

        // Cuantización 4-bit consistente para preservar fidelidad en atención y FFN
        let bit_depth = 4;

        let (dna, centroids, anchors_u8) = if bit_depth == 4 {
            crate::compute::math::genomize_4bit_core(&f32_data, block_size, anchor_threshold)
        } else {
            crate::compute::math::genomize_f32_core(&f32_data, block_size, anchor_threshold, None)
        };

        Ok(GenomicLinear::new(
            dna,
            anchors_u8,
            centroids,
            out_features,
            in_features,
            block_size,
            Vec::new(),
            1e-6,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            bias.unwrap_or_default(),
            bit_depth,
        ))
    }
}

fn unpermute_f32(
    data: &mut [f32],
    _n_head: usize,
    head_dim: usize,
    out_features: usize,
    in_features: usize,
) {
    let mut scratch = vec![0.0f32; out_features * in_features];
    for i in 0..out_features {
        let h = i / head_dim;
        let j = i % head_dim;
        let new_j = if j < head_dim / 2 {
            2 * j
        } else {
            2 * (j - head_dim / 2) + 1
        };
        let interleaved_i = h * head_dim + new_j;
        for k in 0..in_features {
            scratch[i * in_features + k] = data[interleaved_i * in_features + k];
        }
    }
    data.copy_from_slice(&scratch);
}
