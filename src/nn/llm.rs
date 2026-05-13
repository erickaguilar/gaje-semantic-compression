use pyo3::prelude::*;
use crate::nn::linear::GenomicLinear;
use crate::nn::block::RustGenomicBlock;
use crate::kernels::rms_norm_neon;

#[pyclass]
pub struct RustGenomicLLM {
    #[pyo3(get)]
    pub embeddings: GenomicLinear,
    #[pyo3(get)]
    pub blocks: Vec<RustGenomicBlock>,
    #[pyo3(get)]
    pub output_norm: Vec<f32>,
    #[pyo3(get)]
    pub lm_head: GenomicLinear,
    #[pyo3(get)]
    pub eps: f32,
}

#[pymethods]
impl RustGenomicLLM {
    #[new]
    pub fn new(
        embeddings: GenomicLinear,
        blocks: Vec<RustGenomicBlock>,
        output_norm: Vec<f32>,
        lm_head: GenomicLinear,
        eps: f32,
    ) -> Self {
        RustGenomicLLM {
            embeddings,
            blocks,
            output_norm,
            lm_head,
            eps,
        }
    }

    pub fn forward(&mut self, token_id: usize, clear_cache: bool) -> PyResult<Vec<f32>> {
        if clear_cache {
            for block in &mut self.blocks {
                block.clear_cache()?;
            }
        }

        // The position is determined by the cache length of the first block's attention
        let pos = if self.blocks.is_empty() { 0 } else { self.blocks[0].attn.k_cache_len() };

        // Ensure token_id is within bounds (out_features holds vocabulary size here)
        if token_id >= self.embeddings.out_features {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("Token id {} out of bounds", token_id)));
        }

        // 1. Fetch embedding manually (similar to get_row in Python)
        let _n_blocks = self.embeddings.in_features / self.embeddings.block_size;
        let mut h = vec![0.0f32; self.embeddings.in_features]; 

        // Correct Embedding Retrieval:
        h = self.embeddings.get_row(token_id)?;

        // 2. Pass through all blocks
        for block in &mut self.blocks {
            h = block.forward(h, pos)?;
        }

        // 3. Final RMSNorm
        let h_norm = unsafe { rms_norm_neon(&h, &self.output_norm, self.eps) };

        // 4. LM Head Projection
        let logits = self.lm_head.forward(h_norm)?;

        Ok(logits)
    }
}