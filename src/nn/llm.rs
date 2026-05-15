use crate::kernels::rms_norm_neon;
use crate::nn::block::RustGenomicBlock;
use crate::nn::linear::GenomicLinear;
use pyo3::prelude::*;

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
            self.clear_cache()?;
        }

        // The position is exactly the number of tokens already in the cache
        let pos = if self.blocks.is_empty() {
            0
        } else {
            self.blocks[0].attn.k_cache_len()
        };

        if token_id >= self.embeddings.out_features {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Token id {} out of bounds",
                token_id
            )));
        }

        // 1. Fetch embedding
        let mut h = self.embeddings.get_row(token_id)?;

        // 2. Pass through all blocks (position is updated per token)
        for block in &mut self.blocks {
            h = block.forward(h, pos)?;
        }

        // 3. Final RMSNorm
        let h_norm = unsafe { rms_norm_neon(&h, &self.output_norm, self.eps) };

        // 4. LM Head Projection
        let logits = self.lm_head.forward(h_norm)?;

        Ok(logits)
    }

    pub fn embeddings_forward(&self, token_id: usize) -> PyResult<Vec<f32>> {
        if token_id >= self.embeddings.out_features {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Token id {} out of bounds",
                token_id
            )));
        }
        self.embeddings.get_row(token_id)
    }

    pub fn clear_cache(&mut self) -> PyResult<()> {
        for block in &mut self.blocks {
            block.clear_cache()?;
        }
        Ok(())
    }
}
