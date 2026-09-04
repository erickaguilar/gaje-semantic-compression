// =============================================================================
// python — Bindings #[pymethods] de GenomicLLM (feature `python`)
// =============================================================================
#[cfg(feature = "python")]
use pyo3::prelude::*;


#[cfg(feature = "python")]
#[pymethods]
impl GenomicLLM {
    #[new]
    pub fn py_new(
        embeddings: GenomicLinear,
        blocks: Vec<RustGenomicBlock>,
        output_norm: Vec<f32>,
        lm_head: GenomicLinear,
        eps: f32,
    ) -> Self {
        GenomicLLM {
            embeddings,
            blocks,
            output_norm,
            lm_head,
            eps,
            k_wta_ratio: 0.50,
            topology: None,
            quantum_embeddings: None,
            gpu_layers: 0,
            use_gpu: false,
        }
    }

    #[getter]
    pub fn vocab_size(&self) -> usize {
        self.embeddings.out_features
    }

    #[getter]
    pub fn n_embd(&self) -> usize {
        self.embeddings.in_features
    }

    #[getter]
    pub fn embeddings(&self) -> GenomicLinear {
        self.embeddings.clone()
    }

    #[getter]
    pub fn k_wta_ratio(&self) -> f32 {
        self.k_wta_ratio
    }

    pub fn set_k_wta_ratio(&mut self, ratio: f32) -> PyResult<()> {
        self.k_wta_ratio = ratio;
        for block in &mut self.blocks {
            block.k_wta_ratio = ratio;
        }
        Ok(())
    }

    pub fn load_topology(&mut self, json_path: &str) -> PyResult<()> {
        self.load_topology_core(json_path)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        println!("[*] Topología Relacional inyectada desde: {}", json_path);
        Ok(())
    }

    pub fn forward(&mut self, token_id: usize, clear_cache: bool) -> PyResult<Vec<f32>> {
        self.forward_core(token_id, clear_cache)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    pub fn forward_with_hidden(
        &mut self,
        token_id: usize,
        clear_cache: bool,
    ) -> PyResult<(Vec<f32>, Vec<f32>)> {
        self.forward_with_hidden_core(token_id, clear_cache)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    pub fn refine_lm_head(
        &mut self,
        hidden_state: Vec<f32>,
        grad_logits: Vec<f32>,
        lr: f32,
    ) -> PyResult<()> {
        self.lm_head
            .refine_with_grads_core(hidden_state, grad_logits, lr)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    pub fn train_on_sequence(&mut self, tokens: Vec<usize>, lr: f32) -> PyResult<f32> {
        self.train_on_sequence_core(tokens, lr)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }
    pub fn train_sequence_body(&mut self, tokens: Vec<usize>, lr: f32) -> PyResult<f32> {
        self.train_sequence_body_core(tokens, lr)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }
    pub fn train_sequence_full_body(&mut self, tokens: Vec<usize>, lr: f32) -> PyResult<f32> {
        self.train_sequence_full_body_core(tokens, lr)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }
    pub fn clear_cache_py(&mut self) -> PyResult<()> {
        self.clear_cache_core();
        Ok(())
    }

    #[pyo3(signature = (prompt_tokens, max_new_tokens=30, temperature=0.7, repetition_penalty=1.0, eos_token_ids=vec![2, 151643, 151645]))]
    pub fn generate_native_py(
        &mut self,
        prompt_tokens: Vec<usize>,
        max_new_tokens: usize,
        temperature: f32,
        repetition_penalty: f32,
        eos_token_ids: Vec<usize>,
    ) -> PyResult<Vec<usize>> {
        self.generate_native_core(
            prompt_tokens,
            max_new_tokens,
            temperature,
            repetition_penalty,
            eos_token_ids,
        )
        .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    pub fn recalibrate_all_centroids(&mut self, _shift: f32) -> PyResult<()> {
        // ...
        Ok(())
    }

    pub fn apply_vector_equilibrium_alignment_all(&mut self, strength: f32) -> PyResult<()> {
        self.embeddings
            .apply_vector_equilibrium_alignment_core(strength)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        for block in &mut self.blocks {
            block
                .q_gen
                .apply_vector_equilibrium_alignment_core(strength)
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            block
                .k_gen
                .apply_vector_equilibrium_alignment_core(strength)
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            block
                .v_gen
                .apply_vector_equilibrium_alignment_core(strength)
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            block
                .w_o
                .apply_vector_equilibrium_alignment_core(strength)
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            block
                .gate_gen
                .apply_vector_equilibrium_alignment_core(strength)
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            block
                .up_gen
                .apply_vector_equilibrium_alignment_core(strength)
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            block
                .w_down
                .apply_vector_equilibrium_alignment_core(strength)
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
        }
        self.lm_head
            .apply_vector_equilibrium_alignment_core(strength)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(())
    }

    pub fn mutate_all_homeostasis(&mut self, scale: f32) -> PyResult<Vec<f32>> {
        self.mutate_all_homeostasis_core(scale)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    pub fn undo_homeostasis_mutation(&mut self, deltas: Vec<f32>) -> PyResult<()> {
        self.undo_homeostasis_mutation_core(deltas)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    pub fn load_quantum_embeddings_bytes(&mut self, data: &[u8]) -> PyResult<()> {
        let table = crate::core::quantum_codebook::QuantumEmbeddingTableNative::from_bytes(data)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        self.quantum_embeddings = Some(std::sync::Arc::new(table));
        Ok(())
    }

    pub fn has_quantum_embeddings(&self) -> bool {
        self.quantum_embeddings.is_some()
    }

    pub fn unload_quantum_embeddings(&mut self) -> PyResult<()> {
        self.quantum_embeddings = None;
        Ok(())
    }

    #[pyo3(name = "is_gpu_active")]
    pub fn py_is_gpu_active(&self) -> bool {
        self.is_gpu_active()
    }

    #[pyo3(name = "offload_to_gpu")]
    pub fn py_offload_to_gpu(&mut self, layers: usize) -> PyResult<usize> {
        self.offload_to_gpu(layers)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }
}

