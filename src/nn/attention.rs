use crate::compute::kernels::*;
use half::f16;
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct GenomicAttention {
    #[pyo3(get)]
    pub n_head: usize,
    #[pyo3(get)]
    pub n_head_kv: usize,
    #[pyo3(get)]
    pub head_dim: usize,
    pub k_cache: Vec<Vec<f16>>,
    pub v_cache: Vec<Vec<f16>>,
    #[pyo3(get)]
    pub rmsnorm_weight: Vec<f32>,
    #[pyo3(get)]
    pub eps: f32,
    #[pyo3(get)]
    pub rope_base: f32,
}

#[pymethods]
impl GenomicAttention {
    #[new]
    #[pyo3(signature = (n_head, n_head_kv, head_dim, rmsnorm_weight=Vec::new(), eps=1e-6, rope_base=10000.0))]
    pub fn new(
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        rmsnorm_weight: Vec<f32>,
        eps: f32,
        rope_base: f32,
    ) -> Self {
        GenomicAttention {
            n_head,
            n_head_kv,
            head_dim,
            k_cache: Vec::new(),
            v_cache: Vec::new(),
            rmsnorm_weight,
            eps,
            rope_base,
        }
    }

    #[getter]
    pub fn k_cache(&self) -> Vec<Vec<f32>> {
        self.k_cache
            .iter()
            .map(|v| v.iter().map(|&x| x.to_f32()).collect())
            .collect()
    }

    #[getter]
    pub fn v_cache(&self) -> Vec<Vec<f32>> {
        self.v_cache
            .iter()
            .map(|v| v.iter().map(|&x| x.to_f32()).collect())
            .collect()
    }

    #[getter]
    pub fn k_cache_len(&self) -> usize {
        self.k_cache.len()
    }

    pub fn apply_rmsnorm(&self, input: Vec<f32>) -> PyResult<Vec<f32>> {
        if self.rmsnorm_weight.is_empty() {
            return Ok(input);
        }
        Ok(unsafe { rms_norm_neon(&input, &self.rmsnorm_weight, self.eps) })
    }

    pub fn forward_attention(
        &mut self,
        q: Vec<f32>,
        k: Vec<f32>,
        v: Vec<f32>,
        pos: usize,
    ) -> PyResult<Vec<f32>> {
        let head_dim = self.head_dim;
        let n_head = self.n_head;
        let n_head_kv = self.n_head_kv;
        let n_groups = n_head / n_head_kv;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let rope_base = self.rope_base;

        let mut q_rope = q.clone();
        let mut k_rope = k.clone();

        // 1. Apply RoPE to Q and K (using Interleaved format for Qwen2/Llama)
        for h in 0..n_head {
            let h_start = h * head_dim;
            for i in 0..(head_dim / 2) {
                let freq = 1.0 / rope_base.powf((2 * i) as f32 / head_dim as f32);
                let theta = pos as f32 * freq;
                let (sin, cos) = theta.sin_cos();
                let v0 = q_rope[h_start + 2 * i];
                let v1 = q_rope[h_start + 2 * i + 1];
                q_rope[h_start + 2 * i] = v0 * cos - v1 * sin;
                q_rope[h_start + 2 * i + 1] = v0 * sin + v1 * cos;
            }
        }

        for h in 0..n_head_kv {
            let h_start = h * head_dim;
            for i in 0..(head_dim / 2) {
                let freq = 1.0 / rope_base.powf((2 * i) as f32 / head_dim as f32);
                let theta = pos as f32 * freq;
                let (sin, cos) = theta.sin_cos();
                let v0 = k_rope[h_start + 2 * i];
                let v1 = k_rope[h_start + 2 * i + 1];
                k_rope[h_start + 2 * i] = v0 * cos - v1 * sin;
                k_rope[h_start + 2 * i + 1] = v0 * sin + v1 * cos;
            }
        }

        // 2. Store in Cache as F16
        self.k_cache
            .push(k_rope.into_iter().map(f16::from_f32).collect());
        self.v_cache
            .push(v.into_iter().map(f16::from_f32).collect());

        let k_cache = &self.k_cache;
        let v_cache = &self.v_cache;
        let seq_len = k_cache.len();

        // 3. Compute Attention Scores and Output
        let attn_out: Vec<f32> = (0..n_head)
            .into_par_iter()
            .flat_map(|h| {
                let kv_h = h / n_groups;
                // The KV cache stores the entire sequence of K and V vectors for all KV heads.
                // Each entry in k_cache[t] is [n_head_kv * head_dim].
                let kv_h_offset = kv_h * head_dim;

                let q_slice = &q_rope[h * head_dim..(h + 1) * head_dim];

                let mut scores = vec![0.0f32; seq_len];
                let mut max_score = -f32::INFINITY;

                for t in 0..seq_len {
                    let k_slice = &k_cache[t][kv_h_offset..kv_h_offset + head_dim];
                    let mut score = 0.0f32;
                    for i in 0..head_dim {
                        score += q_slice[i] * k_slice[i].to_f32();
                    }
                    score *= scale;
                    scores[t] = score;
                    if score > max_score {
                        max_score = score;
                    }
                }

                let mut sum_exp = 0.0f32;
                for t in 0..seq_len {
                    let s = (scores[t] - max_score).exp();
                    scores[t] = s;
                    sum_exp += s;
                }
                let inv_sum = 1.0 / (sum_exp + 1e-12);

                let mut head_out = vec![0.0f32; head_dim];
                for t in 0..seq_len {
                    let weight = scores[t] * inv_sum;
                    let v_slice = &v_cache[t][kv_h_offset..kv_h_offset + head_dim];
                    for i in 0..head_dim {
                        head_out[i] += weight * v_slice[i].to_f32();
                    }
                }
                head_out
            })
            .collect();

        Ok(attn_out)
    }

    pub fn clear_cache(&mut self) -> PyResult<()> {
        self.k_cache.clear();
        self.v_cache.clear();
        Ok(())
    }
}
