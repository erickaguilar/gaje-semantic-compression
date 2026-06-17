use crate::compute::kernels::*;
use half::f16;
use rayon::prelude::*;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Núcleo de Atención Genómica (Pure Rust)
#[cfg_attr(feature = "python", pyclass)]
#[derive(Clone)]
pub struct GenomicAttention {
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub k_cache: Vec<Vec<f16>>,
    pub v_cache: Vec<Vec<f16>>,
    pub rmsnorm_weight: Vec<f32>,
    pub eps: f32,
    pub rope_base: f32,
    pub rope_style: String,
}

impl GenomicAttention {
    pub fn new(
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        rmsnorm_weight: Vec<f32>,
        eps: f32,
        rope_base: f32,
        rope_style: String,
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
            rope_style,
        }
    }

    pub fn forward_attention_core(
        &mut self,
        q: Vec<f32>,
        k: Vec<f32>,
        v: Vec<f32>,
        pos: usize,
    ) -> Result<Vec<f32>, String> {
        let head_dim = self.head_dim.max(1);
        let n_head = self.n_head;
        let n_head_kv = self.n_head_kv.max(1);
        let n_groups = (n_head / n_head_kv).max(1);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let rope_base = self.rope_base;
        let is_split = self.rope_style == "split";
        let mut q_rope = q;
        let mut k_rope = k;
        let apply_rope = |vec: &mut [f32], heads: usize| {
            for h in 0..heads {
                let h_start = h * head_dim;
                for i in 0..(head_dim / 2) {
                    let freq = 1.0 / (rope_base.powf((2.0 * i as f32) / head_dim as f32));
                    let theta = pos as f32 * freq;
                    let (sin, cos) = theta.sin_cos();
                    if is_split {
                        let v0 = vec[h_start + i];
                        let v1 = vec[h_start + i + head_dim / 2];
                        vec[h_start + i] = v0 * cos - v1 * sin;
                        vec[h_start + i + head_dim / 2] = v0 * sin + v1 * cos;
                    } else {
                        let v0 = vec[h_start + 2 * i];
                        let v1 = vec[h_start + 2 * i + 1];
                        vec[h_start + 2 * i] = v0 * cos - v1 * sin;
                        vec[h_start + 2 * i + 1] = v0 * sin + v1 * cos;
                    }
                }
            }
        };
        apply_rope(&mut q_rope, n_head);
        apply_rope(&mut k_rope, n_head_kv);
        self.k_cache
            .push(k_rope.into_iter().map(f16::from_f32).collect());
        self.v_cache
            .push(v.into_iter().map(f16::from_f32).collect());
        let seq_len = self.k_cache.len();
        let attn_out: Vec<f32> = (0..n_head)
            .into_par_iter()
            .flat_map(|h| {
                if h == 0 {
                    println!("[ENGINE CRITICAL] h=0, pos={}, cache_len_before={}, base={}, style={}", pos, self.k_cache.len(), self.rope_base, self.rope_style);
                }
                let kv_h = h / n_groups;
                let kv_h_off = kv_h * head_dim;
                let q_slice = &q_rope[h * head_dim..(h + 1) * head_dim];
                let mut scores = vec![0.0f32; seq_len];
                let mut max_s = -f32::INFINITY;
                for t in 0..seq_len {
                    let k_head = &self.k_cache[t][kv_h_off..kv_h_off + head_dim];
                    let mut k_f32 = [0.0f32; 256];
                    for (i, &kx) in k_head.iter().enumerate() {
                        k_f32[i] = kx.to_f32();
                    }
                    let s = unsafe { dot_product(q_slice, &k_f32[..head_dim]) } * scale;
                    scores[t] = s;
                    if s > max_s {
                        max_s = s;
                    }
                }
                if h == 0 && pos == 0 {
                    println!("[Debug Attn 0] max_score: {:.4}, seq_len: {}", max_s, seq_len);
                }
                let mut sum_e = 0.0f32;
                for t in 0..seq_len {
                    scores[t] = (scores[t] - max_s).exp();
                    sum_e += scores[t];
                }
                let inv_s = 1.0 / (sum_e + 1e-12);
                let mut head_out = vec![0.0f32; head_dim];
                for t in 0..seq_len {
                    let w = scores[t] * inv_s;
                    let v_head = &self.v_cache[t][kv_h_off..kv_h_off + head_dim];
                    for i in 0..head_dim {
                        head_out[i] += w * v_head[i].to_f32();
                    }
                }
                head_out
            })
            .collect();
        Ok(attn_out)
    }

    pub fn clear_cache_core(&mut self) {
        self.k_cache.clear();
        self.v_cache.clear();
    }
    pub fn k_cache_len(&self) -> usize {
        self.k_cache.len()
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl GenomicAttention {
    #[new]
    #[pyo3(signature = (n_head, n_head_kv, head_dim, rmsnorm_weight=Vec::new(), eps=1e-6, rope_base=10000.0, rope_style="split".to_string()))]
    pub fn py_new(
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        rmsnorm_weight: Vec<f32>,
        eps: f32,
        rope_base: f32,
        rope_style: String,
    ) -> Self {
        GenomicAttention::new(
            n_head,
            n_head_kv,
            head_dim,
            rmsnorm_weight,
            eps,
            rope_base,
            rope_style,
        )
    }
    pub fn forward_attention(
        &mut self,
        q: Vec<f32>,
        k: Vec<f32>,
        v: Vec<f32>,
        pos: usize,
    ) -> PyResult<Vec<f32>> {
        self.forward_attention_core(q, k, v, pos)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }
    pub fn clear_cache(&mut self) -> PyResult<()> {
        self.clear_cache_core();
        Ok(())
    }
}
