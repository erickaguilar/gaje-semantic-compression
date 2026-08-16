use crate::compute::kernels::*;
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
    pub k_cache: Vec<Vec<f32>>,
    pub v_cache: Vec<Vec<f32>>,
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
        let n_head = self.n_head;
        let n_head_kv = self.n_head_kv;
        let head_dim = self.head_dim;
        let n_groups = n_head / n_head_kv;
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
        self.k_cache.push(k_rope);
        self.v_cache.push(v);
        let seq_len = self.k_cache.len();
        let heads_out: Vec<Vec<f32>> = (0..n_head)
            .into_par_iter()
            .map(|h| {
                let kv_h = h / n_groups;
                let kv_h_off = kv_h * head_dim;
                let q_slice = &q_rope[h * head_dim..(h + 1) * head_dim];
                let mut scores = vec![0.0f32; seq_len];
                let mut max_s = -f32::INFINITY;
                for t in 0..seq_len {
                    let k_head = &self.k_cache[t][kv_h_off..kv_h_off + head_dim];
                    let s = unsafe { dot_product(q_slice, k_head) } * scale;
                    scores[t] = s;
                    if s > max_s {
                        max_s = s;
                    }
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
                        head_out[i] += w * v_head[i];
                    }
                }
                head_out
            })
            .collect();

        let mut attn_out = vec![0.0f32; n_head * head_dim];
        for h in 0..n_head {
            let start = h * head_dim;
            attn_out[start..start + head_dim].copy_from_slice(&heads_out[h]);
        }
        Ok(attn_out)
    }

    pub fn clear_cache_core(&mut self) {
        self.k_cache.clear();
        self.v_cache.clear();
    }
    pub fn k_cache_len(&self) -> usize {
        self.k_cache.len()
    }

    /// Forward de atención que además devuelve (salida, pesos softmax, q_rope)
    /// para poder hacer el backward con caché de activaciones (sin re-forward).
    pub fn forward_attention_cached(
        &mut self,
        q: Vec<f32>,
        k: Vec<f32>,
        v: Vec<f32>,
        pos: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
        let n_head = self.n_head;
        let n_head_kv = self.n_head_kv;
        let head_dim = self.head_dim;
        let n_groups = n_head / n_head_kv;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let (mut q_rope, mut k_rope) = (q, k);
        let apply_rope = |vec: &mut [f32], heads: usize, theta: f32| {
            for h in 0..heads {
                let h_start = h * head_dim;
                for i in 0..(head_dim / 2) {
                    let freq = 1.0 / (self.rope_base.powf((2.0 * i as f32) / head_dim as f32));
                    let t = theta * freq;
                    let (sin, cos) = t.sin_cos();
                    if self.rope_style == "split" {
                        let a = vec[h_start + i];
                        let b = vec[h_start + i + head_dim / 2];
                        vec[h_start + i] = a * cos - b * sin;
                        vec[h_start + i + head_dim / 2] = a * sin + b * cos;
                    } else {
                        let a = vec[h_start + 2 * i];
                        let b = vec[h_start + 2 * i + 1];
                        vec[h_start + 2 * i] = a * cos - b * sin;
                        vec[h_start + 2 * i + 1] = a * sin + b * cos;
                    }
                }
            }
        };
        apply_rope(&mut q_rope, n_head, pos as f32);
        apply_rope(&mut k_rope, n_head_kv, pos as f32);
        self.k_cache.push(k_rope);
        self.v_cache.push(v);
        let seq_len = self.k_cache.len();

        let heads_out: Vec<(Vec<f32>, Vec<f32>)> = (0..n_head)
            .into_par_iter()
            .map(|h| {
                let kv_h = h / n_groups;
                let kv_h_off = kv_h * head_dim;
                let q_slice = &q_rope[h * head_dim..(h + 1) * head_dim];
                let mut scores = vec![0.0f32; seq_len];
                let mut max_s = -f32::INFINITY;
                for t in 0..seq_len {
                    let k_head = &self.k_cache[t][kv_h_off..kv_h_off + head_dim];
                    let s = unsafe { dot_product(q_slice, k_head) } * scale;
                    scores[t] = s;
                    if s > max_s {
                        max_s = s;
                    }
                }
                let mut sum_e = 0.0f32;
                for t in 0..seq_len {
                    scores[t] = (scores[t] - max_s).exp();
                    sum_e += scores[t];
                }
                let inv_s = 1.0 / (sum_e + 1e-12);
                let weights: Vec<f32> = scores.iter().map(|s| s * inv_s).collect();
                let mut head_out = vec![0.0f32; head_dim];
                for t in 0..seq_len {
                    let w = weights[t];
                    let v_head = &self.v_cache[t][kv_h_off..kv_h_off + head_dim];
                    for i in 0..head_dim {
                        head_out[i] += w * v_head[i];
                    }
                }
                (head_out, weights)
            })
            .collect();

        let mut attn_out = vec![0.0f32; n_head * head_dim];
        let mut all_weights = vec![0.0f32; n_head * seq_len];
        for h in 0..n_head {
            let start = h * head_dim;
            attn_out[start..start + head_dim].copy_from_slice(&heads_out[h].0);
            all_weights[h * seq_len..(h + 1) * seq_len].copy_from_slice(&heads_out[h].1);
        }
        Ok((attn_out, all_weights, q_rope))
    }

    /// Backward de atención para el paso actual (sin re-forward).
    /// Devuelve (d_q_unrotado, d_k_actual, d_v_actual). Los gradientes de
    /// atención fluyen SOLO al token actual (pos = cache_len - 1); el cache
    /// de tokens pasados no recibe gradientes aquí (se entrena por token).
    pub fn backward_attention_core(
        &mut self,
        d_attn_out: &[f32],
        q_rope: &[f32],
        softmax_weights: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
        let n_head = self.n_head;
        let n_head_kv = self.n_head_kv;
        let head_dim = self.head_dim;
        let n_groups = n_head / n_head_kv;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let seq_len = self.k_cache.len();
        let pos = seq_len.saturating_sub(1);
        let t_last = seq_len - 1;

        let mut d_q = vec![0.0f32; n_head * head_dim];
        let mut d_k = vec![0.0f32; n_head_kv * head_dim];
        let mut d_v = vec![0.0f32; n_head_kv * head_dim];

        for h in 0..n_head {
            let kv_h = h / n_groups;
            let kv_h_off = kv_h * head_dim;
            let q_off = h * head_dim;
            let mut d_w = vec![0.0f32; seq_len];
            for t in 0..seq_len {
                let v_head = &self.v_cache[t][kv_h_off..kv_h_off + head_dim];
                d_w[t] = unsafe { dot_product(&d_attn_out[q_off..q_off + head_dim], v_head) };
            }
            // softmax backward: d_s_t = w_t*(d_w_t - Σ w*d_w)
            let mut dot_wd = 0.0f32;
            for t in 0..seq_len {
                dot_wd += softmax_weights[h * seq_len + t] * d_w[t];
            }
            for t in 0..seq_len {
                let w = softmax_weights[h * seq_len + t];
                let d_s = w * (d_w[t] - dot_wd) * scale;
                let k_head = &self.k_cache[t][kv_h_off..kv_h_off + head_dim];
                for i in 0..head_dim {
                    d_q[q_off + i] += d_s * k_head[i];
                }
                if t == t_last {
                    for i in 0..head_dim {
                        d_k[kv_h_off + i] += d_s * q_rope[q_off + i];
                    }
                }
            }
            // d_v
            for i in 0..head_dim {
                d_v[kv_h_off + i] +=
                    softmax_weights[h * seq_len + t_last] * d_attn_out[q_off + i];
            }
        }

        // Invertir RoPE sobre d_q (la rotación es ortogonal: rotar por -theta)
        let inv = |vec: &mut [f32], heads: usize| {
            for h in 0..heads {
                let h_start = h * head_dim;
                for i in 0..(head_dim / 2) {
                    let freq = 1.0 / (self.rope_base.powf((2.0 * i as f32) / head_dim as f32));
                    let t = (pos as f32) * freq;
                    let (sin, cos) = t.sin_cos();
                    if self.rope_style == "split" {
                        let a = vec[h_start + i];
                        let b = vec[h_start + i + head_dim / 2];
                        vec[h_start + i] = a * cos + b * sin;
                        vec[h_start + i + head_dim / 2] = -a * sin + b * cos;
                    } else {
                        let a = vec[h_start + 2 * i];
                        let b = vec[h_start + 2 * i + 1];
                        vec[h_start + 2 * i] = a * cos + b * sin;
                        vec[h_start + 2 * i + 1] = -a * sin + b * cos;
                    }
                }
            }
        };
        inv(&mut d_q, n_head);
        Ok((d_q, d_k, d_v))
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
