// =============================================================================
// forward — Forwards, entrenamiento y generación de GenomicLLM
// =============================================================================
use crate::nn::llm::GenomicLLM;

impl GenomicLLM {
    pub fn forward_core(&mut self, token_id: usize, clear_cache: bool) -> Result<Vec<f32>, String> {
        if clear_cache {
            self.clear_cache_core();
        }
        let pos = if self.blocks.is_empty() {
            0
        } else {
            self.blocks[0].attn.k_cache_len()
        };

        // Modulación granular para la capa de salida (usamos la última capa como referencia)
        let modulation = self
            .topology
            .as_ref()
            .map(|t| t.get_modulation_factors(self.blocks.len(), 2, 0.5));

        let t_blocks_start = std::time::Instant::now();
        let mut h = self.embeddings.get_row_core(token_id)?;
        for block in &mut self.blocks {
            h = block.forward_core(h, pos)?;
        }
        let h_norm = unsafe { crate::compute::kernels::rms_norm(&h, &self.output_norm, self.eps) };
        let blocks_ms = t_blocks_start.elapsed().as_secs_f32() * 1000.0;

        let entropy = crate::compute::math::calculate_activation_entropy(&h_norm);
        let rna_threshold = if self.blocks.is_empty() {
            0.5
        } else {
            self.blocks[0].rna_threshold
        };
        let activate_rna = crate::compute::math::should_activate_rna(entropy, rna_threshold);

        if self.lm_head.out_features == 0 {
            return Err("LM Head out_features is 0!".to_string());
        }

        let t_head_start = std::time::Instant::now();
        let mut logits = self
            .lm_head
            .forward_core(h_norm, modulation, activate_rna)?;
        let head_ms = t_head_start.elapsed().as_secs_f32() * 1000.0;

        // Visual debug timing if enabled
        if std::env::var("GAJE_PROFILE_VERBOSE").is_ok() {
            eprintln!(
                "⏱️ [Profiling Token] Transformer Blocks: {:.2} ms | LM Head: {:.2} ms",
                blocks_ms, head_ms
            );
        }

        // Filtrado K-WTA en Logits de Salida para mitigar ruido de 2-bits
        if self.k_wta_ratio > 0.0 && self.k_wta_ratio < 1.0 {
            let k = ((logits.len() as f32 * self.k_wta_ratio) as usize).max(1);
            crate::compute::kernels::lateral_inhibition_kwta(&mut logits, k);
        }

        Ok(logits)
    }

    pub fn load_topology_core(&mut self, path: &str) -> Result<(), String> {
        let topo = crate::io::loader::load_topology(path).map_err(|e| e.to_string())?;
        let shared_topo = std::sync::Arc::new(topo);
        self.topology = Some(shared_topo.clone());
        for block in &mut self.blocks {
            block.topology = Some(shared_topo.clone());
        }
        Ok(())
    }

    pub fn forward_with_hidden_core(
        &mut self,
        token_id: usize,
        clear_cache: bool,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        if clear_cache {
            self.clear_cache_core();
        }
        let pos = if self.blocks.is_empty() {
            0
        } else {
            self.blocks[0].attn.k_cache_len()
        };

        let modulation = self
            .topology
            .as_ref()
            .map(|t| t.get_modulation_factors(self.blocks.len(), 2, 0.5));

        let mut h = self.embeddings.get_row_core(token_id)?;
        for block in &mut self.blocks {
            h = block.forward_core(h, pos)?;
        }
        let h_norm = unsafe { crate::compute::kernels::rms_norm(&h, &self.output_norm, self.eps) };

        let entropy = crate::compute::math::calculate_activation_entropy(&h_norm);
        let rna_threshold = if self.blocks.is_empty() {
            0.5
        } else {
            self.blocks[0].rna_threshold
        };
        let activate_rna = crate::compute::math::should_activate_rna(entropy, rna_threshold);

        let logits = self
            .lm_head
            .forward_core(h_norm.clone(), modulation, activate_rna)?;
        Ok((logits, h_norm))
    }

    pub fn forward_phase_gaje_core(
        &mut self,
        token_id: usize,
        k_wta: usize,
    ) -> Result<Vec<f32>, String> {
        let (_, h_norm) = self.forward_with_hidden_core(token_id, false)?;

        let modulation = self
            .topology
            .as_ref()
            .map(|t| t.get_modulation_factors(self.blocks.len(), 2, 0.5));

        let entropy = crate::compute::math::calculate_activation_entropy(&h_norm);
        let rna_threshold = if self.blocks.is_empty() {
            0.5
        } else {
            self.blocks[0].rna_threshold
        };
        let activate_rna = crate::compute::math::should_activate_rna(entropy, rna_threshold);

        let excitation = self
            .lm_head
            .forward_core(h_norm, modulation, activate_rna)?;

        let n_tokens = excitation.len();
        let max_excitation = excitation.iter().fold(0.0f32, |a, &b| a.max(b));
        let threshold = (max_excitation * 0.7).max(0.1);
        let mut candidates = Vec::with_capacity(n_tokens);
        for i in 0..n_tokens {
            let energy = excitation[i];
            if energy >= threshold {
                let intensity = 1.0 + (energy - threshold) / threshold;
                let excess_ratio = (energy - threshold) / threshold;
                let phase = if excess_ratio >= 1.0 {
                    0
                } else {
                    15 - (excess_ratio * 15.0) as u8
                };
                candidates.push((i, intensity, phase));
            }
        }
        candidates.sort_by(|a, b| {
            let res = a.2.cmp(&b.2);
            if res == std::cmp::Ordering::Equal {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                res
            }
        });
        let mut resonance_logits = vec![-100.0f32; n_tokens];
        let num_winners = candidates.len().min(k_wta);
        for i in 0..num_winners {
            let (idx, intensity, phase) = candidates[i];
            let phase_score = (16 - phase) as f32;
            resonance_logits[idx] = intensity * phase_score;
        }
        Ok(resonance_logits)
    }

    pub fn clear_cache_core(&mut self) {
        for block in &mut self.blocks {
            block.clear_cache_core();
        }
    }

    /// Entrena el cuerpo (último bloque) con el gradiente CE real, además del lm_head.
    /// Propaga d_logits de vuelta: lm_head^T -> output_norm backward -> último bloque.
    pub fn train_sequence_body_core(&mut self, tokens: Vec<usize>, lr: f32) -> Result<f32, String> {
        if tokens.len() < 2 {
            return Ok(0.0);
        }
        let mut total_loss = 0.0;
        self.clear_cache_core();
        let n_blocks = self.blocks.len();

        for i in 0..tokens.len() - 1 {
            let (logits, h_norm) = self.forward_with_hidden_core(tokens[i], false)?;
            let target = tokens[i + 1];
            let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum_e = 0.0f32;
            let mut exps = vec![0.0f32; logits.len()];
            for j in 0..logits.len() {
                let e = (logits[j] - max_l).exp();
                exps[j] = e;
                sum_e += e;
            }
            let prob = (exps[target] / sum_e).max(1e-12);
            total_loss -= prob.ln();
            let mut d_logits = vec![0.0f32; logits.len()];
            for j in 0..logits.len() {
                d_logits[j] = exps[j] / sum_e;
            }
            d_logits[target] -= 1.0;
            self.lm_head.refine_with_grads_core(h_norm.clone(), d_logits.clone(), lr)?;

            // Backprop CE al cuerpo: d_hidden = W_lm_head^T * d_logits -> rms_norm_backward -> último bloque
            if n_blocks > 0 {
                let d_h = self.lm_head.backward_core(d_logits)?;
                let d_x_final =
                    crate::compute::kernels::rms_norm_backward(&h_norm, &d_h, &self.output_norm, self.eps);
                // Entrada real del último bloque (re-propaga bloques anteriores, cache ya poblada)
                let mut x_in = self.embeddings.get_row_core(tokens[i])?;
                let pos = self.blocks[0].attn.k_cache_len();
                for blk in &mut self.blocks[..n_blocks - 1] {
                    x_in = blk.forward_core(x_in, pos)?;
                }
                self.blocks[n_blocks - 1].refine_with_grads_core(x_in, d_x_final, pos, lr)?;
            }
        }
        Ok(total_loss / (tokens.len() - 1) as f32)
    }

    /// Vía B — reverse-mode completo: propaga el gradiente CE a través de TODOS
    /// los bloques (de atrás hacia adelante), actualizando el cuerpo entero.
    pub fn train_sequence_full_body_core(
        &mut self,
        tokens: Vec<usize>,
        lr: f32,
    ) -> Result<f32, String> {
        if tokens.len() < 2 {
            return Ok(0.0);
        }
        let mut total_loss = 0.0;
        self.clear_cache_core();
        let n = self.blocks.len();

        for i in 0..tokens.len() - 1 {
            let target = tokens[i + 1];
            // Limpiamos el cache por token para evitar corrupción entre pasos
            // reverse (el refine re-ejecuta forwards que alteran el cache KV).
            self.clear_cache_core();
            let pos = if self.blocks.is_empty() {
                0
            } else {
                self.blocks[0].attn.k_cache_len()
            };

            // Forward capturando la entrada de cada bloque.
            let mut block_inputs: Vec<Vec<f32>> = Vec::with_capacity(n);
            let mut h = self.embeddings.get_row_core(tokens[i])?;
            for blk in &mut self.blocks {
                block_inputs.push(h.clone());
                h = blk.forward_core(h, pos)?;
            }
            let h_norm =
                unsafe { crate::compute::kernels::rms_norm(&h, &self.output_norm, self.eps) };
            let modulation = self
                .topology
                .as_ref()
                .map(|t| t.get_modulation_factors(n.max(1), 2, 0.5));
            let logits = self.lm_head.forward_core(h_norm.clone(), modulation, false)?;

            // Loss CE + d_logits = probs - one_hot
            let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum_e = 0.0f32;
            let mut exps = vec![0.0f32; logits.len()];
            for j in 0..logits.len() {
                let e = (logits[j] - max_l).exp();
                exps[j] = e;
                sum_e += e;
            }
            let prob = (exps[target] / sum_e).max(1e-12);
            total_loss -= prob.ln();
            let mut d_logits = vec![0.0f32; logits.len()];
            for j in 0..logits.len() {
                d_logits[j] = exps[j] / sum_e;
            }
            d_logits[target] -= 1.0;
            self.lm_head.refine_with_grads_core(h_norm.clone(), d_logits.clone(), lr)?;

            // Backprop: lm_head -> output_norm -> bloques (orden inverso)
            let d_h = self.lm_head.backward_core(d_logits)?;
            let mut d_out =
                crate::compute::kernels::rms_norm_backward(&h_norm, &d_h, &self.output_norm, self.eps);
            for b in (0..n).rev() {
                d_out = self.blocks[b].refine_with_grads_core(
                    block_inputs[b].clone(),
                    d_out,
                    pos,
                    lr,
                )?;
            }
        }
        Ok(total_loss / (tokens.len() - 1) as f32)
    }

    /// Entrenamiento del cuerpo con **caché de activaciones** (sin doble-forward).
    /// Guarda las activaciones del forward original de cada bloque y hace el
    /// backward en orden inverso usando exactamente esas activaciones.
    ///
    /// `n_train_blocks`: cuántos bloques desde el final se entrenan (escalera).
    /// `gclip`: grad-clipping global (>0 activo).
    pub fn train_sequence_cached_core(
        &mut self,
        tokens: Vec<usize>,
        lr: f32,
        n_train_blocks: usize,
        gclip: f32,
    ) -> Result<f32, String> {
        if tokens.len() < 2 {
            return Ok(0.0);
        }
        let mut total_loss = 0.0;
        self.clear_cache_core();
        let n = self.blocks.len();
        let start = n.saturating_sub(n_train_blocks);

        for i in 0..tokens.len() - 1 {
            let target = tokens[i + 1];
            let pos = if n > 0 {
                self.blocks[0].attn.k_cache_len()
            } else {
                0
            };

            // Forward con caché por bloque.
            let mut caches: Vec<crate::nn::block::BlockCache> = Vec::with_capacity(n);
            let mut h = self.embeddings.get_row_core(tokens[i])?;
            for blk in &mut self.blocks {
                let (out, cache) = blk.forward_core_cached(h, pos)?;
                caches.push(cache);
                h = out;
            }
            let h_norm =
                unsafe { crate::compute::kernels::rms_norm(&h, &self.output_norm, self.eps) };
            let modulation = self
                .topology
                .as_ref()
                .map(|t| t.get_modulation_factors(n.max(1), 2, 0.5));
            let logits = self.lm_head.forward_core(h_norm.clone(), modulation, false)?;

            // Loss CE + d_logits.
            let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum_e = 0.0f32;
            let mut exps = vec![0.0f32; logits.len()];
            for j in 0..logits.len() {
                let e = (logits[j] - max_l).exp();
                exps[j] = e;
                sum_e += e;
            }
            let prob = (exps[target] / sum_e).max(1e-12);
            total_loss -= prob.ln();
            let mut d_logits = vec![0.0f32; logits.len()];
            for j in 0..logits.len() {
                d_logits[j] = exps[j] / sum_e;
            }
            d_logits[target] -= 1.0;
            self.lm_head
                .refine_with_grads_core(h_norm.clone(), d_logits.clone(), lr)?;

            // Backward por caché en orden inverso (entrenan los bloques `start..n`).
            let d_h = self.lm_head.backward_core(d_logits)?;
            let mut d_out =
                crate::compute::kernels::rms_norm_backward(&h_norm, &d_h, &self.output_norm, self.eps);
            for b in (start..n).rev() {
                d_out = self.blocks[b].backward_core_cached(&caches[b], d_out, lr, gclip)?;
            }
        }
        Ok(total_loss / (tokens.len() - 1) as f32)
    }

    /// Entrenamiento del cuerpo con **lr por capas (layer-wise decay)**.
    ///
    /// Igual que `train_sequence_cached_core` pero asigna a cada bloque un lr
    /// que decae desde el último (cercano a la loss) hacia los primeros:
    /// `lr_b = lr * decay^(depth_b)`, donde `depth_b = (n-1) - b`. Con
    /// `lr_decay = 1.0` equivale a lr uniforme. Permite escalar a MÁS bloques
    /// de forma estable: los bloques tempranos reciben un lr menor, los tardíos
    /// el lr completo.
    pub fn train_sequence_cached_layerwise_core(
        &mut self,
        tokens: Vec<usize>,
        lr: f32,
        n_train_blocks: usize,
        gclip: f32,
        lr_decay: f32,
        train_lm_head: bool,
        progress_every: Option<usize>,
    ) -> Result<f32, String> {
        if tokens.len() < 2 {
            return Ok(0.0);
        }
        let mut total_loss = 0.0;
        self.clear_cache_core();
        let n = self.blocks.len();
        let start = n.saturating_sub(n_train_blocks);
        let t0 = std::time::Instant::now();
        let n_tok = tokens.len() - 1;

        for i in 0..tokens.len() - 1 {
            if let Some(every) = progress_every {
                if i > 0 && i % every == 0 {
                    eprintln!(
                        "  progress {i}/{n_tok} tokens, {:.1}s ({:.0} tok/min)",
                        t0.elapsed().as_secs_f32(),
                        (i as f32) / (t0.elapsed().as_secs_f32() / 60.0).max(1e-6)
                    );
                }
            }
            let target = tokens[i + 1];
            let pos = if n > 0 {
                self.blocks[0].attn.k_cache_len()
            } else {
                0
            };

            let mut caches: Vec<crate::nn::block::BlockCache> = Vec::with_capacity(n);
            let mut h = self.embeddings.get_row_core(tokens[i])?;
            for blk in &mut self.blocks {
                let (out, cache) = blk.forward_core_cached(h, pos)?;
                caches.push(cache);
                h = out;
            }
            let h_norm =
                unsafe { crate::compute::kernels::rms_norm(&h, &self.output_norm, self.eps) };
            let modulation = self
                .topology
                .as_ref()
                .map(|t| t.get_modulation_factors(n.max(1), 2, 0.5));
            let logits = self.lm_head.forward_core(h_norm.clone(), modulation, false)?;

            let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum_e = 0.0f32;
            let mut exps = vec![0.0f32; logits.len()];
            for j in 0..logits.len() {
                let e = (logits[j] - max_l).exp();
                exps[j] = e;
                sum_e += e;
            }
            let prob = (exps[target] / sum_e).max(1e-12);
            total_loss -= prob.ln();
            let mut d_logits = vec![0.0f32; logits.len()];
            for j in 0..logits.len() {
                d_logits[j] = exps[j] / sum_e;
            }
            d_logits[target] -= 1.0;
            if train_lm_head {
                self.lm_head
                    .refine_with_grads_core(h_norm.clone(), d_logits.clone(), lr)?;
            }

            let d_h = self.lm_head.backward_core(d_logits)?;
            let mut d_out =
                crate::compute::kernels::rms_norm_backward(&h_norm, &d_h, &self.output_norm, self.eps);
            for b in (start..n).rev() {
                let depth = (n - 1 - b) as f32;
                let lr_b = lr * lr_decay.powf(depth);
                d_out = self.blocks[b].backward_core_cached(&caches[b], d_out, lr_b, gclip)?;
            }
        }
        Ok(total_loss / (tokens.len() - 1) as f32)
    }

    pub fn eval_ce_core(&mut self, tokens: &[usize]) -> Result<f32, String> {
        if tokens.len() < 2 {
            return Ok(0.0);
        }
        let mut total_loss = 0.0;
        self.clear_cache_core();
        let n = self.blocks.len();
        let n_tok = tokens.len() - 1;
        for i in 0..tokens.len() - 1 {
            let target = tokens[i + 1];
            let pos = if n > 0 {
                self.blocks[0].attn.k_cache_len()
            } else {
                0
            };
            let mut caches: Vec<crate::nn::block::BlockCache> = Vec::with_capacity(n);
            let mut h = self.embeddings.get_row_core(tokens[i])?;
            for blk in &mut self.blocks {
                let (out, cache) = blk.forward_core_cached(h, pos)?;
                caches.push(cache);
                h = out;
            }
            let h_norm =
                unsafe { crate::compute::kernels::rms_norm(&h, &self.output_norm, self.eps) };
            let modulation = self
                .topology
                .as_ref()
                .map(|t| t.get_modulation_factors(n.max(1), 2, 0.5));
            let logits = self.lm_head.forward_core(h_norm.clone(), modulation, false)?;
            let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum_e = 0.0f32;
            for j in 0..logits.len() {
                sum_e += (logits[j] - max_l).exp();
            }
            let prob = ((logits[target] - max_l).exp() / sum_e).max(1e-12);
            total_loss -= prob.ln();
        }
        Ok(total_loss / n_tok as f32)
    }

    pub fn train_on_sequence_core(&mut self, tokens: Vec<usize>, lr: f32) -> Result<f32, String> {
        if tokens.len() < 2 {
            return Ok(0.0);
        }
        let mut total_loss = 0.0;
        self.clear_cache_core();

        let _modulation = self
            .topology
            .as_ref()
            .map(|t| t.get_modulation_factors(self.blocks.len(), 2, 0.5));

        for i in 0..tokens.len() - 1 {
            let (logits, h_final) = self.forward_with_hidden_core(tokens[i], false)?;
            let target = tokens[i + 1];
            let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum_e = 0.0f32;
            let mut exps = vec![0.0f32; logits.len()];
            for j in 0..logits.len() {
                let e = (logits[j] - max_l).exp();
                exps[j] = e;
                sum_e += e;
            }
            let prob = (exps[target] / sum_e).max(1e-12);
            total_loss -= prob.ln();
            let mut d_logits = vec![0.0f32; logits.len()];
            for j in 0..logits.len() {
                d_logits[j] = exps[j] / sum_e;
            }
            d_logits[target] -= 1.0;
            self.lm_head.refine_with_grads_core(h_final, d_logits, lr)?;
        }
        Ok(total_loss / (tokens.len() - 1) as f32)
    }

    pub fn generate_native_core(
        &mut self,
        prompt_tokens: Vec<usize>,
        max_new_tokens: usize,
        temperature: f32,
        repetition_penalty: f32,
        eos_token_ids: Vec<usize>,
    ) -> Result<Vec<usize>, String> {
        if prompt_tokens.is_empty() {
            return Err("Prompt tokens cannot be empty".to_string());
        }

        self.clear_cache_core();

        let mut last_logits = Vec::new();
        for &tid in &prompt_tokens {
            last_logits = self.forward_core(tid, false)?;
        }

        let mut generated = Vec::new();

        for _ in 0..max_new_tokens {
            if last_logits.is_empty() {
                break;
            }

            let mut logits = last_logits.clone();

            if repetition_penalty > 1.0 {
                let mut seen_set = std::collections::HashSet::new();
                for &t in &generated {
                    seen_set.insert(t);
                }
                for &eos_id in &eos_token_ids {
                    seen_set.remove(&eos_id);
                }
                for &t in &seen_set {
                    if t < logits.len() {
                        if logits[t] < 0.0 {
                            logits[t] *= repetition_penalty;
                        } else {
                            logits[t] /= repetition_penalty;
                        }
                    }
                }
            }

            let next_tok = if temperature <= 1e-5 {
                let mut max_idx = 0;
                let mut max_val = f32::NEG_INFINITY;
                for (idx, &val) in logits.iter().enumerate() {
                    if val > max_val {
                        max_val = val;
                        max_idx = idx;
                    }
                }
                max_idx
            } else {
                let mut max_l = f32::NEG_INFINITY;
                for &val in &logits {
                    if val > max_l {
                        max_l = val;
                    }
                }
                let mut probs = vec![0.0f32; logits.len()];
                let mut sum_exp = 0.0f32;
                for (idx, &val) in logits.iter().enumerate() {
                    let p = ((val - max_l) / temperature).exp();
                    probs[idx] = p;
                    sum_exp += p;
                }

                use rand::Rng;
                let mut rng = rand::thread_rng();
                let r = rng.gen::<f32>() * sum_exp;
                let mut cumulative_sum = 0.0f32;
                let mut chosen_idx = 0;
                for (idx, &p) in probs.iter().enumerate() {
                    cumulative_sum += p;
                    if r <= cumulative_sum {
                        chosen_idx = idx;
                        break;
                    }
                }
                chosen_idx
            };

            generated.push(next_tok);

            if eos_token_ids.contains(&next_tok) {
                break;
            }

            // Detector de repeticiones para evitar bucles infinitos en cuantizaciones inestables
            let mut repeated = false;
            for w in 2..=48 {
                if generated.len() >= w * 3 {
                    let last_chunk = &generated[generated.len() - w..];
                    let prev1 = &generated[generated.len() - w * 2..generated.len() - w];
                    let prev2 = &generated[generated.len() - w * 3..generated.len() - w * 2];
                    if last_chunk == prev1 && last_chunk == prev2 {
                        repeated = true;
                        break;
                    }
                }
            }
            if repeated {
                break;
            }

            last_logits = self.forward_core(next_tok, false)?;
        }

        Ok(generated)
    }
}
