use crate::nn::llm::GenomicLLM;
use std::time::Instant;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Núcleo del entrenador genómico (Pure Rust)
pub struct GenomicTrainerCore {
    pub lr: f32,
    pub resonance_weight: f32,
}

impl GenomicTrainerCore {
    pub fn new(lr: f32, resonance_weight: f32) -> Self {
        Self {
            lr,
            resonance_weight,
        }
    }

    pub fn train_step(
        &self,
        model: &mut GenomicLLM,
        input_ids: &[usize],
        target_ids: &[usize],
        phase: u8,
    ) -> Result<f32, String> {
        if input_ids.len() != target_ids.len() {
            return Err("Input and target IDs must have the same length".to_string());
        }

        let seq_len = input_ids.len();
        model.clear_cache_core();
        let mut total_loss = 0.0;

        // Phase 2+: Refinamiento profundo
        if phase >= 2 {
            let mut sequence = input_ids.to_vec();
            sequence.push(target_ids[seq_len - 1]);
            model.train_on_sequence_core(sequence, self.lr * 0.5)?;
        }

        for i in 0..seq_len {
            let token_id = input_ids[i];
            let target_id = target_ids[i];

            let (logits, h_norm) = model.forward_with_hidden_core(token_id, false)?;

            let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum_exp = 0.0f32;
            let mut probs = vec![0.0f32; logits.len()];
            for j in 0..logits.len() {
                let e = (logits[j] - max_l).exp();
                probs[j] = e;
                sum_exp += e;
            }
            for p in &mut probs {
                *p /= sum_exp + 1e-12;
            }

            let ce_loss = -(probs[target_id].max(1e-10)).ln();

            let mut entropy = 0.0f32;
            for &p in &probs {
                if p > 1e-12 {
                    entropy -= p * p.ln();
                }
            }
            let norm_entropy = entropy / (logits.len() as f32).ln();

            let loss = ce_loss + self.resonance_weight * norm_entropy;
            if !loss.is_nan() && !loss.is_infinite() {
                total_loss += loss;
            }

            let mut d_logits = probs;
            d_logits[target_id] -= 1.0;

            for j in 0..d_logits.len() {
                let p = (d_logits[j] + (if j == target_id { 1.0 } else { 0.0 })).max(1e-12);
                let grad_ent = p * (-p.ln() - entropy);
                d_logits[j] = (d_logits[j] + self.resonance_weight * grad_ent).clamp(-1.0, 1.0);
            }

            model
                .lm_head
                .refine_with_grads_core(h_norm, d_logits, self.lr)?;
        }

        Ok(total_loss / seq_len as f32)
    }

    pub fn fit_epoch<F>(
        &self,
        model: &mut GenomicLLM,
        dataset: &[Vec<usize>],
        epoch: usize,
        total_epochs: usize,
        phase: u8,
        start_step: usize,
        mut on_step: F,
    ) -> Result<f32, String>
    where
        F: FnMut(&mut GenomicLLM, usize, f32) -> Result<(), String>,
    {
        let phase_name = match phase {
            1 => "Base (LM Head)",
            2 => "IQAT (Deep)",
            _ => "Evol (Homeostatic)",
        };

        let start = Instant::now();
        let mut epoch_loss = 0.0;
        let mut count = 0;

        // Si start_step >= dataset.len(), reiniciamos a 0 para esta época
        let actual_start = if start_step < dataset.len() {
            start_step
        } else {
            0
        };
        if actual_start > 0 {
            println!("    [>] Reanudando desde muestra #{}", actual_start);
        }

        for (idx, seq) in dataset.iter().enumerate().skip(actual_start) {
            if seq.len() < 2 {
                continue;
            }
            let input = &seq[0..seq.len() - 1];
            let target = &seq[1..seq.len()];

            match self.train_step(model, input, target, phase) {
                Ok(loss) => {
                    epoch_loss += loss;
                    count += 1;
                    // Callback cada 100 muestras (intra-epoch save)
                    if count % 100 == 0 {
                        // El índice absoluto es actual_start + count
                        on_step(model, actual_start + count, epoch_loss / count as f32)?;
                    }
                }
                Err(e) => println!("    [!] Error en secuencia {}: {}", idx, e),
            }
        }

        if count > 0 {
            let avg_loss = epoch_loss / count as f32;
            println!(
                "    - Época {}/{} [{}] | Loss: {:.4} | PPL: {:.2} | {:?}",
                epoch + 1,
                total_epochs,
                phase_name,
                avg_loss,
                avg_loss.exp(),
                start.elapsed()
            );
            Ok(avg_loss)
        } else {
            Err("No valid samples in dataset".to_string())
        }
    }

    pub fn fit(
        &self,
        model: &mut GenomicLLM,
        dataset: &[Vec<usize>],
        epochs: usize,
    ) -> Result<(), String> {
        println!(
            "[*] GenomicTrainer: Iniciando entrenamiento nativo ({} épocas)",
            epochs
        );

        let p1_end = (epochs as f32 * 0.2) as usize;
        let p2_end = (epochs as f32 * 0.7) as usize;

        for epoch in 0..epochs {
            let phase = if epoch < p1_end {
                1
            } else if epoch < p2_end {
                2
            } else {
                3
            };
            self.fit_epoch(model, dataset, epoch, epochs, phase, 0, |_, _, _| Ok(()))?;
        }
        Ok(())
    }

    /// Entrena SOLO el `lm_head` (que en el formato híbrido GAJE es FP32), usando
    /// fase 1. Evita tocar el cuerpo cuantizado Q4_0, que es frágil en training.
    /// Es la vía segura y rápida para pruebas de destilación/SFT.
    pub fn fit_lm_head(
        &self,
        model: &mut GenomicLLM,
        dataset: &[Vec<usize>],
    ) -> Result<f32, String> {
        let mut total = 0.0f32;
        let mut count = 0usize;
        for seq in dataset {
            if seq.len() < 2 {
                continue;
            }
            let input = &seq[0..seq.len() - 1];
            let target = &seq[1..seq.len()];
            match self.train_step(model, input, target, 1) {
                Ok(loss) => {
                    total += loss;
                    count += 1;
                }
                Err(_) => {}
            }
        }
        if count > 0 {
            Ok(total / count as f32)
        } else {
            Err("No valid samples in dataset".to_string())
        }
    }

    /// Evalúa la pérdida cross-entropy promedio de una secuencia usando únicamente forward passes.
    pub fn evaluate_sequence_loss(
        model: &mut GenomicLLM,
        sequence: &[usize],
    ) -> Result<f32, String> {
        if sequence.len() < 2 {
            return Ok(0.0);
        }
        model.clear_cache_core();
        let mut total_loss = 0.0f32;
        let n_tokens = sequence.len() - 1;

        for i in 0..n_tokens {
            let token_id = sequence[i];
            let target_id = sequence[i + 1];

            let logits = model.forward_core(token_id, false)?;
            let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum_exp = 0.0f32;
            for &l in &logits {
                sum_exp += (l - max_l).exp();
            }
            if sum_exp > 0.0 && target_id < logits.len() {
                let target_logit = logits[target_id];
                let log_prob = (target_logit - max_l) - sum_exp.ln();
                let loss = -log_prob;
                if !loss.is_nan() && !loss.is_infinite() {
                    total_loss += loss;
                }
            }
        }
        Ok(total_loss / n_tokens as f32)
    }

    fn mutate_sub_layer(
        model: &mut GenomicLLM,
        block_idx: usize,
        sub_layer_choice: usize,
        mutations: &[(usize, usize, u8)],
    ) {
        use crate::nn::linear::GenomicOperable;
        let target_layer = match sub_layer_choice {
            0 => &mut model.blocks[block_idx].gate_gen,
            1 => &mut model.blocks[block_idx].up_gen,
            2 => &mut model.blocks[block_idx].w_down,
            3 => &mut model.blocks[block_idx].q_gen,
            4 => &mut model.blocks[block_idx].k_gen,
            5 => &mut model.blocks[block_idx].v_gen,
            _ => &mut model.blocks[block_idx].w_o,
        };
        for &(b_idx, s_idx, new_bits) in mutations {
            target_layer.weight_db.mutate(b_idx, s_idx, new_bits);
        }
    }

    /// Ejecuta un paso de entrenamiento de orden cero (SPSA Discreto) sobre los centroides de pesos Q4_0
    /// evaluando únicamente 2 forward passes antitéticos (L+ y L-).
    pub fn train_step_zero_order_spsa(
        model: &mut GenomicLLM,
        sequence: &[usize],
        temp: i32,
        k_coords: usize,
    ) -> Result<f32, String> {
        use crate::nn::linear::GenomicOperable;
        use rand::Rng;

        if sequence.len() < 2 || model.blocks.is_empty() {
            return Ok(0.0);
        }

        let l0 = Self::evaluate_sequence_loss(model, sequence)?;
        let mut rng = rand::thread_rng();

        let block_idx = rng.gen_range(0..model.blocks.len());
        let sub_layer_choice = rng.gen_range(0..7);

        let mutated_coords: Vec<(usize, usize, u8, u8, u8)> = {
            let target_layer = match sub_layer_choice {
                0 => &model.blocks[block_idx].gate_gen,
                1 => &model.blocks[block_idx].up_gen,
                2 => &model.blocks[block_idx].w_down,
                3 => &model.blocks[block_idx].q_gen,
                4 => &model.blocks[block_idx].k_gen,
                5 => &model.blocks[block_idx].v_gen,
                _ => &model.blocks[block_idx].w_o,
            };

            let total_bytes = target_layer.weight_db.len_bytes();
            if total_bytes == 0 {
                return Ok(l0);
            }

            let k = k_coords.min(total_bytes).max(1);
            let mut coords = Vec::with_capacity(k);

            for _ in 0..k {
                let byte_idx = rng.gen_range(0..total_bytes);
                let sub_idx = rng.gen_range(0..2);
                let original_bits = target_layer.weight_db.read(byte_idx, sub_idx);
                let delta = if rng.gen_bool(0.5) { temp } else { -temp };

                let plus_bits = (original_bits as i32 + delta).clamp(0, 15) as u8;
                let minus_bits = (original_bits as i32 - delta).clamp(0, 15) as u8;

                coords.push((byte_idx, sub_idx, original_bits, plus_bits, minus_bits));
            }
            coords
        };

        let plus_vec: Vec<(usize, usize, u8)> = mutated_coords.iter().map(|&(b, s, _, p, _)| (b, s, p)).collect();
        let minus_vec: Vec<(usize, usize, u8)> = mutated_coords.iter().map(|&(b, s, _, _, m)| (b, s, m)).collect();
        let orig_vec: Vec<(usize, usize, u8)> = mutated_coords.iter().map(|&(b, s, o, _, _)| (b, s, o)).collect();

        // 1. Forward positivo (+delta)
        Self::mutate_sub_layer(model, block_idx, sub_layer_choice, &plus_vec);
        let l_plus = Self::evaluate_sequence_loss(model, sequence)?;

        // 2. Forward antitético (-delta)
        Self::mutate_sub_layer(model, block_idx, sub_layer_choice, &minus_vec);
        let l_minus = Self::evaluate_sequence_loss(model, sequence)?;

        // 3. Consolidación de par antitético
        let (best_loss, apply_state) = if l_plus < l_minus && l_plus < l0 {
            (l_plus, 1)
        } else if l_minus < l_plus && l_minus < l0 {
            (l_minus, -1)
        } else {
            (l0, 0)
        };

        match apply_state {
            1 => {
                Self::mutate_sub_layer(model, block_idx, sub_layer_choice, &plus_vec);
            }
            -1 => {
                // Ya se encuentra en minus_vec
            }
            _ => {
                Self::mutate_sub_layer(model, block_idx, sub_layer_choice, &orig_vec);
            }
        }

        Ok(best_loss)
    }

    /// Loop completo de entrenamiento nativo de orden cero con currículo SPSA
    pub fn fit_zero_order(
        &self,
        model: &mut GenomicLLM,
        dataset: &[Vec<usize>],
        epochs: usize,
        k_coords: usize,
    ) -> Result<f32, String> {
        println!(
            "[*] GenomicTrainer: Iniciando entrenamiento de orden cero (SPSA Discreto, {} épocas, k={})",
            epochs, k_coords
        );

        let mut final_loss = 0.0f32;

        for epoch in 0..epochs {
            let start = Instant::now();
            let mut epoch_loss = 0.0f32;
            let mut count = 0usize;

            // Schedule de temperatura T_g (de 3 a 1)
            let temp = if epoch < (epochs / 3) {
                3
            } else if epoch < (2 * epochs / 3) {
                2
            } else {
                1
            };

            for seq in dataset {
                if seq.len() < 2 {
                    continue;
                }
                match Self::train_step_zero_order_spsa(model, seq, temp, k_coords) {
                    Ok(loss) => {
                        epoch_loss += loss;
                        count += 1;
                    }
                    Err(e) => eprintln!("    [!] Error SPSA en muestra: {}", e),
                }
            }

            if count > 0 {
                let avg_loss = epoch_loss / count as f32;
                final_loss = avg_loss;
                println!(
                    "    - Época {}/{} [SPSA Zero-Order (T={})] | Loss: {:.4} | PPL: {:.2} | {:?}",
                    epoch + 1,
                    epochs,
                    temp,
                    avg_loss,
                    avg_loss.exp(),
                    start.elapsed()
                );
            }
        }

        Ok(final_loss)
    }
}

// --- Python Wrapper ---

#[cfg(feature = "python")]
#[pyclass]
pub struct NativeGenomicTrainer {
    pub inner: GenomicTrainerCore,
}

#[cfg(feature = "python")]
#[pymethods]
impl NativeGenomicTrainer {
    #[new]
    #[pyo3(signature = (lr=0.01, resonance_weight=0.05))]
    pub fn new(lr: f32, resonance_weight: f32) -> Self {
        NativeGenomicTrainer {
            inner: GenomicTrainerCore::new(lr, resonance_weight),
        }
    }

    pub fn fit(
        &self,
        model: &mut crate::nn::llm::GenomicLLM,
        dataset: Vec<Vec<usize>>,
        epochs: usize,
    ) -> PyResult<()> {
        self.inner
            .fit(model, &dataset, epochs)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Entrena solo el `lm_head` (FP32) sobre el dataset. Devuelve el loss medio.
    #[pyo3(signature = (model, dataset, lr=0.01))]
    pub fn fit_lm_head(
        &self,
        model: &mut crate::nn::llm::GenomicLLM,
        dataset: Vec<Vec<usize>>,
        lr: f32,
    ) -> PyResult<f32> {
        let core = GenomicTrainerCore::new(lr, self.inner.resonance_weight);
        core.fit_lm_head(model, &dataset)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Entrena el organismo completo con optimización de orden cero (SPSA Discreto) sin backpropagation.
    #[pyo3(signature = (model, dataset, epochs=5, k_coords=32))]
    pub fn fit_zero_order(
        &self,
        model: &mut crate::nn::llm::GenomicLLM,
        dataset: Vec<Vec<usize>>,
        epochs: usize,
        k_coords: usize,
    ) -> PyResult<f32> {
        self.inner
            .fit_zero_order(model, &dataset, epochs, k_coords)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }
}
