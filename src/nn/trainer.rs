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
        Self { lr, resonance_weight }
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

            let ce_loss = -(probs[target_id] + 1e-12).ln();
            
            let mut entropy = 0.0f32;
            for &p in &probs {
                if p > 1e-12 {
                    entropy -= p * p.ln();
                }
            }
            let norm_entropy = entropy / (logits.len() as f32).ln();
            
            let loss = ce_loss + self.resonance_weight * norm_entropy;
            total_loss += loss;

            let mut d_logits = probs;
            d_logits[target_id] -= 1.0;

            for j in 0..d_logits.len() {
                let p = d_logits[j] + (if j == target_id { 1.0 } else { 0.0 });
                let grad_ent = if p > 1e-12 {
                    p * (-p.ln() - entropy)
                } else {
                    0.0
                };
                d_logits[j] += self.resonance_weight * grad_ent;
            }

            model.lm_head.refine_with_grads_core(h_norm, d_logits, self.lr)?;
        }

        Ok(total_loss / seq_len as f32)
    }

    pub fn fit(
        &self,
        model: &mut GenomicLLM,
        dataset: &[Vec<usize>],
        epochs: usize,
    ) -> Result<(), String> {
        println!("[*] GenomicTrainer: Iniciando entrenamiento nativo ({} épocas)", epochs);
        
        let p1_end = (epochs as f32 * 0.2) as usize;
        let p2_end = (epochs as f32 * 0.7) as usize;

        for epoch in 0..epochs {
            let phase = if epoch < p1_end { 1 } else if epoch < p2_end { 2 } else { 3 };
            let phase_name = match phase {
                1 => "Base (LM Head)",
                2 => "IQAT (Deep)",
                _ => "Evol (Homeostatic)",
            };

            let start = Instant::now();
            let mut epoch_loss = 0.0;
            let mut count = 0;

            for seq in dataset {
                if seq.len() < 2 { continue; }
                let input = &seq[0..seq.len()-1];
                let target = &seq[1..seq.len()];
                
                match self.train_step(model, input, target, phase) {
                    Ok(loss) => {
                        epoch_loss += loss;
                        count += 1;
                    },
                    Err(e) => println!("    [!] Error en secuencia: {}", e),
                }
            }

            if count > 0 {
                let avg_loss = epoch_loss / count as f32;
                println!(
                    "    - Época {}/{} [{}] | Loss: {:.4} | PPL: {:.2} | {:?}",
                    epoch + 1, epochs, phase_name, avg_loss, avg_loss.exp(), start.elapsed()
                );
            }
        }
        Ok(())
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
        self.inner.fit(model, &dataset, epochs).map_err(pyo3::exceptions::PyValueError::new_err)
    }
}
