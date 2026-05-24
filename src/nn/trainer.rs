use crate::nn::llm::RustGenomicLLM;
use pyo3::prelude::*;
use std::time::Instant;

#[pyclass]
pub struct NativeGenomicTrainer {
    pub lr: f32,
    pub resonance_weight: f32,
}

#[pymethods]
impl NativeGenomicTrainer {
    #[new]
    #[pyo3(signature = (lr=0.01, resonance_weight=0.05))]
    pub fn new(lr: f32, resonance_weight: f32) -> Self {
        NativeGenomicTrainer {
            lr,
            resonance_weight,
        }
    }

    /// Ejecuta un paso de entrenamiento con Resonancia Semántica (Shannon Entropy)
    pub fn train_step(
        &self,
        model: &mut RustGenomicLLM,
        input_ids: Vec<usize>,
        target_ids: Vec<usize>,
        phase: u8,
    ) -> PyResult<f32> {
        if input_ids.len() != target_ids.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("Input and target IDs must have the same length"));
        }

        let seq_len = input_ids.len();
        model.clear_cache()?;
        let mut total_loss = 0.0;

        // Fase 2+: Refinamiento profundo usando el optimizador de secuencias nativo
        if phase >= 2 {
            let mut sequence = input_ids.clone();
            sequence.push(target_ids[seq_len - 1]);
            model.train_on_sequence(sequence, self.lr * 0.5)?;
        }

        for i in 0..seq_len {
            let token_id = input_ids[i];
            let target_id = target_ids[i];

            // 1. Forward pass capturando estados ocultos
            let (logits, h_norm) = model.forward_with_hidden(token_id, false)?;

            // 2. Calcular Probabilidades (Softmax)
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

            // 3. Loss: Cross Entropy + Resonance (Entropy Penalty)
            let ce_loss = -(probs[target_id] + 1e-12).ln();
            
            let mut entropy = 0.0f32;
            for &p in &probs {
                if p > 1e-12 {
                    entropy -= p * p.ln();
                }
            }
            // Normalizar entropía por log(vocab_size) para escala independiente
            let norm_entropy = entropy / (logits.len() as f32).ln();
            
            let loss = ce_loss + self.resonance_weight * norm_entropy;
            total_loss += loss;

            // 4. Gradiente: dL/dlogits
            let mut d_logits = probs;
            d_logits[target_id] -= 1.0;

            // Gradiente de la entropía: p * (-ln p - H)
            for j in 0..d_logits.len() {
                let p = d_logits[j] + (if j == target_id { 1.0 } else { 0.0 }); // p original
                let grad_ent = if p > 1e-12 {
                    p * (-p.ln() - entropy)
                } else {
                    0.0
                };
                d_logits[j] += self.resonance_weight * grad_ent;
            }

            // 5. Refinamiento del LM Head
            model.refine_lm_head(h_norm, d_logits, self.lr)?;

            // Fase 3: Mutaciones homeostáticas
            if phase >= 3 && i % 8 == 0 {
                model.mutate_all_homeostasis(self.lr * 0.01)?;
            }
        }

        Ok(total_loss / seq_len as f32)
    }

    /// Orquesta el entrenamiento sobre un dataset de secuencias
    pub fn fit(
        &self,
        model: &mut RustGenomicLLM,
        dataset: Vec<Vec<usize>>,
        epochs: usize,
    ) -> PyResult<()> {
        println!("[*] Rust-GenomicTrainer: Iniciando entrenamiento nativo ({} épocas)", epochs);
        
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

            let phase_name = match phase {
                1 => "Base (LM Head)",
                2 => "IQAT (Deep)",
                _ => "Evol (Homeostatic)",
            };

            let start = Instant::now();
            let mut epoch_loss = 0.0;
            let mut count = 0;

            for seq in &dataset {
                if seq.len() < 2 { continue; }
                let input = seq[0..seq.len()-1].to_vec();
                let target = seq[1..seq.len()].to_vec();
                
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
                let ppl = avg_loss.exp();
                println!(
                    "    - Época {}/{} [{}] | Loss: {:.4} | PPL: {:.2} | {:?}",
                    epoch + 1, epochs, phase_name, avg_loss, ppl, start.elapsed()
                );
            }
        }
        Ok(())
    }
}
