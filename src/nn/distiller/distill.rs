// =============================================================================
// distill — GenomicDistiller/NativeGenomicDistiller: ciclo de destilación
// =============================================================================
#[cfg(feature = "python")]
use pyo3::prelude::*;

use std::time::Instant;

use crate::core::tokenizer::GajeTokenizer;
use crate::nn::distiller::council::CouncilOfTeachers;
use crate::nn::llm::GenomicLLM;

/// Implementa el ciclo de destilación nativa.
#[cfg_attr(feature = "python", pyclass)]
#[derive(Clone)]
pub struct NativeGenomicDistiller {
    pub inner: GenomicDistiller,
}

#[cfg_attr(feature = "python", pymethods)]
impl NativeGenomicDistiller {
    #[cfg(feature = "python")]
    #[new]
    #[pyo3(signature = (council, student_tokenizer, distill_weight=0.5))]
    pub fn py_new(
        council: CouncilOfTeachers,
        student_tokenizer: GajeTokenizer,
        distill_weight: f32,
    ) -> Self {
        let mut inner = GenomicDistiller::new(council, student_tokenizer);
        inner.distill_weight = distill_weight;
        NativeGenomicDistiller { inner }
    }

    #[cfg(feature = "python")]
    pub fn fit(
        &self,
        student: &mut GenomicLLM,
        texts: Vec<String>,
        epochs: usize,
        lr: f32,
    ) -> PyResult<()> {
        self.inner
            .fit(student, &texts, epochs, lr)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }
}

#[derive(Clone)]
pub struct GenomicDistiller {
    pub council: CouncilOfTeachers,
    pub student_tokenizer: GajeTokenizer,
    pub distill_weight: f32,
}

impl GenomicDistiller {
    pub fn new(council: CouncilOfTeachers, student_tokenizer: GajeTokenizer) -> Self {
        GenomicDistiller {
            council,
            student_tokenizer,
            distill_weight: 0.5,
        }
    }

    pub fn distill_step(
        &self,
        student: &mut GenomicLLM,
        text: &str,
        lr: f32,
    ) -> Result<f32, String> {
        let mut tokens = self
            .student_tokenizer
            .encode(text, false)
            .map_err(|e| format!("Error tokenizando estudiante: {}", e))?;

        if tokens.len() < 2 {
            return Ok(0.0);
        }

        // Truncar para consistencia con el profesor y velocidad en Android
        if tokens.len() > 512 {
            tokens.truncate(512);
        }

        let student_vocab_size = student.lm_head.out_features;
        let consensus_seq = self.council.get_consensus_probs(text, student_vocab_size);

        if consensus_seq.is_empty() {
            return Ok(0.0);
        }

        student.clear_cache_core();
        let mut total_loss = 0.0;
        let n_steps = (tokens.len() - 1).min(consensus_seq.len());

        // Feedback granular cada 50 tokens para que el usuario sepa que hay vida
        let show_progress = n_steps > 100;

        for i in 0..n_steps {
            if show_progress && i % 50 == 0 {
                print!(".");
                use std::io::Write;
                std::io::stdout().flush().ok();
            }
            let token_id = tokens[i] as usize;
            let target_id = tokens[i + 1] as usize;
            let teacher_probs = &consensus_seq[i];

            let (logits, h_norm) = student.forward_with_hidden_core(token_id, false)?;

            let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum_exp = 0.0f32;
            let mut student_probs = vec![0.0f32; logits.len()];
            for (j, &l) in logits.iter().enumerate() {
                let e = (l - max_l).exp();
                student_probs[j] = e;
                sum_exp += e;
            }
            for p in &mut student_probs {
                *p /= sum_exp + 1e-12;
            }

            let ce_loss = -(student_probs[target_id].max(1e-10)).ln();

            let mut kl_loss = 0.0f32;
            for j in 0..logits.len() {
                let p_t = teacher_probs[j];
                let p_s = student_probs[j];
                if p_t > 1e-10 && p_s > 1e-10 {
                    kl_loss += p_t * (p_t.ln() - p_s.ln());
                }
            }

            let loss = (1.0 - self.distill_weight) * ce_loss + self.distill_weight * kl_loss;
            total_loss += loss;

            let mut d_logits = vec![0.0f32; logits.len()];
            for j in 0..logits.len() {
                let grad_ce = student_probs[j] - (if j == target_id { 1.0 } else { 0.0 });
                let grad_kl = student_probs[j] - teacher_probs[j];
                d_logits[j] = (1.0 - self.distill_weight) * grad_ce + self.distill_weight * grad_kl;
            }

            student
                .lm_head
                .refine_with_grads_core(h_norm, d_logits, lr)?;
        }

        Ok(total_loss / n_steps as f32)
    }

    pub fn fit(
        &self,
        student: &mut GenomicLLM,
        texts: &[String],
        epochs: usize,
        lr: f32,
    ) -> Result<(), String> {
        println!(
            "[*] Iniciando Crianza por Destilación Nativa ({} épocas)",
            epochs
        );

        for epoch in 0..epochs {
            let start = Instant::now();
            let mut epoch_loss = 0.0;
            let mut count = 0;

            for text in texts {
                match self.distill_step(student, text, lr) {
                    Ok(loss) => {
                        epoch_loss += loss;
                        count += 1;
                        if count % 10 == 0 {
                            println!(
                                "    - Muestra {}/{} | Loss: {:.4} | {:?}",
                                count,
                                texts.len(),
                                loss,
                                start.elapsed()
                            );
                        }
                    }
                    Err(e) => println!("    [!] Error en muestra: {}", e),
                }
            }

            if count > 0 {
                let avg_loss = epoch_loss / count as f32;
                println!(
                    "✅ Época {}/{} completada | Loss Promedio: {:.4} | Tiempo: {:?}",
                    epoch + 1,
                    epochs,
                    avg_loss,
                    start.elapsed()
                );
            }
        }
        Ok(())
    }
}

// =============================================================================
// DNI en Línea GPU: activación del pipeline zero-copy
// =============================================================================
impl GenomicDistiller {
    /// Intenta un paso de destilación acelerado en GPU vía GpuOnlineDistiller.
    /// Si la GPU no está disponible, cae de forma transparente a `distill_step` CPU.
    pub fn distill_step_online_gpu(
        &self,
        student: &mut GenomicLLM,
        text: &str,
        lr: f32,
        temperature: f32,
    ) -> Result<f32, String> {
        // Intentar ruta GPU si está compilado con feature gpu y hay contexto
        #[cfg(feature = "gpu")]
        {
                let tokens = self.student_tokenizer.encode(text, false).map_err(|e| e.to_string())?;
                if tokens.len() >= 2 {
                    let vocab = student.lm_head.out_features;
                    let consensus = self.council.get_consensus_probs(text, vocab);
                    if !consensus.is_empty() {
                        let batch = tokens.len().min(16).min(consensus.len());
                        if let Some(distiller) = crate::compute::gpu::pipeline::GpuOnlineDistiller::try_new_global(
                            batch,
                            temperature,
                            self.distill_weight,
                        ) {
                            let mut teacher_batch = Vec::with_capacity(batch * vocab);
                            let mut student_batch = Vec::with_capacity(batch * vocab);
                            student.clear_cache_core();
                            for i in 0..batch {
                                let tid = tokens[i] as usize;
                                let (logits, _) = student.forward_with_hidden_core(tid, false)?;
                                let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                                let sum_exp: f32 = logits.iter().map(|l| (l - max_l).exp()).sum();
                                for &l in &logits {
                                    student_batch.push(((l - max_l).exp()) / (sum_exp + 1e-12));
                                }
                                for &p in &consensus[i] {
                                    teacher_batch.push(p);
                                }
                            }
                            if let Some(q2_db) = match &mut student.lm_head.weight_db {
                                crate::nn::linear::WeightDatabase::GenomicQ2_0(db) => Some(db),
                                _ => None,
                            } {
                                let rows = student.lm_head.out_features;
                                let cols = student.lm_head.in_features;
                                let db_mut: &mut Vec<crate::io::header::blocks::Q2_0Block> = std::sync::Arc::make_mut(q2_db);
                                if let Ok(loss) = distiller.distill_step_online(&teacher_batch, &student_batch, db_mut, lr, rows, cols) {
                                    return Ok(loss);
                                }
                            }
                        }
                    }
                }
        }
        // Fallback CPU
        self.distill_step(student, text, lr)
    }
}
