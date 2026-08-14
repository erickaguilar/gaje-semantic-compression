// =============================================================================
// council — CouncilOfTeachers: orquestación y consenso entre maestros
// =============================================================================
#[cfg(feature = "python")]
use pyo3::prelude::*;

use rayon::prelude::*;

use crate::nn::distiller::teacher::Teacher;

/// Orquestador de múltiples maestros para destilación por consenso.
#[cfg_attr(feature = "python", pyclass)]
#[derive(Clone)]
pub struct CouncilOfTeachers {
    pub teachers: Vec<Teacher>,
}

#[cfg_attr(feature = "python", pymethods)]
impl CouncilOfTeachers {
    #[cfg(feature = "python")]
    #[new]
    pub fn py_new() -> Self {
        Self::new()
    }

    #[cfg(feature = "python")]
    pub fn py_add_teacher(&mut self, teacher: Teacher) {
        self.add_teacher(teacher);
    }
}

impl Default for CouncilOfTeachers {
    fn default() -> Self {
        Self::new()
    }
}

impl CouncilOfTeachers {
    pub fn new() -> Self {
        CouncilOfTeachers {
            teachers: Vec::new(),
        }
    }

    pub fn add_teacher(&mut self, teacher: Teacher) {
        self.teachers.push(teacher);
    }

    /// Obtiene las probabilidades del consejo para una secuencia de texto.
    pub fn get_consensus_probs(&self, text: &str, student_vocab_size: usize) -> Vec<Vec<f32>> {
        if self.teachers.is_empty() {
            return Vec::new();
        }

        let teacher_results: Vec<Vec<Vec<f32>>> = self
            .teachers
            .par_iter()
            .map(|teacher| {
                let mut model = teacher.model.clone();
                let mut tokens = match teacher.tokenizer.encode(text, false) {
                    Ok(t) => t,
                    Err(_) => return Vec::new(),
                };

                // Limitamos a 512 tokens para evitar bloqueos en Android con textos muy largos (Mosaic)
                if tokens.len() > 512 {
                    tokens.truncate(512);
                }

                let mut seq_probs = Vec::with_capacity(tokens.len());
                model.clear_cache_core();

                for &token_id in tokens.iter() {
                    // Feedback silencioso pero útil para depuración interna si fuera necesario
                    match model.forward_core(token_id as usize, false) {
                        Ok(logits) => {
                            let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                            let mut sum_exp = 0.0f32;
                            let mut probs = vec![0.0f32; logits.len()];
                            for (i, &l) in logits.iter().enumerate() {
                                let e = (l - max_l).exp();
                                probs[i] = e;
                                sum_exp += e;
                            }
                            for p in &mut probs {
                                *p /= sum_exp + 1e-12;
                            }

                            if teacher.is_identity_vocab {
                                seq_probs.push(probs);
                            } else {
                                let mut student_probs = vec![0.0f32; student_vocab_size];
                                for (t_id, &p) in probs.iter().enumerate() {
                                    if p > 1e-6 {
                                        if let Some(s_id) =
                                            teacher.vocab_mapping.get(t_id).and_then(|&id| id)
                                        {
                                            student_probs[s_id] += p;
                                        }
                                    }
                                }

                                let s: f32 = student_probs.iter().sum();
                                if s > 0.0 {
                                    for p in &mut student_probs {
                                        *p /= s;
                                    }
                                }
                                seq_probs.push(student_probs);
                            }
                        }
                        Err(_) => {
                            seq_probs
                                .push(vec![1.0 / student_vocab_size as f32; student_vocab_size]);
                        }
                    }
                }
                seq_probs
            })
            .collect();

        let num_teachers = teacher_results.len();
        if num_teachers == 0 {
            return Vec::new();
        }

        let seq_len = teacher_results[0].len();
        let mut consensus = Vec::with_capacity(seq_len);

        for i in 0..seq_len {
            let mut combined = vec![0.0f32; student_vocab_size];
            for t_idx in 0..num_teachers {
                if i < teacher_results[t_idx].len() {
                    let probs = &teacher_results[t_idx][i];
                    for v in 0..student_vocab_size {
                        combined[v] += probs[v];
                    }
                }
            }
            for v in 0..student_vocab_size {
                combined[v] /= num_teachers as f32;
            }
            consensus.push(combined);
        }

        consensus
    }
}
