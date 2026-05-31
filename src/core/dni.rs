/// 🧬 DNIEngine: Motor de Ingestión Neuronal Directa para GAJE-Flow
/// Permite la inyección granular de conocimiento en los pesos de 2 bits
/// mediante evolución dirigida ultrarrápida.

use crate::nn::llm::GenomicLLM;
use crate::core::tokenizer::GajeTokenizer;
use crate::nn::distiller::CouncilOfTeachers;
use rand::Rng;
use rayon::prelude::*;
use std::sync::Arc;

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;

#[cfg(not(feature = "python"))]
use crate::pyo3_shim::*;
#[cfg(not(feature = "python"))]
use crate::pyo3_shim::exceptions::{PyIOError, PyValueError};

#[cfg_attr(feature = "python", pyclass)]
pub struct DNIEngine {
    pub model: GenomicLLM,
    pub tokenizer: Arc<GajeTokenizer>,
    pub council: Option<Arc<CouncilOfTeachers>>,
    /// Ratio de mutación para la ingesta
    pub intensity: f32,
    /// Capas objetivo para la mutación (si está vacío, se usan heurísticas)
    pub target_layers: Vec<String>,
}

#[cfg_attr(feature = "python", pymethods)]
impl DNIEngine {
    #[cfg(feature = "python")]
    #[new]
    #[pyo3(signature = (model, tokenizer, council=None, intensity=0.01, target_layers=Vec::new()))]
    pub fn py_new(
        model: GenomicLLM,
        tokenizer: GajeTokenizer,
        council: Option<CouncilOfTeachers>,
        intensity: f32,
        target_layers: Vec<String>,
    ) -> Self {
        Self {
            model,
            tokenizer: Arc::new(tokenizer),
            council: council.map(Arc::new),
            intensity,
            target_layers,
        }
    }

    /// Ejecuta un paso de ingestión sobre un fragmento de texto.
    pub fn ingest_text(&mut self, text: String, generations: usize, pop_size: usize) -> PyResult<f32> {
        let tokens = self.tokenizer.encode(&text, false)
            .map_err(|e| PyValueError::new_err(format!("Tokenizer error: {}", e)))?;
        
        if tokens.len() < 2 {
            return Ok(0.0);
        }

        // Crear población de mutantes (aislados)
        let mut population: Vec<GenomicLLM> = (0..pop_size)
            .map(|_| {
                let mut mutant = self.model.clone();
                self.apply_targeted_mutation(&mut mutant, self.intensity);
                mutant
            })
            .collect();

        let mut best_fitness = 0.0;

        for gen in 0..generations {
            // Evaluación en paralelo
            let scores: Vec<(usize, f32)> = population.par_iter_mut()
                .enumerate()
                .map(|(idx, mutant)| {
                    let fitness = self.evaluate_mutant(mutant, &tokens);
                    (idx, fitness)
                })
                .collect();

            // Encontrar el mejor de esta generación
            let (best_idx, fitness) = scores.iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();
            
            best_fitness = *fitness;

            if gen < generations - 1 {
                let winner = population[*best_idx].clone();
                // Evolucionar: El ganador sobrevive y el resto son mutaciones del ganador
                population.par_iter_mut().enumerate().for_each(|(i, mutant)| {
                    if i != *best_idx {
                        *mutant = winner.clone();
                        self.apply_targeted_mutation(mutant, self.intensity * (1.0 - (gen as f32 / generations as f32)));
                    }
                });
            } else {
                // Fin de la evolución: El ganador se convierte en el modelo oficial
                self.model = population[*best_idx].clone();
            }
        }

        Ok(best_fitness)
    }
}

impl DNIEngine {
    /// Evalúa la coherencia de un mutante respecto a los tokens objetivo.
    fn evaluate_mutant(&self, mutant: &mut GenomicLLM, tokens: &[u32]) -> f32 {
        mutant.clear_cache_core();
        let mut total_prob = 0.0;
        let mut count = 0;

        for i in 0..tokens.len() - 1 {
            if let Ok(logits) = mutant.forward_core(tokens[i] as usize, false) {
                let target = tokens[i+1] as usize;
                
                // Softmax rápido
                let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut sum_exp = 0.0f32;
                let target_exp = (logits[target] - max_l).exp();
                for &l in &logits {
                    sum_exp += (l - max_l).exp();
                }
                
                total_prob += target_exp / (sum_exp + 1e-12);
                count += 1;
            }
        }

        if count > 0 { total_prob / count as f32 } else { 0.0 }
    }

    /// Aplica mutación bitwise XOR a las capas seleccionadas (Heurística DNI).
    fn apply_targeted_mutation(&self, mutant: &mut GenomicLLM, rate: f32) {
        let mut rng = rand::thread_rng();
        let n_blocks = mutant.blocks.len();
        
        // Si hay capas objetivo definidas, las usamos. 
        // De lo contrario, usamos la heurística DNI (bloques intermedios).
        if !self.target_layers.is_empty() {
            for layer_pattern in &self.target_layers {
                // Soportamos patrones simples como "block.10" o "block.11.ffn"
                for i in 0..n_blocks {
                    let block_name = format!("block.{}", i);
                    if layer_pattern.contains(&block_name) {
                        let block = &mut mutant.blocks[i];
                        let mut target_weights = Vec::new();
                        
                        if layer_pattern.contains("ffn") || !layer_pattern.contains(".") {
                            target_weights.push(&mut block.gate_gen);
                            target_weights.push(&mut block.up_gen);
                            target_weights.push(&mut block.w_down);
                        }
                        
                        if layer_pattern.contains("attn") {
                            target_weights.push(&mut block.q_gen);
                            target_weights.push(&mut block.k_gen);
                            target_weights.push(&mut block.v_gen);
                            target_weights.push(&mut block.w_o);
                        }

                        for layer in target_weights {
                            let mut db = (*layer.database).clone();
                            let mut changed = false;
                            for byte in &mut db {
                                if rng.gen::<f32>() < rate {
                                    *byte ^= rng.gen::<u8>();
                                    changed = true;
                                }
                            }
                            if changed {
                                layer.database = Arc::new(db);
                            }
                        }
                    }
                }
            }
            return;
        }

        // Heurística DNI: Mutar principalmente bloques intermedios
        let start_block = (n_blocks / 10).max(2);
        let end_block = n_blocks - 1;

        for i in start_block..end_block {
            let block = &mut mutant.blocks[i];
            
            let layers = [
                &mut block.gate_gen, 
                &mut block.up_gen, 
                &mut block.w_down
            ];

            for layer in layers {
                let mut db = (*layer.database).clone();
                let mut changed = false;
                for byte in &mut db {
                    if rng.gen::<f32>() < rate {
                        *byte ^= rng.gen::<u8>();
                        changed = true;
                    }
                }
                if changed {
                    layer.database = Arc::new(db);
                }
            }
        }
    }
}
