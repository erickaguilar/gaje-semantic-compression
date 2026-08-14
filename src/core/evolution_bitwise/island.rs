// =============================================================================
// island — IslandModel: islas paralelas y migración
// =============================================================================
use rayon::prelude::*;
use std::sync::Arc;

use crate::core::evolution_bitwise::engine::SpikingEvolutionEngine;
use crate::core::tokenizer::GajeTokenizer;
use crate::core::topology::CentroidGraph;
use crate::nn::distiller::CouncilOfTeachers;
use crate::nn::llm::GenomicLLM;
use crate::nn::spiking::layer::GajeNeuromorphicLayer;

/// Motor de Evolución por Poblaciones Paralelas (Island Model).
pub struct IslandModel {
    pub islands: Vec<SpikingEvolutionEngine>,
    pub migration_rate: usize,
    pub topology_map: Option<Arc<CentroidGraph>>,
    pub generation: usize,
    pub council: Option<Arc<CouncilOfTeachers>>,
    pub tokenizer: Option<Arc<GajeTokenizer>>,
}

impl IslandModel {
    pub fn new(
        initial_model: Vec<GajeNeuromorphicLayer>,
        num_islands: usize,
        pop_per_island: usize,
        centroides_real: [f32; 4],
        centroides_imag: [f32; 4],
        mutation_rate: f32,
        migration_rate: usize,
        topology_map: Option<Arc<CentroidGraph>>,
    ) -> Self {
        let mut islands = Vec::with_capacity(num_islands);
        for _ in 0..num_islands {
            islands.push(SpikingEvolutionEngine::new(
                initial_model.clone(),
                pop_per_island,
                centroides_real,
                centroides_imag,
                mutation_rate,
            ));
        }
        Self {
            islands,
            migration_rate,
            topology_map,
            generation: 0,
            council: None,
            tokenizer: None,
        }
    }

    /// Crea un Island Model específico para LLMs.
    pub fn new_llm(
        initial_llm: GenomicLLM,
        num_islands: usize,
        pop_per_island: usize,
        mutation_rate: f32,
        migration_rate: usize,
        topology_map: Option<Arc<CentroidGraph>>,
    ) -> Self {
        let mut islands = Vec::with_capacity(num_islands);
        for _ in 0..num_islands {
            islands.push(SpikingEvolutionEngine::new_llm(
                initial_llm.clone(),
                pop_per_island,
                mutation_rate,
            ));
        }
        Self {
            islands,
            migration_rate,
            topology_map,
            generation: 0,
            council: None,
            tokenizer: None,
        }
    }

    pub fn set_council(&mut self, council: Arc<CouncilOfTeachers>, tokenizer: Arc<GajeTokenizer>) {
        self.council = Some(council);
        self.tokenizer = Some(tokenizer);
    }

    /// Evalúa la población usando el Fitness Híbrido (Coherence + Needle).
    pub fn evaluate_hybrid(&mut self, texts: &[String]) {
        if self.council.is_none() || self.tokenizer.is_none() {
            return;
        }
        let council = self.council.as_ref().unwrap().clone();
        let tokenizer = self.tokenizer.as_ref().unwrap().clone();

        for island in &mut self.islands {
            island.population.par_iter_mut().for_each(|organism| {
                if let Some(llm) = &mut organism.llm {
                    let mut total_score = 0.0;
                    let mut coherence_score = 0.0;
                    for text in texts {
                        let consensus = council.get_consensus_probs(text, llm.lm_head.out_features);
                        let tokens = tokenizer.encode(text, false).unwrap_or_default();

                        llm.clear_cache_core();
                        let mut match_p = 0.0;
                        let steps = (tokens.len() - 1).min(consensus.len());

                        for i in 0..steps {
                            if let Ok(logits) = llm.forward_core(tokens[i] as usize, false) {
                                let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                                let mut sum_exp = 0.0f32;
                                let mut probs = vec![0.0f32; logits.len()];
                                for (j, &l) in logits.iter().enumerate() {
                                    let e = (l - max_l).exp();
                                    probs[j] = e;
                                    sum_exp += e;
                                }
                                for p in &mut probs {
                                    *p /= sum_exp + 1e-12;
                                }
                                let teacher_p = &consensus[i];
                                for (j, &tp) in teacher_p.iter().enumerate() {
                                    match_p += (tp * probs[j]).sqrt();
                                }
                            }
                        }
                        if steps > 0 {
                            coherence_score += match_p / (steps as f32);
                        }
                    }
                    coherence_score /= texts.len() as f32;
                    total_score += coherence_score;
                    organism.fitness = total_score;
                }
            });

            island.population.sort_by(|a, b| {
                b.fitness
                    .partial_cmp(&a.fitness)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
    }

    pub fn step(&mut self) {
        self.islands.par_iter_mut().for_each(|island| {
            island.evolve();
        });
        self.generation += 1;
        if self.generation.is_multiple_of(self.migration_rate) {
            self.migrate();
        }
    }

    pub fn migrate(&mut self) {
        let num_islands = self.islands.len();
        if num_islands < 2 {
            return;
        }
        let mut migration_pool = Vec::with_capacity(num_islands);
        for island in &self.islands {
            if let Some(best) = island.population.first() {
                migration_pool.push(best.clone());
            }
        }
        for i in 0..num_islands {
            let target_island_idx = (i + 1) % num_islands;
            let immigrant = migration_pool[i].clone();
            if let Some(worst) = self.islands[target_island_idx].population.last_mut() {
                *worst = immigrant;
            }
        }
    }
}
