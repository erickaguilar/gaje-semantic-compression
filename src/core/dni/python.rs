// =============================================================================
// python — Bindings #[pymethods] y métodos de ingestión de DNIEngine
// =============================================================================
use rayon::prelude::*;

#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(not(feature = "python"))]
use crate::pyo3_shim::exceptions::PyValueError;
#[cfg(not(feature = "python"))]
use crate::pyo3_shim::*;

use crate::core::dni::{DNIEngine, SemanticNiche};
use crate::core::tokenizer::GajeTokenizer;
use crate::nn::distiller::CouncilOfTeachers;
use crate::nn::llm::GenomicLLM;

#[cfg_attr(feature = "python", pymethods)]
impl DNIEngine {
    #[cfg(feature = "python")]
    #[new]
    #[pyo3(signature = (model, tokenizer, council=None, intensity=0.01, target_layers=Vec::new(), niche=SemanticNiche::General))]
    pub fn py_new(
        model: GenomicLLM,
        tokenizer: GajeTokenizer,
        council: Option<CouncilOfTeachers>,
        intensity: f32,
        target_layers: Vec<String>,
        niche: SemanticNiche,
    ) -> Self {
        let mut engine = Self {
            model,
            tokenizer: std::sync::Arc::new(tokenizer),
            council: council.map(std::sync::Arc::new),
            intensity,
            target_layers,
            validation_tokens: Vec::new(),
            original_dna_hash: Vec::new(),
            niche,
        };
        engine.initialize_original_hash();
        engine
    }

    pub fn set_validation_text(&mut self, text: String) {
        if let Ok(tokens) = self.tokenizer.encode(&text, false) {
            self.validation_tokens = tokens;
        }
    }

    pub fn ingest_text(
        &mut self,
        text: String,
        generations: usize,
        pop_size: usize,
    ) -> PyResult<f32> {
        let tokens = self
            .tokenizer
            .encode(&text, false)
            .map_err(|e| PyValueError::new_err(format!("Tokenizer error: {}", e)))?;
        if tokens.len() < 2 {
            return Ok(0.0);
        }
        let activations = self.profile_activations(&tokens);

        // Temperatura Genómica Inicial (T_g)
        let mut t_g = 1.0f32;
        let base_sigma = 2.0f32;

        let mut population: Vec<GenomicLLM> = (0..pop_size)
            .map(|_| {
                let mut mutant = self.model.clone();
                // En el primer paso usamos la temperatura máxima y sigma máximo
                self.apply_targeted_mutation_v2(
                    &mut mutant,
                    self.intensity * t_g,
                    &activations,
                    base_sigma * t_g,
                );
                mutant
            })
            .collect();

        let mut best_fitness = 0.0;
        for gen in 0..generations {
            // Curva de Enfriamiento Termodinámico (Tercera Ley)
            t_g = 1.0 - (gen as f32 / generations as f32);
            let current_intensity = self.intensity * t_g;
            let current_sigma = base_sigma * t_g;

            let scores: Vec<(usize, f32)> = population
                .par_iter_mut()
                .enumerate()
                .map(|(idx, mutant)| {
                    let new_knowledge_fitness = self.evaluate_mutant(mutant, &tokens);
                    let mut final_fitness = new_knowledge_fitness;
                    if !self.validation_tokens.is_empty() {
                        let base_preservation =
                            self.evaluate_mutant(mutant, &self.validation_tokens);
                        final_fitness = (new_knowledge_fitness * 0.8) + (base_preservation * 0.2);
                    }
                    (idx, final_fitness)
                })
                .collect();
            let (best_idx, fitness) = scores
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();
            best_fitness = *fitness;

            if gen < generations - 1 {
                let winner = population[*best_idx].clone();
                population
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(i, mutant)| {
                        if i != *best_idx {
                            *mutant = winner.clone();
                            // Aplicamos mutación difusa con enfriamiento
                            self.apply_targeted_mutation_v2(
                                mutant,
                                current_intensity,
                                &activations,
                                current_sigma,
                            );
                        }
                    });
            } else {
                self.model = population[*best_idx].clone();
            }
        }
        Ok(best_fitness)
    }

    pub fn ingest_document(
        &mut self,
        document: String,
        generations: usize,
        pop_size: usize,
    ) -> PyResult<f32> {
        let chunks = self.chunk_text(&document);
        if chunks.is_empty() {
            return Ok(0.0);
        }
        let num_cpus = rayon::current_num_threads();
        let chunks_per_island = (chunks.len() as f32 / num_cpus as f32).ceil() as usize;
        let island_results: Vec<(GenomicLLM, f32)> = chunks
            .chunks(chunks_per_island)
            .enumerate()
            .par_bridge()
            .map(|(_island_id, island_chunks)| {
                let mut island_engine = DNIEngine {
                    model: self.model.clone(),
                    tokenizer: self.tokenizer.clone(),
                    council: self.council.clone(),
                    intensity: self.intensity,
                    target_layers: self.target_layers.clone(),
                    validation_tokens: self.validation_tokens.clone(),
                    original_dna_hash: Vec::new(),
                    niche: self.niche,
                };
                let mut total_fitness = 0.0;
                for chunk in island_chunks {
                    if let Ok(f) =
                        island_engine.ingest_text(chunk.to_string(), generations, pop_size)
                    {
                        total_fitness += f;
                    }
                }
                let avg_fitness = if island_chunks.is_empty() {
                    0.0
                } else {
                    total_fitness / island_chunks.len() as f32
                };
                (island_engine.model, avg_fitness)
            })
            .collect();
        if !island_results.is_empty() {
            let mut best_overall_fitness: f32 = 0.0;
            let mut final_model = self.model.clone();
            for (mutant_model, fitness) in island_results {
                self.merge_models(&mut final_model, &mutant_model);
                best_overall_fitness = best_overall_fitness.max(fitness);
            }
            self.model = final_model;
            Ok(best_overall_fitness)
        } else {
            Ok(0.0)
        }
    }

    pub fn ingest_specialized(
        &mut self,
        logic_doc: String,
        grammar_doc: String,
        generations: usize,
        pop_size: usize,
    ) -> PyResult<f32> {
        let mut logic_island = DNIEngine {
            model: self.model.clone(),
            tokenizer: self.tokenizer.clone(),
            council: self.council.clone(),
            intensity: self.intensity,
            target_layers: self.target_layers.clone(),
            validation_tokens: self.validation_tokens.clone(),
            original_dna_hash: Vec::new(),
            niche: SemanticNiche::Logic,
        };
        let mut grammar_island = DNIEngine {
            model: self.model.clone(),
            tokenizer: self.tokenizer.clone(),
            council: self.council.clone(),
            intensity: self.intensity,
            target_layers: self.target_layers.clone(),
            validation_tokens: self.validation_tokens.clone(),
            original_dna_hash: Vec::new(),
            niche: SemanticNiche::Grammar,
        };
        let (res_l, res_g) = rayon::join(
            || logic_island.ingest_document(logic_doc, generations, pop_size),
            || grammar_island.ingest_document(grammar_doc, generations, pop_size),
        );
        let logic_fitness = res_l?;
        let grammar_fitness = res_g?;
        self.migrate_knowledge(&mut logic_island.model, &mut grammar_island.model);
        self.model = logic_island.model;
        Ok((logic_fitness + grammar_fitness) / 2.0)
    }
}
