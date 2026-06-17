use crate::core::tokenizer::GajeTokenizer;
use crate::nn::distiller::CouncilOfTeachers;
use crate::nn::linear::GenomicOperable;
/// 🧬 DNIEngine: Motor de Ingestión Neuronal Directa para GAJE-Flow
/// Permite la inyección granular de conocimiento en los pesos de 2 bits
/// mediante evolución dirigida ultrarrápida.
use crate::nn::llm::GenomicLLM;
use rand::Rng;
use rayon::prelude::*;
use std::sync::Arc;

#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(not(feature = "python"))]
use crate::pyo3_shim::exceptions::PyValueError;
#[cfg(not(feature = "python"))]
use crate::pyo3_shim::*;

/// # 🏝️ Island Model: Especialización por Nichos
#[cfg_attr(feature = "python", pyclass(eq, eq_int))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticNiche {
    General,
    Logic,
    Grammar,
    Memory,
}

#[cfg_attr(feature = "python", pyclass)]
pub struct DNIEngine {
    pub model: GenomicLLM,
    pub tokenizer: Arc<GajeTokenizer>,
    pub council: Option<Arc<CouncilOfTeachers>>,
    pub intensity: f32,
    pub target_layers: Vec<String>,
    pub validation_tokens: Vec<u32>,
    pub original_dna_hash: Vec<u64>,
    pub niche: SemanticNiche,
}

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
            tokenizer: Arc::new(tokenizer),
            council: council.map(Arc::new),
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

impl DNIEngine {
    pub fn initialize_original_hash(&mut self) {
        self.original_dna_hash = Self::calculate_dna_hash(&self.model);
    }

    fn calculate_dna_hash(model: &GenomicLLM) -> Vec<u64> {
        let mut hashes = Vec::new();
        for block in &model.blocks {
            hashes.push(block.gate_gen.database_ref().iter().map(|&b| b as u64).sum());
            hashes.push(block.up_gen.database_ref().iter().map(|&b| b as u64).sum());
            hashes.push(block.w_down.database_ref().iter().map(|&b| b as u64).sum());
        }
        hashes
    }

    fn profile_activations(&mut self, tokens: &[u32]) -> Vec<Vec<f32>> {
        let n_blocks = self.model.blocks.len();
        let mut activation_stats = vec![Vec::new(); n_blocks];
        self.model.clear_cache_core();
        for &token in tokens {
            if let Ok((_, h_final)) = self.model.forward_with_hidden_core(token as usize, false) {
                for stats in activation_stats.iter_mut() {
                    if stats.is_empty() {
                        *stats = vec![0.0f32; h_final.len()];
                    }
                    for (j, &val) in h_final.iter().enumerate() {
                        stats[j] += val.abs();
                    }
                }
            }
        }
        activation_stats
    }

    fn calculate_fuzzy_membership(idx: usize, sorted_anchors: &[usize], sigma: f32) -> f32 {
        if sorted_anchors.is_empty() || sigma <= 1e-6 {
            return 0.0;
        }

        // Búsqueda binaria para encontrar el ancla más cercana en O(log N)
        let pos = match sorted_anchors.binary_search(&idx) {
            Ok(_) => return 1.0, // Coincidencia exacta (ancla protegida)
            Err(p) => p,
        };

        let mut min_dist_sq = f32::MAX;

        // Comprobar vecinos inmediatos (izquierdo y derecho)
        if pos < sorted_anchors.len() {
            let dist = (sorted_anchors[pos] as f32 - idx as f32).abs();
            if dist < 10.0 {
                min_dist_sq = min_dist_sq.min(dist * dist);
            }
        }
        if pos > 0 {
            let dist = (idx as f32 - sorted_anchors[pos - 1] as f32).abs();
            if dist < 10.0 {
                min_dist_sq = min_dist_sq.min(dist * dist);
            }
        }

        if min_dist_sq == f32::MAX {
            0.0
        } else {
            (-min_dist_sq / (2.0 * sigma * sigma)).exp()
        }
    }

    fn apply_targeted_mutation_v2(
        &self,
        mutant: &mut GenomicLLM,
        rate: f32,
        activations: &[Vec<f32>],
        sigma: f32,
    ) {
        let mut rng = rand::thread_rng();
        let n_blocks = mutant.blocks.len();
        for i in 0..n_blocks {
            let block = &mut mutant.blocks[i];
            let layer_stats = activations.get(i);
            let layers = [&mut block.gate_gen, &mut block.up_gen, &mut block.w_down];
            for layer in layers {
                // Cálculo de entropía local para ajustar sigma dinámicamente
                let h = crate::compute::math::calculate_genomic_entropy_core(layer.database_ref());
                let local_sigma = sigma * (1.0 + h);

                // Optimizacion 1: Usar Vec ordenado en lugar de HashSet para búsqueda binaria
                let mut sorted_anchors: Vec<usize> = layer
                    .anchor_indices
                    .iter()
                    .map(|&idx| idx as usize)
                    .collect();
                sorted_anchors.sort_unstable();

                let n_neurons = layer.out_features;
                let row_len_bytes = layer.weight_db.len_bytes() / n_neurons;
                let bit_depth = layer.weight_db.bit_depth();
                let params_per_byte = 8 / bit_depth;

                for row in 0..n_neurons {
                    let mut row_rate = rate;
                    if let Some(stats) = layer_stats {
                        if let Some(&act) = stats.get(row) {
                            if act < 0.1 {
                                row_rate *= 5.0;
                            } else if act > 10.0 {
                                row_rate *= 0.1;
                            }
                        }
                    }

                    // Optimizacion 2: Si row_rate es extremadamente bajo, podemos saltar la fila
                    if row_rate < 1e-8 {
                        continue;
                    }

                    let row_start = row * row_len_bytes;
                    for byte_idx in 0..row_len_bytes {
                        let global_byte_idx = row_start + byte_idx;

                        for s in 0..params_per_byte as usize {
                            if rng.gen::<f32>() < row_rate {
                                let input_idx = byte_idx * params_per_byte as usize + s;
                                let global_weight_idx = row * layer.in_features + input_idx;

                                let membership = Self::calculate_fuzzy_membership(
                                    global_weight_idx,
                                    &sorted_anchors,
                                    local_sigma,
                                );

                                // Aplicamos la penalización de membership
                                if rng.gen::<f32>() < (1.0 - membership) {
                                    let current_bits = layer.weight_db.read(global_byte_idx, s);
                                    let max_val = (1 << bit_depth) - 1;
                                    let mutation = rng.gen::<u8>() % (max_val + 1);
                                    
                                    if mutation != current_bits {
                                        layer.weight_db.mutate(global_byte_idx, s, mutation);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn evaluate_mutant(&self, mutant: &mut GenomicLLM, tokens: &[u32]) -> f32 {
        mutant.clear_cache_core();
        let mut total_prob = 0.0;
        let mut count = 0;
        for i in 0..tokens.len() - 1 {
            if let Ok(logits) = mutant.forward_core(tokens[i] as usize, false) {
                let target = tokens[i + 1] as usize;
                let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut sum_exp = 0.0f32;
                for &l in &logits {
                    sum_exp += (l - max_l).exp();
                }
                total_prob += (logits[target] - max_l).exp() / (sum_exp + 1e-12);
                count += 1;
            }
        }
        if count > 0 {
            total_prob / count as f32
        } else {
            0.0
        }
    }

    fn chunk_text(&self, text: &str) -> Vec<String> {
        text.lines()
            .map(|l| l.trim().to_string())
            .filter(|l| l.len() > 20)
            .collect()
    }

    fn merge_models(&self, base: &mut GenomicLLM, mutant: &GenomicLLM) {
        let mut rng = rand::thread_rng();
        for (i, b_base) in base.blocks.iter_mut().enumerate() {
            let b_mutant = &mutant.blocks[i];
            let layers = [
                (&mut b_base.gate_gen, &b_mutant.gate_gen),
                (&mut b_base.up_gen, &b_mutant.up_gen),
                (&mut b_base.w_down, &b_mutant.w_down),
                (&mut b_base.q_gen, &b_mutant.q_gen),
                (&mut b_base.k_gen, &b_mutant.k_gen),
                (&mut b_base.v_gen, &b_mutant.v_gen),
                (&mut b_base.w_o, &b_mutant.w_o),
            ];
            for (l_base, l_mutant) in layers {
                let db_base_vec = l_base.database_ref().to_vec();
                let db_mutant = l_mutant.database_ref();
                if db_base_vec != db_mutant {
                    let mut db_base = db_base_vec;
                    let crossover_point = rng.gen_range(0..db_base.len());
                    for j in crossover_point..db_base.len() {
                        db_base[j] = db_mutant[j];
                    }
                    l_base.database_mut().copy_from_slice(&db_base);
                }
            }
        }
    }

    fn migrate_knowledge(&self, logic_model: &mut GenomicLLM, grammar_model: &mut GenomicLLM) {
        let mut rng = rand::thread_rng();
        for i in 0..logic_model.blocks.len() {
            let blk_l = &mut logic_model.blocks[i];
            let blk_g = &mut grammar_model.blocks[i];
            
            let db_l = blk_l.w_down.database_ref().to_vec();
            let mut db_g = blk_g.w_down.database_ref().to_vec();
            for j in 0..db_l.len() {
                if rng.gen_bool(0.1) {
                    db_g[j] = db_l[j];
                }
            }
            blk_g.w_down.database_mut().copy_from_slice(&db_g);

            let mut db_attn_l = blk_l.w_o.database_ref().to_vec();
            let db_attn_g = blk_g.w_o.database_ref();
            for j in 0..db_attn_l.len() {
                if rng.gen_bool(0.1) {
                    db_attn_l[j] = db_attn_g[j];
                }
            }
            blk_l.w_o.database_mut().copy_from_slice(&db_attn_l);

        }
    }
}
