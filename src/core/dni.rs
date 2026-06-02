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
use crate::pyo3_shim::exceptions::PyValueError;

#[cfg_attr(feature = "python", pyclass)]
pub struct DNIEngine {
    pub model: GenomicLLM,
    pub tokenizer: Arc<GajeTokenizer>,
    pub council: Option<Arc<CouncilOfTeachers>>,
    /// Ratio de mutación para la ingesta
    pub intensity: f32,
    /// Capas objetivo para la mutación (si está vacío, se usan heurísticas)
    pub target_layers: Vec<String>,
    /// 🧬 Fase 4: Puntos de Control de Identidad (Tokens de validación)
    pub validation_tokens: Vec<u32>,
    /// Pesos originales para calcular la deriva genómica
    pub original_dna_hash: Vec<u64>,
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
        let dna_hash = Self::calculate_dna_hash(&model);
        Self {
            model,
            tokenizer: Arc::new(tokenizer),
            council: council.map(Arc::new),
            intensity,
            target_layers,
            validation_tokens: Vec::new(),
            original_dna_hash: dna_hash,
        }
    }

    /// Configura tokens de validación para prevenir el olvido catastrófico.
    pub fn set_validation_text(&mut self, text: String) {
        if let Ok(tokens) = self.tokenizer.encode(&text, false) {
            self.validation_tokens = tokens;
        }
    }

    /// Ejecuta un paso de ingestión sobre un fragmento de texto.
    pub fn ingest_text(&mut self, text: String, generations: usize, pop_size: usize) -> PyResult<f32> {
        let tokens = self.tokenizer.encode(&text, false)
            .map_err(|e| PyValueError::new_err(format!("Tokenizer error: {}", e)))?;
        
        if tokens.len() < 2 {
            return Ok(0.0);
        }

        // 🧬 Fase 2: Targeting por Activación
        let activations = self.profile_activations(&tokens);

        // Crear población de mutantes (aislados)
        let mut population: Vec<GenomicLLM> = (0..pop_size)
            .map(|_| {
                let mut mutant = self.model.clone();
                self.apply_targeted_mutation_v2(&mut mutant, self.intensity, &activations);
                mutant
            })
            .collect();

        let mut best_fitness = 0.0;

        for gen in 0..generations {
            // Evaluación en paralelo
            let scores: Vec<(usize, f32)> = population.par_iter_mut()
                .enumerate()
                .map(|(idx, mutant)| {
                    // 🧬 Fase 4: Fitness Multiobjetivo (Nuevo Conocimiento + Preservación)
                    let new_knowledge_fitness = self.evaluate_mutant(mutant, &tokens);
                    let mut final_fitness = new_knowledge_fitness;

                    if !self.validation_tokens.is_empty() {
                        let base_preservation = self.evaluate_mutant(mutant, &self.validation_tokens);
                        final_fitness = (new_knowledge_fitness * 0.8) + (base_preservation * 0.2);
                    }

                    (idx, final_fitness)
                })
                .collect();

            // Encontrar el mejor de esta generación
            let (best_idx, fitness) = scores.iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();
            
            best_fitness = *fitness;

            if gen < generations - 1 {
                let winner = population[*best_idx].clone();
                population.par_iter_mut().enumerate().for_each(|(i, mutant)| {
                    if i != *best_idx {
                        *mutant = winner.clone();
                        let current_intensity = self.intensity * (1.0 - (gen as f32 / generations as f32));
                        self.apply_targeted_mutation_v2(mutant, current_intensity, &activations);
                    }
                });
            } else {
                self.model = population[*best_idx].clone();
            }
        }

        Ok(best_fitness)
    }

    /// Ejecuta la ingesta de un documento completo utilizando el Modelo de Islas Paralelas.
    pub fn ingest_document(&mut self, document: String, generations: usize, pop_size: usize) -> PyResult<f32> {
        let chunks = self.chunk_text(&document);
        if chunks.is_empty() { return Ok(0.0); }

        println!("[*] Iniciando Island Model DNI: {} fragmentos detectados.", chunks.len());
        
        // 🧬 Fase 3: Procesamiento por Islas Paralelas
        let num_cpus = rayon::current_num_threads();
        let chunks_per_island = (chunks.len() as f32 / num_cpus as f32).ceil() as usize;
        
        let island_results: Vec<(GenomicLLM, f32)> = chunks.chunks(chunks_per_island)
            .enumerate()
            .par_bridge()
            .map(|(island_id, island_chunks)| {
                let mut island_engine = DNIEngine {
                    model: self.model.clone(),
                    tokenizer: self.tokenizer.clone(),
                    council: self.council.clone(),
                    intensity: self.intensity,
                    target_layers: self.target_layers.clone(),
                    validation_tokens: self.validation_tokens.clone(),
                    original_dna_hash: Vec::new(),
                };
                
                let mut total_fitness = 0.0;
                for chunk in island_chunks {
                    if let Ok(f) = island_engine.ingest_text(chunk.to_string(), generations, pop_size) {
                        total_fitness += f;
                    }
                }
                
                let avg_fitness = if island_chunks.is_empty() { 0.0 } else { total_fitness / island_chunks.len() as f32 };
                println!("    [🏝️ Isla #{}] Ingesta completada. Fitness Medio: {:.6}", island_id, avg_fitness);
                (island_engine.model, avg_fitness)
            })
            .collect();

        // 🧬 Fase 3: Fusión de Mutantes (Knowledge Fusion)
        if !island_results.is_empty() {
            println!("[*] Fusionando conocimientos de {} islas...", island_results.len());
            let mut best_overall_fitness: f32 = 0.0;
            let mut final_model = self.model.clone();

            for (mutant_model, fitness) in island_results {
                self.merge_models(&mut final_model, &mutant_model);
                best_overall_fitness = best_overall_fitness.max(fitness);
            }
            
            self.model = final_model;

            // 🧬 Fase 4: Reportar Deriva Genómica
            let current_hash = Self::calculate_dna_hash(&self.model);
            let drift = self.calculate_drift(&current_hash);
            println!("[📊] Deriva Genómica Detectada: {:.4}% (Capas mutadas respecto al original)", drift * 100.0);

            Ok(best_overall_fitness)
        } else {
            Ok(0.0)
        }
    }
}

impl DNIEngine {
    /// Inicializa manualmente el hash original (útil para el CLI nativo).
    pub fn initialize_original_hash(&mut self) {
        self.original_dna_hash = Self::calculate_dna_hash(&self.model);
    }

    /// Calcula un hash simplificado de los pesos para medir la deriva.
    fn calculate_dna_hash(model: &GenomicLLM) -> Vec<u64> {
        let mut hashes = Vec::new();
        for block in &model.blocks {
            hashes.push(block.gate_gen.database.iter().map(|&b| b as u64).sum());
            hashes.push(block.up_gen.database.iter().map(|&b| b as u64).sum());
            hashes.push(block.w_down.database.iter().map(|&b| b as u64).sum());
        }
        hashes
    }

    /// Calcula el porcentaje de deriva genómica.
    fn calculate_drift(&self, current_hash: &[u64]) -> f32 {
        if self.original_dna_hash.is_empty() || self.original_dna_hash.len() != current_hash.len() {
            return 0.0;
        }
        let mut changed_layers = 0;
        for i in 0..self.original_dna_hash.len() {
            if self.original_dna_hash[i] != current_hash[i] {
                changed_layers += 1;
            }
        }
        changed_layers as f32 / self.original_dna_hash.len() as f32
    }

    /// Perfila las activaciones del modelo para identificar neuronas candidatas.
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

    /// Aplica mutación bitwise quirúrgica respetando anclas y priorizando neuronas silenciosas.
    fn apply_targeted_mutation_v2(&self, mutant: &mut GenomicLLM, rate: f32, activations: &[Vec<f32>]) {
        let mut rng = rand::thread_rng();
        let n_blocks = mutant.blocks.len();

        let start_block = if n_blocks > 4 { n_blocks / 4 } else { 0 };
        let end_block = n_blocks - 1;

        for i in start_block..=end_block {
            let block = &mut mutant.blocks[i];
            let layer_stats = activations.get(i);
            
            let layers = [
                &mut block.gate_gen, 
                &mut block.up_gen, 
                &mut block.w_down
            ];

            for layer in layers {
                let protected_indices: std::collections::HashSet<usize> = layer.anchor_indices.iter().map(|&idx| idx as usize).collect();

                let mut db = (*layer.database).clone();
                let mut changed = false;
                let n_neurons = layer.out_features;

                for row in 0..n_neurons {
                    let mut row_rate = rate;
                    if let Some(stats) = layer_stats {
                        if let Some(&act) = stats.get(row) {
                            if act < 0.1 { row_rate *= 5.0; }
                            else if act > 10.0 { row_rate *= 0.1; }
                        }
                    }

                    let row_start = row * (layer.database.len() / n_neurons);
                    let row_end = row_start + (layer.database.len() / n_neurons);

                    for byte_idx in row_start..row_end {
                        let mut is_protected = false;
                        for s in 0..4 {
                            let input_idx = (byte_idx - row_start) * 4 + s;
                            if protected_indices.contains(&(row * layer.in_features + input_idx)) {
                                is_protected = true;
                                break;
                            }
                        }

                        if !is_protected && rng.gen::<f32>() < row_rate {
                            *db.get_mut(byte_idx).unwrap() ^= rng.gen::<u8>();
                            changed = true;
                        }
                    }
                }

                if changed {
                    layer.database = Arc::new(db);
                }
            }
        }
    }

    /// Evalúa la coherencia de un mutante respecto a los tokens objetivo.
    fn evaluate_mutant(&self, mutant: &mut GenomicLLM, tokens: &[u32]) -> f32 {
        mutant.clear_cache_core();
        let mut total_prob = 0.0;
        let mut count = 0;

        for i in 0..tokens.len() - 1 {
            if let Ok(logits) = mutant.forward_core(tokens[i] as usize, false) {
                let target = tokens[i+1] as usize;
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

    /// Divide un documento en fragmentos semánticos (Cromosomas).
    fn chunk_text(&self, text: &str) -> Vec<String> {
        text.lines()
            .map(|l| l.trim().to_string())
            .filter(|l| l.len() > 20)
            .collect()
    }

    /// Fusiona dos modelos genómicos mediante Crossover Bitwise (Fusión de Conocimiento).
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
                let mut db_base = (*l_base.database).clone();
                let db_mutant = &l_mutant.database;
                if db_base != **db_mutant {
                    let len = db_base.len();
                    let crossover_point = rng.gen_range(0..len);
                    for j in crossover_point..len {
                        db_base[j] = db_mutant[j];
                    }
                    l_base.database = Arc::new(db_base);
                }
            }
        }
    }
}
