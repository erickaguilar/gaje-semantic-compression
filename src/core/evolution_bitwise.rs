use crate::nn::spiking::layer::GajeNeuromorphicLayer;
use crate::compute::scheduler::NeuromorphicScheduler;
use crate::core::topology::CentroidGraph;
use crate::nn::llm::GenomicLLM;
use crate::nn::distiller::CouncilOfTeachers;
use crate::core::tokenizer::GajeTokenizer;
use rand::Rng;
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

/// Representa un "Organismo" Neuromórfico o LLM en la población evolutiva.
#[derive(Clone)]
pub struct NeuromorphicOrganism {
    pub layers: Vec<GajeNeuromorphicLayer>,
    pub llm: Option<GenomicLLM>, // Opcional para modelos LLM como Silver Fetus
    pub fitness: f32,
}

impl NeuromorphicOrganism {
    pub fn new(layers: Vec<GajeNeuromorphicLayer>) -> Self {
        Self { layers, llm: None, fitness: 0.0 }
    }

    pub fn from_llm(llm: GenomicLLM) -> Self {
        Self { layers: Vec::new(), llm: Some(llm), fitness: 0.0 }
    }

    /// Aplica mutaciones bitwise ultra-rápidas.
    pub fn mutate(&mut self, rate: f32) {
        let mut rng = rand::thread_rng();
        
        // Mutar capas spiking si existen
        for layer in &mut self.layers {
            for byte in &mut layer.packed_weights {
                if rng.gen::<f32>() < rate {
                    *byte ^= rng.gen::<u8>();
                }
            }
        }

        // Mutar LLM si existe (Paso crucial para Silver Fetus)
        if let Some(llm) = &mut self.llm {
            for block in &mut llm.blocks {
                let layers_to_mutate = [
                    &mut block.q_gen, &mut block.k_gen, &mut block.v_gen, 
                    &mut block.w_o, &mut block.gate_gen, &mut block.up_gen, &mut block.w_down
                ];
                
                for layer in layers_to_mutate {
                    // Dado que database es Arc<Vec<u8>>, necesitamos hacerlo mutable.
                    // En la evolución, cada organismo debe tener su propia copia de los pesos.
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

    pub fn crossover(&self, other: &Self) -> Self {
        let mut rng = rand::thread_rng();
        
        if let (Some(llm_a), Some(llm_b)) = (&self.llm, &other.llm) {
            let mut child_llm = llm_a.clone();
            for (i, block) in child_llm.blocks.iter_mut().enumerate() {
                let other_block = &llm_b.blocks[i];
                
                let pairs = [
                    (&mut block.q_gen, &other_block.q_gen),
                    (&mut block.k_gen, &other_block.k_gen),
                    (&mut block.v_gen, &other_block.v_gen),
                    (&mut block.w_o, &other_block.w_o),
                    (&mut block.gate_gen, &other_block.gate_gen),
                    (&mut block.up_gen, &other_block.up_gen),
                    (&mut block.w_down, &other_block.w_down),
                ];

                for (l_child, l_other) in pairs {
                    let mut db_child = (*l_child.database).clone();
                    let db_other = &l_other.database;
                    let len = db_child.len();
                    if len > 1 {
                        let cp = rng.gen_range(0..len);
                        for j in cp..len {
                            db_child[j] = db_other[j];
                        }
                        l_child.database = Arc::new(db_child);
                    }
                }
            }
            return Self::from_llm(child_llm);
        }

        let mut child_layers = self.layers.clone();
        for (i, layer) in child_layers.iter_mut().enumerate() {
            let other_layer = &other.layers[i];
            let len = layer.packed_weights.len();
            if len > 1 {
                let crossover_point = rng.gen_range(0..len);
                for j in crossover_point..len {
                    layer.packed_weights[j] = other_layer.packed_weights[j];
                }
            }
        }
        Self { layers: child_layers, llm: None, fitness: 0.0 }
    }
}

/// Motor de Evolución por Poblaciones Paralelas (Island Model).
pub struct IslandModelEngine {
    pub islands: Vec<SpikingEvolutionEngine>,
    pub migration_rate: usize,
    pub topology_map: Option<Arc<CentroidGraph>>,
    pub generation: usize,
    pub council: Option<Arc<CouncilOfTeachers>>,
    pub tokenizer: Option<Arc<GajeTokenizer>>,
}

impl IslandModelEngine {
    pub fn new(
        initial_model: Vec<GajeNeuromorphicLayer>,
        num_islands: usize,
        pop_per_island: usize,
        centroides: [f32; 4],
        mutation_rate: f32,
        migration_rate: usize,
        topology_map: Option<Arc<CentroidGraph>>,
    ) -> Self {
        let mut islands = Vec::with_capacity(num_islands);
        for _ in 0..num_islands {
            islands.push(SpikingEvolutionEngine::new(
                initial_model.clone(),
                pop_per_island,
                centroides,
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
        if self.council.is_none() || self.tokenizer.is_none() { return; }
        let council = self.council.as_ref().unwrap().clone();
        let tokenizer = self.tokenizer.as_ref().unwrap().clone();
        
        // Realizar Needle Test periódicamente
        let run_needle = self.generation % 10 == 0;
        let needle_key = "La contraseña secreta es: ORO-SILVER-2026";
        let needle_question = "¿Cuál es la contraseña secreta?";

        for island in &mut self.islands {
            island.population.par_iter_mut().for_each(|organism| {
                if let Some(llm) = &mut organism.llm {
                    let mut total_score = 0.0;
                    
                    // 1. Fitness de Coherencia (70%)
                    let mut coherence_score = 0.0;
                    for text in texts {
                        let consensus = council.get_consensus_probs(text, llm.lm_head.out_features);
                        let tokens = tokenizer.encode(text, false).unwrap_or_default();
                        
                        llm.clear_cache_core();
                        let mut match_p = 0.0;
                        let steps = (tokens.len() - 1).min(consensus.len());
                        
                        for i in 0..steps {
                            if let Ok(logits) = llm.forward_core(tokens[i] as usize, false) {
                                // Softmax manual rápido
                                let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                                let mut sum_exp = 0.0f32;
                                let mut probs = vec![0.0f32; logits.len()];
                                for (j, &l) in logits.iter().enumerate() {
                                    let e = (l - max_l).exp();
                                    probs[j] = e;
                                    sum_exp += e;
                                }
                                for p in &mut probs { *p /= sum_exp + 1e-12; }
                                
                                // Similitud con el consenso del maestro
                                let teacher_p = &consensus[i];
                                for (j, &tp) in teacher_p.iter().enumerate() {
                                    match_p += (tp * probs[j]).sqrt(); // Hellinger-ish similarity
                                }
                            }
                        }
                        if steps > 0 { coherence_score += match_p / (steps as f32); }
                    }
                    coherence_score /= texts.len() as f32;
                    total_score += coherence_score * 0.7;

                    // 2. Fitness de Retención / Needle Test (30%)
                    if run_needle {
                        let context = format!("{} ... contenido aleatorio ... {}", needle_key, needle_question);
                        let tokens = tokenizer.encode(&context, false).unwrap_or_default();
                        llm.clear_cache_core();
                        
                        let mut needle_success = 0.0;
                        for (i, &t) in tokens.iter().enumerate() {
                            if let Ok(logits) = llm.forward_core(t as usize, false) {
                                if i == tokens.len() - 1 {
                                    // Ver si el token más probable es parte de la respuesta "ORO"
                                    let mut best_id = 0;
                                    let mut max_l = f32::NEG_INFINITY;
                                    for (id, &l) in logits.iter().enumerate() {
                                        if l > max_l { max_l = l; best_id = id; }
                                    }
                                    if let Ok(word) = tokenizer.decode(&[best_id as u32], true) {
                                        if word.contains("ORO") || word.contains("SILVER") {
                                            needle_success = 1.0;
                                        }
                                    }
                                }
                            }
                        }
                        total_score += needle_success * 0.3;
                    } else {
                        total_score += 0.15; // Placeholder score para no sesgar
                    }

                    organism.fitness = total_score;
                }
            });
            
            // Re-ordenar por el nuevo fitness híbrido
            island.population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));
        }
    }

    pub fn step(&mut self) {
        self.islands.par_iter_mut().for_each(|island| {
            island.evolve();
        });
        self.generation += 1;
        if self.generation % self.migration_rate == 0 {
            self.migrate();
        }
    }

    pub fn migrate(&mut self) {
        let num_islands = self.islands.len();
        if num_islands < 2 { return; }
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

/// Motor de Evolución para el Emulador Neuromórfico Industrial.
pub struct SpikingEvolutionEngine {
    pub population: Vec<NeuromorphicOrganism>,
    pub centroides: [f32; 4],
    pub mutation_rate: f32,
}

impl SpikingEvolutionEngine {
    pub fn new(
        initial_model: Vec<GajeNeuromorphicLayer>,
        pop_size: usize,
        centroides: [f32; 4],
        mutation_rate: f32,
    ) -> Self {
        let mut population = Vec::with_capacity(pop_size);
        for _ in 0..pop_size {
            let mut organism = NeuromorphicOrganism::new(initial_model.clone());
            organism.mutate(mutation_rate); 
            population.push(organism);
        }
        Self { population, centroides, mutation_rate }
    }

    /// Crea un motor de evolución específico para LLMs.
    pub fn new_llm(
        initial_llm: GenomicLLM,
        pop_size: usize,
        mutation_rate: f32,
    ) -> Self {
        let mut population = Vec::with_capacity(pop_size);
        for _ in 0..pop_size {
            let mut organism = NeuromorphicOrganism::from_llm(initial_llm.clone());
            organism.mutate(mutation_rate);
            population.push(organism);
        }
        Self { 
            population, 
            centroides: [0.0; 4], 
            mutation_rate 
        }
    }

    pub fn evaluate(&mut self, input_spikes: &[(usize, usize)], target_frequencies: &[f32]) {
        let centroides = self.centroides;
        self.population.par_iter_mut().for_each(|organism| {
            let mut scheduler = NeuromorphicScheduler::new(centroides, 1);
            for &(layer_id, neuron_id) in input_spikes {
                scheduler.inject_spike(layer_id, neuron_id, 0, 0, 1.0);
            }
            let output_spikes = scheduler.run_to_completion(&mut organism.layers);
            let mut score = 0.0;
            let total_outputs = output_spikes.len() as f32;
            if total_outputs > 0.0 {
                score = total_outputs / (target_frequencies.len() as f32);
            }
            organism.fitness = score;
        });
        self.population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).expect("Fitness NaN detectado"));
    }

    pub fn evolve(&mut self) {
        let pop_size = self.population.len();
        let elite_count = (pop_size / 10).max(1);
        let mut new_population = Vec::with_capacity(pop_size);
        for i in 0..elite_count {
            new_population.push(self.population[i].clone());
        }
        let mut rng = rand::thread_rng();
        while new_population.len() < pop_size {
            let parent_a_idx = rng.gen_range(0..elite_count);
            let parent_b_idx = rng.gen_range(0..elite_count);
            let parent_a = &self.population[parent_a_idx];
            let parent_b = &self.population[parent_b_idx];
            let mut child = parent_a.crossover(parent_b);
            child.mutate(self.mutation_rate);
            new_population.push(child);
        }
        self.population = new_population;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evolution_flow_soa() {
        let centroides = [0.0, 0.5, 0.8, 1.2];
        
        // Crear modelo base SoA: 1 capa, 1 neurona, 1 input
        let layer = GajeNeuromorphicLayer::new(1, 1, 1.0, 0.9);
        let layers = vec![layer];

        let mut engine = SpikingEvolutionEngine::new(layers, 10, centroides, 0.1);
        
        // Evaluar
        engine.evaluate(&[(0, 0)], &[1.0]);
        let best_fitness_before = engine.population[0].fitness;
        
        engine.evolve();
        engine.evaluate(&[(0, 0)], &[1.0]);
        let best_fitness_after = engine.population[0].fitness;
        
        assert!(best_fitness_after >= best_fitness_before);
    }
}
