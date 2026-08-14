// =============================================================================
// engine — SpikingEvolutionEngine: población y evolución
// =============================================================================
use rand::Rng;
use rayon::prelude::*;

use crate::compute::scheduler::NeuromorphicScheduler;
use crate::core::evolution_bitwise::organism::NeuromorphicOrganism;
use crate::nn::llm::GenomicLLM;
use crate::nn::spiking::layer::GajeNeuromorphicLayer;

pub struct SpikingEvolutionEngine {
    pub population: Vec<NeuromorphicOrganism>,
    pub centroides_real: [f32; 4],
    pub centroides_imag: [f32; 4],
    pub mutation_rate: f32,
}

impl SpikingEvolutionEngine {
    pub fn new(
        initial_model: Vec<GajeNeuromorphicLayer>,
        pop_size: usize,
        centroides_real: [f32; 4],
        centroides_imag: [f32; 4],
        mutation_rate: f32,
    ) -> Self {
        let mut population = Vec::with_capacity(pop_size);
        for _ in 0..pop_size {
            let mut organism = NeuromorphicOrganism::new(initial_model.clone());
            organism.mutate(mutation_rate);
            population.push(organism);
        }
        Self {
            population,
            centroides_real,
            centroides_imag,
            mutation_rate,
        }
    }

    pub fn new_llm(initial_llm: GenomicLLM, pop_size: usize, mutation_rate: f32) -> Self {
        let mut population = Vec::with_capacity(pop_size);
        for _ in 0..pop_size {
            let mut organism = NeuromorphicOrganism::from_llm(initial_llm.clone());
            organism.mutate(mutation_rate);
            population.push(organism);
        }
        Self {
            population,
            centroides_real: [0.0; 4],
            centroides_imag: [0.0; 4],
            mutation_rate,
        }
    }

    pub fn evaluate(&mut self, input_spikes: &[(usize, usize)], target_frequencies: &[f32]) {
        let c_r = self.centroides_real;
        let c_im = self.centroides_imag;
        self.population.par_iter_mut().for_each(|organism| {
            let mut scheduler = NeuromorphicScheduler::new(c_r, c_im, 1);
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
        self.population.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .expect("Fitness NaN detectado")
        });
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
