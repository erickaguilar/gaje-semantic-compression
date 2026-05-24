use crate::nn::spiking::layer::GajeNeuromorphicLayer;
use crate::compute::scheduler::NeuromorphicScheduler;
use rand::Rng;
use rayon::prelude::*;

/// Representa un "Organismo" Neuromórfico en la población evolutiva usando SoA.
#[derive(Clone)]
pub struct NeuromorphicOrganism {
    pub layers: Vec<GajeNeuromorphicLayer>,
    pub fitness: f32,
}

impl NeuromorphicOrganism {
    pub fn new(layers: Vec<GajeNeuromorphicLayer>) -> Self {
        Self { layers, fitness: 0.0 }
    }

    /// Aplica mutaciones bitwise ultra-rápidas directamente sobre el buffer de pesos masivo.
    pub fn mutate(&mut self, rate: f32) {
        let mut rng = rand::thread_rng();
        for layer in &mut self.layers {
            for byte in &mut layer.packed_weights {
                if rng.gen::<f32>() < rate {
                    // Mutación Bitwise: XOR con una máscara aleatoria
                    let mask = rng.gen::<u8>();
                    *byte ^= mask;
                }
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
            organism.mutate(mutation_rate); // Diversidad inicial
            population.push(organism);
        }
        Self {
            population,
            centroides,
            mutation_rate,
        }
    }

    /// Evalúa la población en paralelo utilizando Spike Frequency Accuracy (SFA).
    pub fn evaluate(&mut self, input_spikes: &[(usize, usize)], target_frequencies: &[f32]) {
        let centroides = self.centroides;
        
        self.population.par_iter_mut().for_each(|organism| {
            let mut scheduler = NeuromorphicScheduler::new(centroides, 1);
            
            // 1. Inyectar estímulos
            for &(layer_id, neuron_id) in input_spikes {
                scheduler.inject_spike(layer_id, neuron_id, 0, 0, 1.0);
            }

            // 2. Ejecutar simulación
            let output_spikes = scheduler.run_to_completion(&mut organism.layers);

            // 3. Calcular Fitness (SFA)
            let mut score = 0.0;
            let total_outputs = output_spikes.len() as f32;
            
            if total_outputs > 0.0 {
                score = total_outputs / (target_frequencies.len() as f32);
            }
            
            organism.fitness = score;
        });

        // Ordenar por fitness (descendente)
        self.population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).expect("Fitness NaN detectado"));
    }

    /// Produce la siguiente generación mediante selección y mutación.
    pub fn evolve(&mut self) {
        let pop_size = self.population.len();
        let elite_count = (pop_size / 10).max(1); // Mantener el 10% mejor
        
        let mut new_population = Vec::with_capacity(pop_size);
        
        // 1. Elitismo
        for i in 0..elite_count {
            new_population.push(self.population[i].clone());
        }

        // 2. Reproducción y Mutación
        while new_population.len() < pop_size {
            let parent_idx = rand::thread_rng().gen_range(0..elite_count);
            let mut child = self.population[parent_idx].clone();
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
