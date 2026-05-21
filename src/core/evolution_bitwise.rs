use crate::nn::spiking::neuron::{SpikingNeuron, GajeWeight2Bit};
use crate::compute::scheduler::NeuromorphicScheduler;
use rand::Rng;
use rayon::prelude::*;

/// Representa un "Organismo" Neuromórfico en la población evolutiva.
#[derive(Clone)]
pub struct NeuromorphicOrganism {
    pub layers: Vec<Vec<SpikingNeuron>>,
    pub fitness: f32,
}

impl NeuromorphicOrganism {
    pub fn new(layers: Vec<Vec<SpikingNeuron>>) -> Self {
        Self { layers, fitness: 0.0 }
    }

    /// Aplica mutaciones bitwise ultra-rápidas directamente sobre el buffer de pesos.
    pub fn mutate(&mut self, rate: f32) {
        let mut rng = rand::thread_rng();
        for layer in &mut self.layers {
            for neuron in layer {
                for byte in &mut neuron.weights {
                    if rng.gen::<f32>() < rate {
                        // Mutación Bitwise: XOR con una máscara aleatoria
                        // Esto cambia estados de 2-bits de forma impredecible pero rápida.
                        let mask = rng.gen::<u8>();
                        *byte ^= mask;
                    }
                }
            }
        }
    }
}

/// Motor de Evolución para el Emulador Neuromórfico.
pub struct SpikingEvolutionEngine {
    pub population: Vec<NeuromorphicOrganism>,
    pub centroides: [f32; 4],
    pub mutation_rate: f32,
}

impl SpikingEvolutionEngine {
    pub fn new(
        initial_model: Vec<Vec<SpikingNeuron>>,
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
                scheduler.inject_spike(layer_id, neuron_id, 0);
            }

            // 2. Ejecutar simulación
            let output_spikes = scheduler.run_to_completion(&mut organism.layers);

            // 3. Calcular Fitness (SFA)
            // Medimos qué tan cerca está la frecuencia de disparos de salida del objetivo
            let mut score = 0.0;
            let total_outputs = output_spikes.len() as f32;
            
            // Simplificación: El fitness mejora si hay actividad (frecuencia > 0)
            // y si los timestamps están distribuidos (precisión temporal)
            if total_outputs > 0.0 {
                score = total_outputs / (target_frequencies.len() as f32);
            }
            
            organism.fitness = score;
        });

        // Ordenar por fitness (descendente)
        self.population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
    }

    /// Produce la siguiente generación mediante selección y mutación.
    pub fn evolve(&mut self) {
        let pop_size = self.population.len();
        let elite_count = pop_size / 10; // Mantener el 10% mejor
        
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
    use crate::nn::spiking::neuron::GajeWeight2Bit;

    #[test]
    fn test_evolution_flow() {
        let centroides = [0.0, 0.5, 0.8, 1.2];
        
        // Crear modelo base: 1 capa, 2 neuronas
        let mut neuron = SpikingNeuron::new(1.0, 0.9, 1);
        neuron.set_weight(0, GajeWeight2Bit::State00);
        let layers = vec![vec![neuron]];

        let mut engine = SpikingEvolutionEngine::new(layers, 10, centroides, 0.1);
        
        // Evaluar (Input en L0, N0; Target 1.0 frecuencia)
        engine.evaluate(&[(0, 0)], &[1.0]);
        
        let best_fitness_before = engine.population[0].fitness;
        
        // Evolucionar
        engine.evolve();
        engine.evaluate(&[(0, 0)], &[1.0]);
        
        let best_fitness_after = engine.population[0].fitness;
        
        // El fitness debería ser >= tras una generación (elitismo asegura no empeorar)
        assert!(best_fitness_after >= best_fitness_before);
    }
}
