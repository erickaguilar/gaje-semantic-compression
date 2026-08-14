// =============================================================================
// organism — NeuromorphicOrganism: mutación bitwise y crossover
// =============================================================================
use rand::Rng;

use crate::nn::linear::GenomicOperable;
use crate::nn::llm::GenomicLLM;
use crate::nn::spiking::layer::GajeNeuromorphicLayer;

/// Representa un "Organismo" Neuromórfico o LLM en la población evolutiva.
#[derive(Clone)]
pub struct NeuromorphicOrganism {
    pub layers: Vec<GajeNeuromorphicLayer>,
    pub llm: Option<GenomicLLM>, // Opcional para modelos LLM como Silver Fetus
    pub fitness: f32,
}

impl NeuromorphicOrganism {
    pub fn new(layers: Vec<GajeNeuromorphicLayer>) -> Self {
        Self {
            layers,
            llm: None,
            fitness: 0.0,
        }
    }

    pub fn from_llm(llm: GenomicLLM) -> Self {
        Self {
            layers: Vec::new(),
            llm: Some(llm),
            fitness: 0.0,
        }
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
                    &mut block.q_gen,
                    &mut block.k_gen,
                    &mut block.v_gen,
                    &mut block.w_o,
                    &mut block.gate_gen,
                    &mut block.up_gen,
                    &mut block.w_down,
                ];

                for layer in layers_to_mutate {
                    let db_len = layer.weight_db.len_bytes();
                    let bit_depth = layer.weight_db.bit_depth();
                    let params_per_byte = 8 / bit_depth;

                    for i in 0..db_len {
                        for s in 0..params_per_byte as usize {
                            if rng.gen::<f32>() < rate {
                                let max_val = (1 << bit_depth) - 1;
                                let mutation = rng.gen::<u8>() % (max_val + 1);
                                layer.weight_db.mutate(i, s, mutation);
                            }
                        }
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
                    let len = l_child.weight_db.len_bytes();
                    if len > 1 {
                        let cp = rng.gen_range(0..len);
                        let db_child_mut = l_child.database_mut();
                        let db_other = l_other.database_ref();
                        for j in cp..len {
                            db_child_mut[j] = db_other[j];
                        }
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
        Self {
            layers: child_layers,
            llm: None,
            fitness: 0.0,
        }
    }
}
