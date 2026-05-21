use crate::nn::spiking::neuron::SpikingNeuron;

/// Mecanismo de Atención Neuromórfica (Spiking Self-Attention).
/// En esta arquitectura, la atención se modula mediante la densidad de disparos.
pub struct SpikingAttention {
    pub query_neurons: Vec<SpikingNeuron>,
    pub key_neurons: Vec<SpikingNeuron>,
    pub value_neurons: Vec<SpikingNeuron>,
    pub dim: usize,
    pub num_heads: usize,
}

impl SpikingAttention {
    pub fn new(dim: usize, num_heads: usize, threshold: f32, decay: f32) -> Self {
        let mut query_neurons = Vec::with_capacity(dim);
        let mut key_neurons = Vec::with_capacity(dim);
        let mut value_neurons = Vec::with_capacity(dim);

        for _ in 0..dim {
            query_neurons.push(SpikingNeuron::new(threshold, decay, dim));
            key_neurons.push(SpikingNeuron::new(threshold, decay, dim));
            value_neurons.push(SpikingNeuron::new(threshold, decay, dim));
        }

        Self {
            query_neurons,
            key_neurons,
            value_neurons,
            dim,
            num_heads,
        }
    }
}
