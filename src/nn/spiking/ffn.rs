use crate::nn::spiking::neuron::SpikingNeuron;

/// Capa FFN Neuromórfica.
/// Implementa la lógica de proyección lineal usando neuronas LIF.
pub struct SpikingFFN {
    pub neurons: Vec<SpikingNeuron>,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl SpikingFFN {
    pub fn new(input_dim: usize, output_dim: usize, threshold: f32, decay: f32) -> Self {
        let mut neurons = Vec::with_capacity(output_dim);
        for _ in 0..output_dim {
            neurons.push(SpikingNeuron::new(threshold, decay, input_dim));
        }
        Self {
            neurons,
            input_dim,
            output_dim,
        }
    }

    /// Acceso a una neurona específica para integración.
    pub fn get_neuron_mut(&mut self, index: usize) -> Option<&mut SpikingNeuron> {
        self.neurons.get_mut(index)
    }
}
