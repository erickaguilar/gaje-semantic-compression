// =============================================================================
// init — Construcción, reset y manejo de anclas de GajeNeuromorphicLayer
// =============================================================================
use crate::compute::lagrangian::LagrangianEngine;
use crate::nn::spiking::layer::GajeNeuromorphicLayer;

impl GajeNeuromorphicLayer {
    pub fn new(num_neurons: usize, weights_per_neuron: usize, threshold: f32, decay: f32) -> Self {
        let row_size = (num_neurons + 3) / 4;
        let packed_size = weights_per_neuron * row_size;

        // Inicialización de alta entropía (Ruido blanco genómico)
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut packed_weights = vec![0u8; packed_size];
        for byte in packed_weights.iter_mut() {
            *byte = rng.gen();
        }

        Self {
            membrane_potentials_real: vec![0.0; num_neurons],
            membrane_potentials_imag: vec![0.0; num_neurons],
            thresholds: vec![threshold; num_neurons],
            decays: vec![decay; num_neurons],
            packed_weights,
            anchor_indices: Vec::new(),
            anchor_values: Vec::new(),
            anchor_row_ptrs: vec![0; num_neurons + 1],
            num_neurons,
            weights_per_neuron,
            k_wta: (num_neurons / 10).max(1),
            rms_ema: 1.0,
            lagrangian: LagrangianEngine::new(1.0),
        }
    }

    pub fn reset_potentials(&mut self) {
        self.membrane_potentials_real.fill(0.0);
        self.membrane_potentials_imag.fill(0.0);
    }

    pub fn anchors_sparse_buffer(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GAJE");
        let count = self.anchor_indices.len();
        out.extend_from_slice(&(count as u32).to_le_bytes());
        for &idx in self.anchor_indices.iter() {
            out.extend_from_slice(&idx.to_le_bytes());
        }
        for &val in self.anchor_values.iter() {
            out.extend_from_slice(&val.to_le_bytes());
        }
        for &ptr in self.anchor_row_ptrs.iter() {
            out.extend_from_slice(&(ptr as u64).to_le_bytes());
        }
        out
    }

    pub fn load_anchors_from_u8(&mut self, anchors_u8: &[u8]) {
        if anchors_u8.len() >= 4 && &anchors_u8[0..4] == b"GAJE" {
            let count = u32::from_le_bytes(anchors_u8[4..8].try_into().unwrap()) as usize;
            let mut indices = Vec::with_capacity(count);
            let mut values = Vec::with_capacity(count);
            let mut row_ptrs = vec![0; self.num_neurons + 1];
            let idx_s = 8;
            let val_s = idx_s + count * 4;
            let ptr_s = val_s + count * 2;
            for i in 0..count {
                indices.push(u32::from_le_bytes(
                    anchors_u8[idx_s + i * 4..idx_s + i * 4 + 4]
                        .try_into()
                        .unwrap(),
                ));
                values.push(half::f16::from_le_bytes(
                    anchors_u8[val_s + i * 2..val_s + i * 2 + 2]
                        .try_into()
                        .unwrap(),
                ));
            }
            for i in 0..=self.num_neurons {
                row_ptrs[i] = u64::from_le_bytes(
                    anchors_u8[ptr_s + i * 8..ptr_s + i * 8 + 8]
                        .try_into()
                        .unwrap(),
                ) as usize;
            }
            self.anchor_indices = indices;
            self.anchor_values = values;
            self.anchor_row_ptrs = row_ptrs;
        }
    }
}
