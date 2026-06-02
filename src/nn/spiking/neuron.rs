use crate::compute::lagrangian::LagrangianEngine;
use crate::compute::event_queue::SpikeEvent;

/// Representación de los 4 estados posibles de los centroides de 2-bits.
/// Estos estados se mapean directamente a los centroides calibrados durante el entrenamiento.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum GajeWeight2Bit {
    State00 = 0, // 0b00
    State01 = 1, // 0b01
    State10 = 2, // 0b10
    State11 = 3, // 0b11
}

impl From<u8> for GajeWeight2Bit {
    fn from(value: u8) -> Self {
        match value & 0x03 {
            0 => GajeWeight2Bit::State00,
            1 => GajeWeight2Bit::State01,
            2 => GajeWeight2Bit::State10,
            3 => GajeWeight2Bit::State11,
            _ => unreachable!(),
        }
    }
}

/// Estructura de una neurona neuromórfica emulada (Leaky Integrate-and-Fire).
/// Diseñada para procesamiento de 2-bits sin multiplicaciones.
#[derive(Clone)]
pub struct SpikingNeuron {
    pub membrane_potential_real: f32,    // El "voltaje" acumulado (Real)
    pub membrane_potential_imag: f32,    // El "voltaje" acumulado (Imaginario)
    pub threshold: f32,             // Umbral de disparo (Ancla FFN)
    pub decay: f32,                 // Factor de fuga (Leaky) entre 0.0 y 1.0
    pub weights: Vec<u8>,           // Pesos empaquetados (4 pesos de 2-bits por u8)
    pub num_weights: usize,         // Número total de pesos individuales
    pub lagrangian: LagrangianEngine, // Motor de física semántica
}

impl SpikingNeuron {
    /// Crea una nueva neurona LIF.
    pub fn new(threshold: f32, decay: f32, num_weights: usize) -> Self {
        // Calculamos cuántos u8 necesitamos para num_weights (4 pesos por byte)
        let packed_size = (num_weights + 3) / 4;
        Self {
            membrane_potential_real: 0.0,
            membrane_potential_imag: 0.0,
            threshold,
            decay,
            weights: vec![0; packed_size],
            num_weights,
            lagrangian: LagrangianEngine::new(1.0), // Masa unitaria por defecto
        }
    }

    /// Obtiene un peso individual descomprimiéndolo al vuelo.
    #[inline(always)]
    pub fn get_weight(&self, index: usize) -> GajeWeight2Bit {
        let byte_index = index / 4;
        let bit_shift = (index % 4) * 2;
        let packed_byte = self.weights[byte_index];
        GajeWeight2Bit::from((packed_byte >> bit_shift) & 0x03)
    }

    /// Establece un peso individual empaquetándolo.
    pub fn set_weight(&mut self, index: usize, weight: GajeWeight2Bit) {
        let byte_index = index / 4;
        let bit_shift = (index % 4) * 2;
        let weight_val = weight as u8;
        
        // Limpiamos los 2 bits actuales y ponemos el nuevo valor
        self.weights[byte_index] &= !(0x03 << bit_shift);
        self.weights[byte_index] |= weight_val << bit_shift;
    }

    /// Integra un impulso eléctrico (Spike) entrante usando el Principio de Mínima Acción.
    /// * `input_index`: Índice del peso.
    /// * `centroides_real/imag`: Centroides de fase.
    /// * `semantic_resistance`: Resistencia impuesta por las anclas de estabilidad.
    pub fn integrate_lagrangian(
        &mut self, 
        input_index: usize, 
        centroides_real: &[f32; 4], 
        centroides_imag: &[f32; 4],
        semantic_resistance: f32
    ) {
        let weight = self.get_weight(input_index);
        let delta_real = centroides_real[weight as usize];
        let delta_imag = centroides_imag[weight as usize];

        // Calculamos la velocidad actual (impulso entrante)
        let velocity = (delta_real.powi(2) + delta_imag.powi(2)).sqrt();
        
        // El Lagrangiano ajusta la integración: si la resistencia es alta, 
        // la aceleración geodésica frena el avance.
        let acceleration = self.lagrangian.geodesic_acceleration(-semantic_resistance);
        let velocity_adjusted = (velocity + acceleration).max(0.0);

        // Escalamos el delta por la velocidad ajustada
        if velocity > 0.0 {
            let scale = velocity_adjusted / velocity;
            self.membrane_potential_real += delta_real * scale;
            self.membrane_potential_imag += delta_imag * scale;
        }
    }

    /// Integra un impulso eléctrico (Spike) entrante.
    /// ¡Cero Multiplicaciones!: Solo suma el valor del centroide correspondiente.
    pub fn integrate(&mut self, input_index: usize, centroides_real: &[f32; 4], centroides_imag: &[f32; 4]) {
        let weight = self.get_weight(input_index);
        self.membrane_potential_real += centroides_real[weight as usize];
        self.membrane_potential_imag += centroides_imag[weight as usize];
    }

    /// Verifica si la neurona debe disparar.
    /// Si dispara, el potencial se resetea. Si no, se aplica la fuga (decay).
    pub fn check_spike(&mut self) -> bool {
        let magnitude = (self.membrane_potential_real.powi(2) + self.membrane_potential_imag.powi(2)).sqrt();
        if magnitude >= self.threshold {
            self.membrane_potential_real = 0.0;
            self.membrane_potential_imag = 0.0;
            true
        } else {
            // Aplicar fuga
            if magnitude > 0.0 {
                self.membrane_potential_real *= self.decay;
                self.membrane_potential_imag *= self.decay;
            }
            false
        }
    }

    /// Verifica si la neurona debe disparar y calcula el evento resultante con retraso Lagrangiano.
    /// El retraso en la Rueda de Tiempo es proporcional a la resistencia semántica (Energía Potencial).
    pub fn check_spike_lagrangian(
        &mut self, 
        current_tick: u64,
        source_id: usize,
        semantic_resistance: f32
    ) -> Option<SpikeEvent> {
        let magnitude = (self.membrane_potential_real.powi(2) + self.membrane_potential_imag.powi(2)).sqrt();
        
        if magnitude >= self.threshold {
            // Resetear potencial
            self.membrane_potential_real = 0.0;
            self.membrane_potential_imag = 0.0;

            // Calcular retraso físico basado en la resistencia (Energía Potencial)
            let delay = self.lagrangian.calculate_timing_delay(semantic_resistance);
            
            // Convertir el retraso en ticks y sub-ticks (fase)
            let total_delay_ticks = delay.floor() as u64;
            let phase_delay = ((delay - delay.floor()) * 16.0) as u8;

            Some(SpikeEvent {
                timestamp: current_tick + total_delay_ticks,
                phase_offset: phase_delay.min(15),
                intensity: magnitude,
                source_neuron_id: source_id,
                target_layer_id: 0, // A definir por el llamador
                target_neuron_id: 0, // A definir por el llamador
            })
        } else {
            // Aplicar fuga
            if magnitude > 0.0 {
                self.membrane_potential_real *= self.decay;
                self.membrane_potential_imag *= self.decay;
            }
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuron_integration_and_spike() {
        let mut neuron = SpikingNeuron::new(1.0, 0.9, 4);
        let c_r = [-0.1, 0.2, 0.5, 0.8];
        let c_im = [0.0, 0.0, 0.0, 0.0];

        // Configurar pesos
        neuron.set_weight(0, GajeWeight2Bit::State11); // 0.8
        neuron.set_weight(1, GajeWeight2Bit::State01); // 0.2

        // Integrar primer peso: 0.8 (No debería disparar aún)
        neuron.integrate(0, &c_r, &c_im);
        assert!(!neuron.check_spike());
        
        // Integrar segundo peso
        neuron.integrate(1, &c_r, &c_im);
        assert!(!neuron.check_spike());

        // Forzar disparo
        neuron.set_weight(2, GajeWeight2Bit::State11);
        neuron.integrate(2, &c_r, &c_im);
        assert!(neuron.check_spike());
        assert_eq!(neuron.membrane_potential_real, 0.0);
    }

    #[test]
    fn test_bit_packing() {
        let mut neuron = SpikingNeuron::new(1.0, 0.9, 8);
        
        neuron.set_weight(0, GajeWeight2Bit::State00);
        neuron.set_weight(1, GajeWeight2Bit::State01);
        neuron.set_weight(2, GajeWeight2Bit::State10);
        neuron.set_weight(3, GajeWeight2Bit::State11);

        assert_eq!(neuron.get_weight(0), GajeWeight2Bit::State00);
        assert_eq!(neuron.get_weight(1), GajeWeight2Bit::State01);
        assert_eq!(neuron.get_weight(2), GajeWeight2Bit::State10);
        assert_eq!(neuron.get_weight(3), GajeWeight2Bit::State11);
        
        // Verificar que ocupan solo 1 byte (4 pesos x 2 bits)
        assert_eq!(neuron.weights[0], 0b11100100); 
    }

    #[test]
    fn test_lagrangian_inference() {
        let mut neuron = SpikingNeuron::new(1.0, 0.9, 4);
        let c_r = [0.0, 0.0, 0.0, 2.0]; // Peso fuerte (2.0)
        let c_im = [0.0, 0.0, 0.0, 0.0];

        neuron.set_weight(0, GajeWeight2Bit::State11);

        // Escenario 1: Resistencia Semántica Baja (Coherente)
        neuron.integrate_lagrangian(0, &c_r, &c_im, 0.0);
        let spike = neuron.check_spike_lagrangian(100, 1, 0.0).unwrap();
        assert_eq!(spike.timestamp, 100); // Disparo inmediato
        assert_eq!(spike.phase_offset, 0);

        // Escenario 2: Resistencia Semántica Media (Incoherente)
        neuron.membrane_potential_real = 0.0;
        // Resistencia 0.5 -> Aceleración -0.5. Velocidad efectiva = 2.0 - 0.5 = 1.5.
        // Potencial acumulado = 1.5 (supera el umbral 1.0)
        neuron.integrate_lagrangian(0, &c_r, &c_im, 0.5);
        let spike_delayed = neuron.check_spike_lagrangian(100, 1, 1.5).unwrap();
        
        // Con resistencia 1.5 (en el momento del disparo), el retraso es ln(1+1.5) ≈ 0.91 ticks.
        // Tick 100 + 0 = 100. Fase = 0.91 * 16 ≈ 14.
        assert_eq!(spike_delayed.timestamp, 100);
        assert!(spike_delayed.phase_offset > 0);
        
        // Escenario 3: Resistencia Semántica Muy Alta (Bloqueo)
        neuron.membrane_potential_real = 0.0;
        neuron.integrate_lagrangian(0, &c_r, &c_im, 2.5); // Supera el peso de 2.0
        let no_spike = neuron.check_spike_lagrangian(100, 1, 2.5);
        assert!(no_spike.is_none());
    }
}
