// =============================================================================
// tests — Tests unitarios de GajeNeuromorphicLayer
// =============================================================================
use crate::nn::spiking::layer::GajeNeuromorphicLayer;
use crate::nn::spiking::neuron::GajeWeight2Bit;

#[test]
fn test_refine_step() {
    let mut layer = GajeNeuromorphicLayer::new(4, 1, 1.0, 0.9);
    // Inicializar pesos a 0 para determinismo
    layer.packed_weights.fill(0);

    let deltas = vec![1.0, 0.0, 1.0, 0.0];
    layer.refine_step(0, deltas, 1.0);
    assert_eq!(layer.packed_weights[0] & 0x03, 1);
    assert_eq!((layer.packed_weights[0] >> 4) & 0x03, 1);
    let deltas_neg = vec![-1.0, 0.0, 0.0, 0.0];
    layer.refine_step(0, deltas_neg, 1.0);
    assert_eq!(layer.packed_weights[0] & 0x03, 0);
}

#[test]
fn test_soa_integration() {
    let c_r = [0.0, -0.2, 0.2, 1.0]; // State 00 es neutral
    let c_im = [0.0, 0.0, 0.0, 0.0];
    let mut layer = GajeNeuromorphicLayer::new(10, 5, 0.5, 0.9);
    // Inicializar pesos a 0 para determinismo
    layer.packed_weights.fill(0);

    layer.set_weight(2, 0, GajeWeight2Bit::State11 as u8);
    layer.integrate_batch(0, c_r, c_im, 2.0);
    assert!(layer.membrane_potentials_real[2] > 2.0);
    let spikes = layer.check_spikes();
    assert_eq!(spikes.len(), 1);
    assert_eq!(spikes[0].0, 2);
    assert!(spikes[0].1 > 4.0);
    assert_eq!(spikes[0].2, 0);
    assert_eq!(layer.membrane_potentials_real[2], 0.0);
}

#[test]
fn test_soa_lagrangian_integration() {
    let c_r = [0.0, 0.0, 0.0, 2.0];
    let c_im = [0.0, 0.0, 0.0, 0.0];
    let mut layer = GajeNeuromorphicLayer::new(10, 5, 0.5, 0.9);
    layer.packed_weights.fill(0);
    layer.set_weight(5, 0, GajeWeight2Bit::State11 as u8);

    // Sin resistencia
    layer.integrate_batch_lagrangian(0, c_r, c_im, 1.0, 0.0);
    assert!(layer.membrane_potentials_real[5] > 1.9); // 2.0 - epsilon

    // Con resistencia que bloquea
    layer.reset_potentials();
    layer.integrate_batch_lagrangian(0, c_r, c_im, 1.0, 2.1);
    assert_eq!(layer.membrane_potentials_real[5], 0.0);
}
