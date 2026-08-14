// =============================================================================
// tests — Tests unitarios de GenomicLinear
// =============================================================================
use super::*;
use crate::nn::linear::database::WeightDatabase;

#[test]
fn test_q4_0_linear_forward_roundtrip() {
    // Generate a 2x32 weight matrix
    // Row 0: 0.0, 0.1, ..., 3.1
    // Row 1: 0.0, -0.1, ..., -3.1
    let row0: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
    let row1: Vec<f32> = (0..32).map(|i| i as f32 * -0.1).collect();

    let mut blocks = Vec::new();

    // Row 0 quantization
    let min0 = 0.0f32;
    let max0 = 3.1f32;
    let scale0 = (max0 - min0) / 15.0;
    let inv_scale0 = 1.0 / scale0;
    let mut qs0 = [0u8; 16];
    for k in 0..16 {
        let q0 = (((row0[k * 2] - min0) * inv_scale0).round().clamp(0.0, 15.0)) as u8;
        let q1 = (((row0[k * 2 + 1] - min0) * inv_scale0)
            .round()
            .clamp(0.0, 15.0)) as u8;
        qs0[k] = q0 | (q1 << 4);
    }
    blocks.push(crate::io::header::Q4_0Block {
        scale: half::f16::from_f32(scale0),
        min: half::f16::from_f32(min0),
        qs: qs0,
    });

    // Row 1 quantization
    let min1 = -3.1f32;
    let max1 = 0.0f32;
    let scale1 = (max1 - min1) / 15.0;
    let inv_scale1 = 1.0 / scale1;
    let mut qs1 = [0u8; 16];
    for k in 0..16 {
        let q0 = (((row1[k * 2] - min1) * inv_scale1).round().clamp(0.0, 15.0)) as u8;
        let q1 = (((row1[k * 2 + 1] - min1) * inv_scale1)
            .round()
            .clamp(0.0, 15.0)) as u8;
        qs1[k] = q0 | (q1 << 4);
    }
    blocks.push(crate::io::header::Q4_0Block {
        scale: half::f16::from_f32(scale1),
        min: half::f16::from_f32(min1),
        qs: qs1,
    });

    // Convert blocks to byte vector
    let raw_bytes = unsafe {
        std::slice::from_raw_parts(
            blocks.as_ptr() as *const u8,
            blocks.len() * std::mem::size_of::<crate::io::header::Q4_0Block>(),
        )
        .to_vec()
    };

    // Instantiate GenomicLinear with bit_depth=4 and empty centroids to trigger Q4_0 variant detection
    let linear = GenomicLinear::new(
        raw_bytes,
        Vec::new(), // anchors
        Vec::new(), // centroids (empty!)
        2,          // out_features
        32,         // in_features
        32,         // block_size
        Vec::new(), // rmsnorm
        1e-6,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(), // bias
        4,          // bit_depth
    );

    assert!(matches!(linear.weight_db, WeightDatabase::GenomicQ4_0(_)));

    // Create input activation: all 1.0s
    let input = vec![1.0f32; 32];
    let output = linear.forward_core(input, None, false).unwrap();

    assert_eq!(output.len(), 2);

    // Verify output Row 0 (sum of row0 weights)
    let dequant0: Vec<f32> = (0..32).map(|i| blocks[0].dequantize_weight(i)).collect();
    let expected0: f32 = dequant0.iter().sum();
    assert!((output[0] - expected0).abs() < 1e-4);

    // Verify output Row 1
    let dequant1: Vec<f32> = (0..32).map(|i| blocks[1].dequantize_weight(i)).collect();
    let expected1: f32 = dequant1.iter().sum();
    assert!((output[1] - expected1).abs() < 1e-4);
}