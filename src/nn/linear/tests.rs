// =============================================================================
// tests — Tests unitarios de GenomicLinear
// =============================================================================
use super::*;
use crate::nn::linear::database::WeightDatabase;

#[test]
fn test_lm_head_fp32_update() {
    // lm_head simulado FP32: out=3, in=4, pesos row-major (3x4)
    let w: Vec<f32> = (0..12).map(|i| i as f32 * 0.5).collect();
    let raw: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();

    let mut linear = GenomicLinear::new(
        raw,
        Vec::new(), // anchors
        Vec::new(), // centroids (vacío -> no genómico)
        3,          // out_features
        4,          // in_features
        4,          // block_size
        Vec::new(),
        1e-6,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(), // bias
        32,         // bit_depth -> GenomicF32
    );
    assert!(matches!(linear.weight_db, WeightDatabase::GenomicF32(_)));

    let w_before: Vec<f32> = match &linear.weight_db {
        WeightDatabase::GenomicF32(db) => db.as_ref().clone(),
        _ => unreachable!(),
    };

    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let out_before = linear.forward_core(input.clone(), None, false).unwrap();

    // grad_ce para target=0 => probs - one_hot ≈ [-0.75, 0.25, 0.0] (tamaño=out)
    let grads = vec![-0.75f32, 0.25, 0.0];
    linear.refine_with_grads_core(input, grads, 1e-2).unwrap();

    let w_after: Vec<f32> = match &linear.weight_db {
        WeightDatabase::GenomicF32(db) => db.as_ref().clone(),
        _ => unreachable!(),
    };
    let max_delta = w_before
        .iter()
        .zip(&w_after)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_delta > 0.0,
        "FP32 weights must change after refine (path was a silent no-op)"
    );

    let out_after = linear.forward_core(vec![1.0f32, 2.0, 3.0, 4.0], None, false).unwrap();
    assert!(
        (out_after[0] - out_before[0]).abs() > 0.0,
        "logit 0 must change after update"
    );
}

#[test]
fn test_q4_0_scale_min_update() {
    // Cuerpo Q4_0: out=2, in=32 -> 1 bloque por fila, 2 bloques en total.
    // W[i,j] = q*scale + min; q fijo. Solo scale/min deben recalibrarse (IQAT).
    use crate::io::header::Q4_0Block;
    let mk_block = |q: u8| Q4_0Block {
        scale: half::f16::from_f32(0.1),
        min: half::f16::from_f32(-0.5),
        qs: [q | (q << 4); 16],
    };
    let blocks = vec![mk_block(5), mk_block(9)];
    let raw_bytes: Vec<u8> = unsafe {
        std::slice::from_raw_parts(
            blocks.as_ptr() as *const u8,
            blocks.len() * std::mem::size_of::<Q4_0Block>(),
        )
        .to_vec()
    };

    let mut linear = GenomicLinear::new(
        raw_bytes,
        Vec::new(),
        Vec::new(), // centroids vacíos -> GenomicQ4_0
        2,
        32,
        32,
        Vec::new(),
        1e-6,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        4,
    );
    assert!(matches!(linear.weight_db, WeightDatabase::GenomicQ4_0(_)));

    let (s0, m0) = match &linear.weight_db {
        WeightDatabase::GenomicQ4_0(db) => (db[0].scale.to_f32(), db[0].min.to_f32()),
        _ => unreachable!(),
    };

    let input = vec![1.0f32; 32];
    let out_before = linear.forward_core(input.clone(), None, false).unwrap();

    let grads = vec![1.0f32, -1.0];
    linear.refine_with_grads_core(input, grads, 1e-2).unwrap();

    let (s1, m1) = match &linear.weight_db {
        WeightDatabase::GenomicQ4_0(db) => (db[0].scale.to_f32(), db[0].min.to_f32()),
        _ => unreachable!(),
    };
    assert!(
        (s1 - s0).abs() > 0.0 || (m1 - m0).abs() > 0.0,
        "Q4_0 scale/min must change after refine (was a no-op)"
    );

    let out_after = linear.forward_core(vec![1.0f32; 32], None, false).unwrap();
    assert!(
        (out_after[0] - out_before[0]).abs() > 0.0,
        "Q4_0 output must change after scale/min update"
    );
}

#[test]
fn test_q8_0_scale_update() {
    // Cuerpo Q8_0: out=1, in=32 -> 1 bloque. W = q8*scale; solo scale cambia.
    use crate::io::header::Q8_0Block;
    let q8 = Q8_0Block {
        scale: half::f16::from_f32(0.1),
        qs: [5i8; 32],
    };
    let raw_bytes: Vec<u8> = unsafe {
        std::slice::from_raw_parts(
            &q8 as *const Q8_0Block as *const u8,
            std::mem::size_of::<Q8_0Block>(),
        )
        .to_vec()
    };

    let mut linear = GenomicLinear::new(
        raw_bytes,
        Vec::new(),
        Vec::new(), // centroids vacíos -> GenomicQ8_0
        1,
        32,
        32,
        Vec::new(),
        1e-6,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        8,
    );
    assert!(matches!(linear.weight_db, WeightDatabase::GenomicQ8_0(_)));

    let s0 = match &linear.weight_db {
        WeightDatabase::GenomicQ8_0(db) => db[0].scale.to_f32(),
        _ => unreachable!(),
    };

    let input = vec![1.0f32; 32];
    let out_before = linear.forward_core(input.clone(), None, false).unwrap();

    linear.refine_with_grads_core(input, vec![1.0f32], 1e-2).unwrap();

    let s1 = match &linear.weight_db {
        WeightDatabase::GenomicQ8_0(db) => db[0].scale.to_f32(),
        _ => unreachable!(),
    };
    assert!(
        (s1 - s0).abs() > 0.0,
        "Q8_0 scale must change after refine (was a no-op)"
    );

    let out_after = linear.forward_core(vec![1.0f32; 32], None, false).unwrap();
    assert!(
        (out_after[0] - out_before[0]).abs() > 0.0,
        "Q8_0 output must change after scale update"
    );
}

#[test]
fn test_backward_transpose_q4_0() {
    // backward_core debe devolver d_input = W^T · d_output.
    // W[i,j] = q*scale+min con layout row-major: fila i = out, col j = in.
    use crate::io::header::Q4_0Block;
    let mk_block = |q: u8| Q4_0Block {
        scale: half::f16::from_f32(0.5),
        min: half::f16::from_f32(0.1),
        qs: [q | (q << 4); 16],
    };
    // out=2, in=32 -> fila 0 q=2, fila 1 q=4. W[0,j]=2*0.5+0.1=1.1; W[1,j]=4*0.5+0.1=2.1
    let blocks = vec![mk_block(2), mk_block(4)];
    let raw_bytes: Vec<u8> = unsafe {
        std::slice::from_raw_parts(
            blocks.as_ptr() as *const u8,
            blocks.len() * std::mem::size_of::<Q4_0Block>(),
        )
        .to_vec()
    };
    let linear = GenomicLinear::new(
        raw_bytes, Vec::new(), Vec::new(), 2, 32, 32, Vec::new(), 1e-6,
        Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), 4,
    );

    let d_output = vec![2.0f32, 3.0];
    let d_input = linear.backward_core(d_output).unwrap();
    // d_input[j] = 2*1.1 + 3*2.1 = 2.2 + 6.3 = 8.5 para todo j
    assert_eq!(d_input.len(), 32);
    for v in &d_input {
        assert!((v - 8.5).abs() < 1e-3, "expected 8.5, got {}", v);
    }
}

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
