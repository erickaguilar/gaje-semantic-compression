// =============================================================================
// tests — Pruebas unitarias de la cabecera .flat v2 y bloques de cuantización
// =============================================================================
use super::*;

#[test]
fn test_header_v1_backward_compatibility() {
    let mut header_bytes = [0u8; 4096];
    header_bytes[0..4].copy_from_slice(b"GAJE");
    // version 0x000907
    header_bytes[4..8].copy_from_slice(&0x000907u32.to_le_bytes());
    // flags, num_tensors
    header_bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
    header_bytes[12..16].copy_from_slice(&12u32.to_le_bytes());
    // rest of layout fields are zero

    let header = FlatHeaderV2::from_bytes(&header_bytes).unwrap();
    assert_eq!(header.quantization_type(), QuantFormat::LegacyCentroids);
    assert_eq!(header.effective_group_size(), 16);
    assert_eq!(header.num_tensors, 12);
    assert_eq!(header.version, 0x000907);
}

#[test]
fn test_header_v2_q4_0() {
    let mut header_bytes = [0u8; 4096];
    header_bytes[0..4].copy_from_slice(b"GAJE");
    header_bytes[4..8].copy_from_slice(&0x000908u32.to_le_bytes());
    // group_size = 32
    header_bytes[48..52].copy_from_slice(&32u32.to_le_bytes());
    // quant_format = 1 (Q4_0)
    header_bytes[52..56].copy_from_slice(&1u32.to_le_bytes());

let header = FlatHeaderV2::from_bytes(&header_bytes).unwrap();
        assert_eq!(header.quantization_type(), QuantFormat::Q4_0);
        assert_eq!(header.effective_group_size(), 32);
        assert_eq!(header.version, 0x000908);
}

#[test]
fn test_header_v2_q2_0() {
    let mut header_bytes = [0u8; 4096];
    header_bytes[0..4].copy_from_slice(b"GAJE");
    header_bytes[4..8].copy_from_slice(&0x000908u32.to_le_bytes());
    // group_size = 32
    header_bytes[48..52].copy_from_slice(&32u32.to_le_bytes());
    // quant_format = 3 (Q2_0)
    header_bytes[52..56].copy_from_slice(&3u32.to_le_bytes());

    let header = FlatHeaderV2::from_bytes(&header_bytes).unwrap();
    assert_eq!(header.quantization_type(), QuantFormat::Q2_0);
    assert_eq!(header.effective_group_size(), 32);
}

#[test]
fn test_invalid_magic_bytes() {
    let mut header_bytes = [0u8; 4096];
    header_bytes[0..4].copy_from_slice(b"XGAJ");

    let res = FlatHeaderV2::from_bytes(&header_bytes);
    assert!(res.is_err());
    assert!(matches!(res.err().unwrap(), HeaderError::InvalidMagic));
}

#[test]
fn test_q4_0_block_roundtrip() {
    let f32_weights: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect(); // 0.0 to 3.1
    let min_val = 0.0f32;
    let max_val = 3.1f32;
    let scale = (max_val - min_val) / 15.0;
    let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

    let mut qs = [0u8; 16];
    for k in 0..16 {
        let q0 = (((f32_weights[k * 2] - min_val) * inv_scale)
            .round()
            .clamp(0.0, 15.0)) as u8;
        let q1 = (((f32_weights[k * 2 + 1] - min_val) * inv_scale)
            .round()
            .clamp(0.0, 15.0)) as u8;
        qs[k] = q0 | (q1 << 4);
    }

    let block = Q4_0Block {
        scale: half::f16::from_f32(scale),
        min: half::f16::from_f32(min_val),
        qs,
    };

    // Dequantize and check error bounds
    for i in 0..32 {
        let original = f32_weights[i];
        let dequantized = block.dequantize_weight(i);
        let err = (original - dequantized).abs();
        assert!(
            err <= 0.11,
            "Original {} vs dequantized {} error {} above step/2 limit",
            original,
            dequantized,
            err
        );
    }
}

#[test]
fn test_actual_q4_0_model_loading() {
    let model_path = "models/production/qwen2_0_5b_q4_0.gaje.flat";
    if std::path::Path::new(model_path).exists() {
        use crate::nn::linear::WeightDatabase;
        let reader = crate::io::loader::GajeFlatFileReader::open(model_path).unwrap();
        assert_eq!(reader.header.quantization_type(), QuantFormat::Q4_0);
        assert_eq!(reader.header.effective_group_size(), 32);

        // Load a linear layer blk.0.attn_output
        let linear = reader.get_linear("blk.0.attn_output", 32).unwrap();
        assert_eq!(linear.bit_depth(), 4);
        assert!(matches!(linear.weight_db, WeightDatabase::GenomicQ4_0(_)));

        // Dequantize row 0
        let row0 = linear.get_row_core(0).unwrap();
        assert_eq!(row0.len(), linear.in_features);

        // Verify that dequantized weights are valid (not NaN, not all 0)
        let mut all_zero = true;
        for &w in &row0 {
            assert!(!w.is_nan());
            if w != 0.0 {
                all_zero = false;
            }
        }
        assert!(!all_zero, "Dequantized weights should not be all zeros");
    }
}

#[test]
fn test_q2_0_block_roundtrip() {
    let f32_weights: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect(); // 0.0 to 3.1
    let min_val = 0.0f32;
    let max_val = 3.1f32;
    let scale = (max_val - min_val) / 3.0; // 4 niveles (2 bits)
    let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

    let mut qs = [0u8; 8];
    for k in 0..8 {
        let mut byte = 0u8;
        for j in 0..4 {
            let q = (((f32_weights[k * 4 + j] - min_val) * inv_scale)
                .round()
                .clamp(0.0, 3.0)) as u8;
            byte |= q << (j * 2);
        }
        qs[k] = byte;
    }

    let block = Q2_0Block {
        scale: half::f16::from_f32(scale),
        min: half::f16::from_f32(min_val),
        qs,
    };

    // Verifica que cada q_value devuelve el código de 2 bits correcto
    for i in 0..32 {
        let expected = (((f32_weights[i] - min_val) * inv_scale).round().clamp(0.0, 3.0)) as u8;
        assert_eq!(block.q_value(i), expected, "q_value idx {} mismatch", i);
    }

    // Dequantize y chequeo de error <= paso/2 (paso = scale)
    for i in 0..32 {
        let original = f32_weights[i];
        let dequantized = block.dequantize_weight(i);
        let err = (original - dequantized).abs();
        assert!(
            err <= scale / 2.0 + 1e-4,
            "Original {} vs dequantized {} error {} above step/2 limit",
            original,
            dequantized,
            err
        );
    }
}

#[test]
fn test_q8_0_block_quantize_dequantize() {
    let mut f32_weights = [0.0f32; 32];
    for i in 0..32 {
        f32_weights[i] = (i as f32) * 0.1 - 1.6;
    }

    let mut max_abs = 0.0f32;
    for &v in &f32_weights {
        let abs_v = v.abs();
        if abs_v > max_abs {
            max_abs = abs_v;
        }
    }

    let scale = max_abs / 127.0;
    let inv_scale = if scale > 1e-7 { 1.0 / scale } else { 0.0 };

    let mut qs = [0i8; 32];
    for k in 0..32 {
        qs[k] = if scale > 1e-7 {
            (f32_weights[k] * inv_scale).round().clamp(-128.0, 127.0) as i8
        } else {
            0
        };
    }

    let block = Q8_0Block {
        scale: half::f16::from_f32(scale),
        qs,
    };

    // Dequantize and check error bounds
    for i in 0..32 {
        let original = f32_weights[i];
        let dequantized = block.dequantize_weight(i);
        let err = (original - dequantized).abs();
        assert!(
            err <= scale / 2.0 + 1e-5,
            "Original {} vs dequantized {} error {} above step/2 limit",
            original,
            dequantized,
            err
        );
    }
}
