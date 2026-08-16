// =============================================================================
// integracion — Smoke end-to-end: el cuerpo Q4_0/Q8_0 se entrena de verdad
// =============================================================================
use crate::io::flat_reader::load_genomic_auto;

#[test]
fn test_body_refine_changes_weights_on_real_model() {
    let path = "models/production/smollm2_4bit.gaje.flat";
    if !std::path::Path::new(path).exists() {
        eprintln!("skip: modelo no presente en {}", path);
        return;
    }
    let mut model = load_genomic_auto(path).expect("cargar modelo");
    assert!(!model.blocks.is_empty(), "el modelo debe tener bloques");

    eprintln!(
        "variants: gate={:?} up={:?} down={:?} wo={:?} q={:?} k={:?} v={:?}",
        std::mem::discriminant(&model.blocks[0].gate_gen.weight_db),
        std::mem::discriminant(&model.blocks[0].up_gen.weight_db),
        std::mem::discriminant(&model.blocks[0].w_down.weight_db),
        std::mem::discriminant(&model.blocks[0].w_o.weight_db),
        std::mem::discriminant(&model.blocks[0].q_gen.weight_db),
        std::mem::discriminant(&model.blocks[0].k_gen.weight_db),
        std::mem::discriminant(&model.blocks[0].v_gen.weight_db),
    );

    // Snapshots de los centroides de los linears del cuerpo antes de refinar.
    // El cuerpo se guarda como Genomic4Bit (basado en centroides); el update de
    // refine_with_grads_core muta self.centroids (no los bytes del database).
    let before_gate = model.blocks[0].gate_gen.centroids.clone();
    let before_down = model.blocks[0].w_down.centroids.clone();
    let before_wo = model.blocks[0].w_o.centroids.clone();

    let input_tokens = [4usize, 5, 6];
    for &tok in &input_tokens {
        let x = model.embeddings.get_row_core(tok).unwrap();
        let out = model.blocks[0].forward_core(x.clone(), 0).unwrap();
        let grads: Vec<f32> = out.iter().map(|&v| (v * 0.01).sin()).collect();
        model.blocks[0]
            .refine_with_grads_core(x, grads, 0, 1e-3)
            .expect("refine bloque");
    }

    let changed = model.blocks[0].gate_gen.centroids != before_gate
        || model.blocks[0].w_down.centroids != before_down
        || model.blocks[0].w_o.centroids != before_wo;
    assert!(changed, "el cuerpo Genomic4Bit debió mutar sus centroides tras refine");
}