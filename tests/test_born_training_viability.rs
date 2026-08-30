use _impl::nn::llm::birth::{create_born_organism, BornConfig};

#[test]
fn test_born_q2_0_training_viability_and_loss_decrease() {
    println!("\n🧬 ==================================================================");
    println!("🧪 TEST DE VIABILIDAD: Bucle de Entrenamiento Mínimo para Born 2-Bit (max.gaje)");
    println!("==================================================================");

    let config = BornConfig {
        name: "max_test".to_string(),
        vocab_size: 256,
        dim: 64,
        n_layers: 4,
        n_heads: 2,
        intermediate_dim: 128,
        eps: 1e-6,
        k_wta_ratio: 0.15,
    };

    let mut model = create_born_organism(config);

    // Secuencia de entrenamiento sintética (patrón recurrente de tokens)
    let sequence = vec![
        10usize, 20, 30, 40, 50, 60, 70, 80,
        10, 20, 30, 40, 50, 60, 70, 80,
        10, 20, 30, 40, 50, 60, 70, 80,
    ];

    let lr = 0.01f32;
    let epochs = 15;
    let mut initial_loss = 0.0f32;
    let mut final_loss = 0.0f32;

    println!("📊 Ejecutando {} épocas de entrenamiento con STE Cuaternario...", epochs);

    for epoch in 0..epochs {
        let loss = model
            .train_sequence_cached_layerwise_core(
                sequence.clone(),
                lr,
                4,    // entrenar los 4 bloques
                1.0,  // gradient clipping
                0.95, // layer-wise decay
                true, // entrenar lm_head
                None,
            )
            .expect("El paso de entrenamiento debió ejecutarse sin error");

        assert!(loss.is_finite(), "La pérdida debe ser finita, no NaN");

        if epoch == 0 {
            initial_loss = loss;
        }
        final_loss = loss;

        println!("  • Época {:>2}/{}: Loss = {:.4}", epoch + 1, epochs, loss);
    }

    println!("\n📈 Resumen de Convergencia:");
    println!("  • Loss Inicial : {:.4}", initial_loss);
    println!("  • Loss Final   : {:.4}", final_loss);
    let delta = initial_loss - final_loss;
    let reduction_pct = (delta / initial_loss) * 100.0;
    println!("  • Reducción    : {:.4} ({:.2}%)", delta, reduction_pct);

    assert!(final_loss < initial_loss, "La pérdida debe decrecer tras el entrenamiento");
    assert!(reduction_pct > 15.0, "La reducción de pérdida debe ser superior al 15%");
    println!("✅ VIABILIDAD EMPÍRICA CERTIFICADA: El organismo nacido aprende con STE Cuaternario");
    println!("==================================================================\n");
}
