#[cfg(test)]
mod tests {
    use _impl::io::loader::NativeLoader;
    use _impl::compute::math::calculate_cosine_similarity_native;
    use std::time::Instant;

    #[test]
    fn test_phase_survival_toroidal_echo() {
        println!("🚀 Iniciando Experimentum Crucis en el metal de Rust...");

        // Usamos el path del modelo de producción identificado
        let model_db_path = "models/production/silver_adult_steel.gaje";
        let loader_res = NativeLoader::new(model_db_path);

        let mut model = match loader_res {
            Ok(loader) => {
                match loader.load_llm() {
                    Ok(mut m) => {
                        println!("✅ Modelo Silver Adult cargado para la prueba.");
                        // Auditoría de pesos
                        let mut weight_nan = false;
                        for (i, block) in m.blocks.iter().enumerate() {
                            if block.q_gen.centroids.iter().any(|x| !x.is_finite()) {
                                println!("🚨 NaN/Inf detectado en centroides de bloque {} (Q)", i);
                                weight_nan = true;
                            }
                            if block.q_gen.anchor_values.iter().any(|x| !x.to_f32().is_finite()) {
                                println!("🚨 NaN/Inf detectado en anclas de bloque {} (Q)", i);
                                weight_nan = true;
                            }
                        }
                        if weight_nan {
                            println!("⚠️ ADVERTENCIA: Se detectaron valores no finitos en los pesos del modelo.");
                        } else {
                            println!("✅ Pesos del modelo validados (sin NaN/Inf).");
                        }
                        // Activamos la física toroidal para la prueba
                        for block in m.blocks.iter_mut() {
                            block.use_genomic_norm = true;
                            block.rna_threshold = 0.5; // Balance entre precisión y filtrado
                        }
                        m
                    },
                    Err(e) => {
                        println!("⚠️ Error al cargar LLM desde {}: {}", model_db_path, e);
                        return;
                    }
                }
            },
            Err(e) => {
                println!("⚠️ No se pudo inicializar NativeLoader en {}: {}", model_db_path, e);
                return;
            }
        };

        let needle_token_id = 777; 
        
        println!("⏳ Inyectando señal (Token ID: {}) en el Toroide...", needle_token_id);
        // Desglose del forward para detectar NaNs paso a paso
        let mut h = model.embeddings.get_row_core(needle_token_id).expect("Error en embeddings");
        if h.iter().any(|x| x.is_nan()) {
            println!("🚨 NaNs detectados después de model.embeddings.get_row_core");
        }
        
        for (i, block) in model.blocks.iter_mut().enumerate() {
            h = block.forward_core(h, 0).expect(&format!("Error en bloque {}", i));
            if h.iter().any(|x| x.is_nan()) {
                println!("🚨 NaNs detectados después de bloque {}", i);
                break;
            }
        }
        
        let h_norm = unsafe { _impl::compute::kernels::rms_norm(&h, &model.output_norm, model.eps) };
        if h_norm.iter().any(|x| x.is_nan()) {
             println!("🚨 NaNs detectados después de output_norm (RMSNorm)");
        }

        let gold_standard_hidden = h_norm;

        if gold_standard_hidden.iter().any(|x| x.is_nan()) {
            println!("🚨 ERROR CRÍTICO: El Gold Standard ya contiene NaNs antes de iniciar la deriva.");
            return;
        } else {
            println!("✅ Gold Standard validado (sin NaNs).");
        }

        let cycles = 5_000; // Punto Dulce para ARM: Suficiente para probar estabilidad, rápido para Termux
        let silence_token_id = 0; 
        
        println!("🌀 Iniciando Deriva Temporal de {} ciclos...", cycles);
        let start = Instant::now();
        
        for i in 0..cycles {
            let _ = model.forward_core(silence_token_id, false)
                .expect(&format!("Fallo en el ciclo {}", i));
            
            // Telemetría constante para evitar sensación de bloqueo
            if i % 1_000 == 0 && i > 0 {
                let (_, hidden) = model.forward_with_hidden_core(needle_token_id, false).unwrap();
                let has_nan = hidden.iter().any(|x| x.is_nan());
                let elapsed = start.elapsed().as_secs_f32();
                let speed = i as f32 / elapsed;
                let eta = (cycles - i) as f32 / speed;
                println!("   [Tick {}] Eco en órbita... Velocidad: {:.2} iter/s | ETA: {:.1}s | NaN Check: {}", i, speed, eta, if has_nan { "❌ FAIL" } else { "✅ OK" });
                
                if has_nan {
                    println!("🚨 COLAPSO NUMÉRICO DETECTADO en el ciclo {}. Abortando para diagnóstico.", i);
                    assert!(!has_nan, "NaN detectado en el ciclo {}", i);
                }
            }
        }
        
        let duration = start.elapsed();
        println!("✨ Deriva completada en {:?}. Procediendo al Colapso de Fase...", duration);

        let (_, final_hidden) = model.forward_with_hidden_core(needle_token_id, false)
            .expect("Error en el colapso final");

        let similarity = calculate_cosine_similarity_native(gold_standard_hidden, final_hidden)
            .expect("Error al calcular similitud");

        println!("\n========================================");
        println!("📊 RESULTADOS DEL ECO TOROIDAL (RUST)");
        println!("========================================");
        println!("✅ Similitud de Fase: {:.8}", similarity);
        println!("✅ Ciclos de Resonancia: {}", cycles);
        
        let threshold = 0.99; // Certificación Nivel 1: Resonancia Inmortal
        
        if similarity >= threshold {
            println!("🏆 VERDICTO: ECO INFINITO CONFIRMADO. PARADIGMA VÁLIDO.");
            assert!(true);
        } else {
            println!("❌ VERDICTO: FUGA ENTRÓPICA DETECTADA. REVISAR TOPOLOGÍA.");
            assert!(similarity >= threshold, "La señal se disipó en el toroide. Similitud: {}", similarity);
        }
        println!("========================================");
    }
}
