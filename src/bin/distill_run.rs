use _impl::nn::distiller::{CouncilOfTeachers, Teacher, GenomicDistiller};
use _impl::io::flat_reader::GajeFlatFileReader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = if std::path::Path::new("models/born/max_512_pro.gaje").exists() {
        "models/born/max_512_pro.gaje"
    } else {
        "models/born/max_512.gaje"
    };

    println!("📦 Cargando modelo alumno para destilación optimizada: {}", model_path);
    let (mut student, tok) = _impl::nn::repl::load_model_and_tokenizer(model_path)?;
    let reader = GajeFlatFileReader::open(model_path)?;
    let config = reader.load_config()?;

    println!("🧠 Cargando modelo maestro 3B...");
    let (teacher_model, teacher_tok) = _impl::nn::repl::load_model_and_tokenizer("models/production/gaje_pro_3b.flat")?;
    let mut teacher = Teacher {
        name: "pro3b".to_string(),
        model: teacher_model,
        tokenizer: teacher_tok.clone(),
        vocab_mapping: vec![],
        is_identity_vocab: true,
    };
    teacher.vocab_mapping = (0..teacher_tok.vocab_size()).map(Some).collect();

    let mut council = CouncilOfTeachers::new();
    council.add_teacher(teacher);
    let distiller = GenomicDistiller::new(council, tok.clone());

    let dataset_path = "data/curated_150_distill.jsonl";
    let data = std::fs::read_to_string(dataset_path)?;
    let mut texts = Vec::new();
    for line in data.lines() {
        if line.trim().is_empty() { continue; }
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            if let Some(t) = v.get("text").and_then(|x| x.as_str()) {
                texts.push(t.to_string());
            } else {
                texts.push(line.to_string());
            }
        } else {
            texts.push(line.to_string());
        }
    }

    let total_epochs = 15;
    let n_texts = texts.len();
    println!("🚀 Iniciando Ciclo Optimizado de Destilación DNI:");
    println!("   • Alumno:         max_512 (Dimensión D=512, 12 bloques, 208 MB)");
    println!("   • Maestro:        gaje_pro_3b (36 bloques, 2048 dim, 4.0 GB)");
    println!("   • Dataset:        {} pares curados ({})", n_texts, dataset_path);
    println!("   • Épocas Totales: {}", total_epochs);
    println!("   • LR Schedule:    Cosine Decay (0.0030 -> 0.0003)\n");

    let lr_max = 0.0030f32;
    let lr_min = 0.0003f32;

    for epoch in 0..total_epochs {
        let progress = epoch as f32 / total_epochs as f32;
        let lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + (std::f32::consts::PI * progress).cos());
        println!("🔥 Época {:02}/{} | LR: {:.6}", epoch + 1, total_epochs, lr);

        let mut epoch_loss = 0.0f32;
        let mut count = 0;

        for (i, txt) in texts.iter().enumerate() {
            match distiller.distill_step_online_gpu(&mut student, txt, lr, 1.0) {
                Ok(loss) => {
                    epoch_loss += loss;
                    count += 1;
                    if (i + 1) % 15 == 0 || (i + 1) == n_texts {
                        println!("   [{:03}/{:03}] Pérdida actual: {:.4}", i + 1, n_texts, loss);
                    }
                }
                Err(e) => {
                    eprintln!("   ⚠️ Error en paso {}: {}", i + 1, e);
                }
            }
        }

        let avg_loss = if count > 0 { epoch_loss / count as f32 } else { 0.0 };
        println!("   ✨ Resumen Época {:02}: Pérdida promedio = {:.4}\n", epoch + 1, avg_loss);

        // Guardado intermedio cada 5 épocas
        if (epoch + 1) % 5 == 0 {
            let out_pro = "models/born/max_512_pro.gaje";
            let out_base = "models/born/max_512.gaje";
            _impl::io::flat_writer::save_genomic_flat_q(out_pro, &student, &config, Some(&tok), 2)?;
            _impl::io::flat_writer::save_genomic_flat_q(out_base, &student, &config, Some(&tok), 2)?;
            println!("💾 Checkpoint guardado en {} y {}", out_pro, out_base);
        }
    }

    let out_pro = "models/born/max_512_pro.gaje";
    let out_base = "models/born/max_512.gaje";
    _impl::io::flat_writer::save_genomic_flat_q(out_pro, &student, &config, Some(&tok), 2)?;
    _impl::io::flat_writer::save_genomic_flat_q(out_base, &student, &config, Some(&tok), 2)?;
    println!("\n✅ ¡Destilación de 15 épocas completada con éxito!");
    println!("💾 Organismo final listo en: {}", out_pro);

    Ok(())
}
