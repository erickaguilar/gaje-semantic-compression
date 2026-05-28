use _impl::nn::distiller::{CouncilOfTeachers, Teacher, GenomicDistiller};
use _impl::core::tokenizer::GajeTokenizer;
use _impl::io::loader::NativeLoader;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::fs;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 INICIANDO MICRO-DESTILACIÓN DE ALTA FIDELIDAD 🧪");
    println!("--------------------------------------------------");

    // Manejador de interrupción (Graceful Shutdown)
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        println!("\n[!] Interrupción detectada (Ctrl+C). Finalizando de forma segura...");
        r.store(false, Ordering::SeqCst);
    }).expect("Error configurando el manejador de señales");

    // 1. Configuración de Rutas
    let student_path = "models/micro_circular_test.gaje";
    let teacher_model_path = "models/gguf/smollm2-135m-f16.gguf";
    let tokenizer_path = "models/core/tokenizer.json";
    let dataset_path = "data/datasets/micro_test.txt"; // Usaremos este como base pero más repetido o uno mayor
    let output_path = "models/micro_distilled_coherence.gaje";

    // 2. Cargar Recursos
    println!("[*] Cargando recursos y tokenizador...");
    let tokenizer = Arc::new(GajeTokenizer::from_file(tokenizer_path)?);
    
    // Preparar dataset (más robusto)
    let raw_text = fs::read_to_string(dataset_path)?;
    let samples: Vec<String> = raw_text.lines()
        .map(|s| s.trim().to_string())
        .filter(|s| s.len() > 10)
        .collect();
    
    // Duplicar samples para simular un dataset mayor si es necesario
    let mut dataset = Vec::new();
    for _ in 0..5 { dataset.extend(samples.clone()); }
    println!("📊 Dataset preparado: {} muestras.", dataset.len());

    // 3. Cargar Estudiante
    println!("[*] Cargando Estudiante (Micro-Organismo)...");
    let loader = NativeLoader::new(student_path)?;
    let mut student = loader.load_llm()?;
    let config = loader.load_config()?;

    // 4. Configurar Consejo de Profesores
    println!("[*] Configurando Profesor (SmolLM2)...");
    let mut council = CouncilOfTeachers::new();
    let teacher = Teacher::new(
        "SmolLM2-Master".to_string(),
        teacher_model_path,
        tokenizer_path,
        &tokenizer
    )?;
    council.add_teacher(teacher);

    // 5. Configurar Destilador
    let distiller = GenomicDistiller::new(council, (*tokenizer).clone());
    let epochs = 20;
    let lr = 0.001;

    println!("\n🚀 Iniciando Ciclo de Destilación ({} épocas)...", epochs);
    
    'training_loop: for epoch in 0..epochs {
        let start = Instant::now();
        let mut epoch_loss = 0.0;
        let mut count = 0;

        for text in &dataset {
            if !running.load(Ordering::SeqCst) {
                println!("    [!] Abortando entrenamiento por interrupción de usuario.");
                break 'training_loop;
            }

            match distiller.distill_step(&mut student, text, lr) {
                Ok(loss) => {
                    epoch_loss += loss;
                    count += 1;
                    
                    // Intra-epoch save cada 50 muestras
                    if count % 50 == 0 {
                        let mut current_config = config.clone();
                        current_config.config.state = format!("distilling-epoch-{}-step-{}", epoch + 1, count);
                        _impl::io::loader::save_genomic_model(output_path, &student, &current_config, Some(&tokenizer))?;
                        println!("      [Intra-Epoch Checkpoint] {} muestras | Loss: {:.4} | State: {}", count, loss, current_config.config.state);
                    }
                },
                Err(e) => println!("    [!] Error en muestra: {}", e),
            }
        }

        let avg_loss = epoch_loss / count as f32;
        println!("✅ Época {}/{} completada | Loss: {:.4} | Tiempo: {:?}", epoch + 1, epochs, avg_loss, start.elapsed());
        
        // Final epoch checkpoint
        let mut final_config = config.clone();
        final_config.config.state = format!("completed-epoch-{}", epoch + 1);
        _impl::io::loader::save_genomic_model(output_path, &student, &final_config, Some(&tokenizer))?;
    }

    println!("\n[*] Guardado final de seguridad...");
    let mut exit_config = config.clone();
    exit_config.config.state = "graceful-exit".to_string();
    _impl::io::loader::save_genomic_model(output_path, &student, &exit_config, Some(&tokenizer))?;

    println!("\n✨ Destilación completada o interrumpida. Modelo guardado en: {}", output_path);
    Ok(())
}
