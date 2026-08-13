use _impl::core::evolution_bitwise::IslandModel;
use _impl::core::tokenizer::GajeTokenizer;
use _impl::core::topology::CentroidGraph;
use _impl::nn::distiller::{CouncilOfTeachers, Teacher};
use std::fs;
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧬 SILVER BREEDER: Laboratorio de Evolución por Poblaciones (Fase 5.0) 🧬");
    println!("-----------------------------------------------------------------------");

    let args: Vec<String> = std::env::args().collect();
    let student_path = if args.len() > 1 {
        &args[1]
    } else {
        "models/silver_adult_anchored.gaje"
    };
    let tokenizer_path = "models/core/tokenizer.json";
    let dataset_path = if args.len() > 2 {
        &args[2]
    } else {
        "data/datasets/dataset_es_ext.txt"
    };
    let teacher_model_path = "models/gguf/smollm2-135m-f16.gguf";
    let topology_path = "models/core/topology_es.json";

    // 2. Cargar Tokenizador y Dataset
    println!("[*] Cargando recursos base...");
    let tokenizer = Arc::new(GajeTokenizer::from_file(tokenizer_path)?);
    let dataset_content = fs::read_to_string(dataset_path)?;
    let samples: Vec<String> = dataset_content
        .lines()
        .map(|s| s.trim().to_string())
        .filter(|s| s.len() > 10)
        .collect();
    println!(
        "📊 Dataset cargado: {} muestras para evaluación.",
        samples.len()
    );

    // 3. Cargar Estudiante (Silver Adult)
    println!("[*] Cargando Estudiante: {}...", student_path);
    let (mut student_llm, config) = {
        let loader = _impl::io::loader::NativeLoader::new(student_path)?;
        let student_llm = loader.load_llm()?;
        let config = loader.load_config()?;
        (student_llm, config)
    };

    // 3.1. Aplicar Anclaje Algebraico Q(zeta_16) para estabilización
    let codebook_path = "models/core/algebraic_codebook.json";
    if let Ok(f) = std::fs::File::open(codebook_path) {
        let val: serde_json::Value = serde_json::from_reader(f)?;
        if let Some(arr) = val.get("centroids").and_then(|c| c.as_array()) {
            if arr.len() == 4 {
                let algebraic_c: [f32; 4] = [
                    arr[0].as_f64().unwrap_or(0.0) as f32,
                    arr[1].as_f64().unwrap_or(0.0) as f32,
                    arr[2].as_f64().unwrap_or(0.0) as f32,
                    arr[3].as_f64().unwrap_or(0.0) as f32,
                ];
                println!(
                    "[*] Estabilizando fases con Anclaje Algebraico: {:?}",
                    algebraic_c
                );

                let apply_to_layer = |layer: &mut _impl::nn::linear::GenomicLinear| {
                    for i in 0..layer.centroids.len() {
                        layer.centroids[i] = algebraic_c[i % 4];
                    }
                };

                apply_to_layer(&mut student_llm.embeddings);
                apply_to_layer(&mut student_llm.lm_head);
                for block in &mut student_llm.blocks {
                    apply_to_layer(&mut block.q_gen);
                    apply_to_layer(&mut block.k_gen);
                    apply_to_layer(&mut block.v_gen);
                    apply_to_layer(&mut block.w_o);
                    apply_to_layer(&mut block.gate_gen);
                    apply_to_layer(&mut block.up_gen);
                    apply_to_layer(&mut block.w_down);
                }
            }
        }
    }

    // 4. Configurar el Consejo de Maestros
    println!("[*] Configurando Consejo de Maestros...");
    let mut council = CouncilOfTeachers::new();
    let teacher = Teacher::new(
        "SmolLM2-Expert".to_string(),
        teacher_model_path,
        tokenizer_path,
        &tokenizer,
    )?;
    council.add_teacher(teacher);
    let council = Arc::new(council);

    // 5. Cargar Topología Relacional (Opcional)
    let topology = if fs::metadata(topology_path).is_ok() {
        println!("[*] Cargando Topología Relacional: {}...", topology_path);
        let topo_str = fs::read_to_string(topology_path)?;
        let topo: CentroidGraph = serde_json::from_str(&topo_str)?;
        Some(Arc::new(topo))
    } else {
        println!("⚠️ No se encontró topología, procediendo con evolución estocástica pura.");
        None
    };

    // 6. Inicializar el Island Model Engine
    println!("[*] Inicializando Island Model (4 Islas, 16 individuos cada una)...");
    let mut engine = IslandModel::new_llm(
        student_llm,
        4,     // num_islands
        16,    // pop_per_island
        0.005, // mutation_rate (0.5% de los bits)
        50,    // migration_rate (cada 50 gen)
        topology,
    );
    engine.set_council(council, tokenizer.clone());

    // 7. Bucle de Evolución Principal
    let num_generations = 200;
    println!(
        "\n🚀 Iniciando Gran Evolución ({} generaciones)...",
        num_generations
    );

    for gen in 1..=num_generations {
        let start = Instant::now();

        // A. Evaluación Híbrida (Coherence + Needle)
        engine.evaluate_hybrid(&samples);

        // B. Paso Evolutivo (Mutación + Crossover + Migración)
        engine.step();

        let best_fitness = engine.islands[0].population[0].fitness;
        let duration = start.elapsed();

        if gen % 10 == 0 || gen == 1 {
            println!(
                "📈 Gen {:3} | Best Fitness: {:.4} | Tiempo: {:?} | Islas: {}",
                gen,
                best_fitness,
                duration,
                engine.islands.len()
            );
        }

        // C. Guardar Punto de Control (Sobre-escribiendo para estabilidad)
        if gen % 10 == 0 {
            let checkpoint_path = "models/checkpoints/silverfetus-checkpoint.gaje";
            if let Some(best_organism) = engine.islands[0].population.first() {
                if let Some(llm) = &best_organism.llm {
                    _impl::io::loader::save_genomic_model(
                        checkpoint_path,
                        llm,
                        &config,
                        Some(&tokenizer),
                    )?;
                    println!(
                        "💾 Checkpoint actualizado (Gen {}): {}",
                        gen, checkpoint_path
                    );
                }
            }
        }
    }

    println!("\n✨ Evolución completada con éxito. El Silver Fetus ha madurado.");
    Ok(())
}
