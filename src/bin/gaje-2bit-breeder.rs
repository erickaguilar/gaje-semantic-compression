use _impl::core::evolution_bitwise::IslandModel;
use _impl::core::tokenizer::GajeTokenizer;
use _impl::core::topology::CentroidGraph;
use _impl::nn::distiller::{CouncilOfTeachers, Teacher};
use std::fs;
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧬 GAJE 2-BIT BREEDER: Experimento de Evolución Continua 🧬");
    println!("----------------------------------------------------------");

    let args: Vec<String> = std::env::args().collect();
    let student_path = if args.len() > 1 {
        &args[1]
    } else {
        "models/production/smollm2_2bit_flat.gaje.flat"
    };
    let tokenizer_path = "models/core/tokenizer.json";
    let dataset_path = if args.len() > 2 {
        &args[2]
    } else {
        "data/datasets/dataset_es_ext.txt"
    };
    let teacher_model_path = "models/gguf/smollm2-135m-f16.gguf";
    let topology_path = "models/core/topology_es.json";

    // 1. Cargar Tokenizador y Dataset
    println!("[*] Cargando recursos base...");
    let tokenizer = Arc::new(GajeTokenizer::from_file(tokenizer_path)?);
    let dataset_content = fs::read_to_string(dataset_path)?;
    let samples: Vec<String> = dataset_content
        .lines()
        .map(|s| s.trim().to_string())
        .filter(|s| s.len() > 10)
        .take(10) // Tomamos 10 muestras para una evaluación evolutiva ultrarrápida
        .collect();
    println!(
        "📊 Dataset cargado: {} muestras para evaluación rápida.",
        samples.len()
    );

    // 2. Cargar Estudiante (2-bit Flat)
    println!("[*] Cargando Estudiante (2-bit Flat): {}...", student_path);
    let mut student_llm = _impl::io::loader::load_genomic_auto(student_path)?;
    
    // Auto-detectar configuración del modelo cargado
    let n_embd = student_llm.embeddings.out_features;
    let n_head = student_llm.blocks[0].attn.n_head;
    let n_head_kv = student_llm.blocks[0].attn.n_head_kv;
    let n_blocks = student_llm.blocks.len();
    let vocab_size = student_llm.embeddings.in_features;
    let eps = student_llm.eps;
    let rope_base = student_llm.blocks[0].attn.rope_base;
    let rope_style = student_llm.blocks[0].attn.rope_style.clone();

    let config = _impl::io::loader::ModelConfig {
        config: _impl::io::loader::ArchConfig {
            name: "smollm2-2bit-embryo".to_string(),
            version: "0.9.7-embryo".to_string(),
            tokenizer_id: "HuggingFaceTB/SmolLM2-135M-Instruct".to_string(),
            rope_base,
            ffn_act: "swiglu".to_string(),
            use_genomic_norm: false,
            rope_style,
            anchor_threshold: 0.1,
            ffn_anchor_threshold: 0.1,
            rna_threshold: 0.5,
            unpermute_weights: false,
            apply_smollm_rope_patch: false,
            tie_word_embeddings: true,
            dni: "".to_string(),
            state: "stable".to_string(),
        },
        n_embd,
        n_head,
        n_head_kv,
        n_blocks,
        vocab_size: Some(vocab_size),
        eps,
    };

    println!(
        "   └─ Dims detectadas: Emb={n_embd}, Heads={n_head}/{n_head_kv}, Blocks={n_blocks}, Vocab={vocab_size}, RoPE Base={rope_base}"
    );

    // 3. Aplicar Anclaje Algebraico Q(zeta_16) para estabilización
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

    // 4. Configurar el Consejo de Maestros (Teacher Council)
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

    // 6. Inicializar el Island Model Engine (Optimizado para experimento rápido)
    println!("[*] Inicializando Island Model (3 Islas, 12 individuos cada una)...");
    let mut engine = IslandModel::new_llm(
        student_llm,
        3,     // num_islands
        12,    // pop_per_island
        0.0002, // mutation_rate (0.02% de los bits para evitar destrucción del genoma)
        10,    // migration_rate (cada 10 gen)
        topology,
    );
    engine.set_council(council, tokenizer.clone());

    // 7. Bucle de Evolución Principal (Experimento rápido de 20 generaciones)
    let num_generations = 20;
    println!(
        "\n🚀 Iniciando Evolución de 2-bits ({} generaciones)...",
        num_generations
    );

    for gen in 1..=num_generations {
        let start = Instant::now();

        // A. Evaluación Híbrida (Coherence contra el Maestro)
        engine.evaluate_hybrid(&samples);

        // B. Paso Evolutivo (Mutación + Crossover + Migración)
        engine.step();

        let best_fitness = engine.islands[0].population[0].fitness;
        let duration = start.elapsed();

        println!(
            "📈 Gen {:2}/{:2} | Best Coherence Fitness: {:.6} | Tiempo: {:?}",
            gen,
            num_generations,
            best_fitness,
            duration
        );

        // C. Guardar el mejor organismo si mejora
        if gen == num_generations {
            let checkpoint_path = "models/checkpoints/smollm2_2bit_evolved.gaje";
            if let Some(best_organism) = engine.islands[0].population.first() {
                if let Some(llm) = &best_organism.llm {
                    _impl::io::loader::save_genomic_model(
                        checkpoint_path,
                        llm,
                        &config,
                        Some(&tokenizer),
                    )?;
                    println!(
                        "\n💾 ¡Evolución de 2-bits salvada exitosamente! {}",
                        checkpoint_path
                    );
                }
            }
        }
    }

    println!("\n✨ Experimento evolutivo de 2-bits finalizado con éxito.");
    Ok(())
}
