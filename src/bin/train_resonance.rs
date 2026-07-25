use _impl::io::loader::{save_genomic_model, NativeLoader};
use _impl::nn::trainer::GenomicTrainerCore;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "models/checkpoints/gold_embryo.gaje";
    let topology_path = "models/core/topology_es.json";
    let dataset_path = "data/datasets/dataset_es_ext.txt";
    let output_path = "models/checkpoints/gold_embryo_guided.gaje";
    let epochs = 100;
    let lr = 0.0005;

    println!("🧬 GAJE Large-Scale Native Resonance Training (Phase 4.2)");
    println!("[*] Loading model from: {}", model_path);

    let (mut model, config, tokenizer) = {
        let loader = NativeLoader::new(model_path)?;
        let config = loader.load_config()?;
        let model = loader.load_llm()?;
        let tokenizer = loader.load_tokenizer()?;
        (model, config, tokenizer)
    };

    if Path::new(topology_path).exists() {
        println!("[*] Injecting topology: {}", topology_path);
        model.load_topology_core(topology_path)?;
    } else {
        println!("⚠️ Warning: Topology not found, proceeding without guidance.");
    }

    println!("[*] Loading dataset: {}", dataset_path);
    let file = File::open(dataset_path)?;
    let reader = BufReader::new(file);
    let mut dataset = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let tokens = tokenizer.encode(&line, false)?;
        dataset.push(tokens.into_iter().map(|t| t as usize).collect());
    }
    println!("📊 Dataset loaded: {} sequences.", dataset.len());

    let dataset_ref: Vec<Vec<usize>> = dataset;
    let trainer = GenomicTrainerCore::new(lr, 0.05);
    println!("[*] Starting Resonance phase...");

    let p1_end = (epochs as f32 * 0.2) as usize;
    let p2_end = (epochs as f32 * 0.7) as usize;

    for epoch in 0..epochs {
        let phase = if epoch < p1_end {
            1
        } else if epoch < p2_end {
            2
        } else {
            3
        };
        trainer
            .fit_epoch(
                &mut model,
                &dataset_ref,
                epoch,
                epochs,
                phase,
                0,
                |_, _, _| Ok(()),
            )
            .map_err(|e| e.to_string())?;

        // Checkpoint: Sobre-escribir el archivo de salida en cada época
        save_genomic_model(output_path, &model, &config, Some(&tokenizer))?;
        println!("    [Checkpoint] Progreso guardado en: {}", output_path);
    }

    println!("✨ Training completed successfully.");
    Ok(())
}
