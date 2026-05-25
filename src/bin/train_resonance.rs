use _impl::io::loader::{NativeLoader, save_genomic_model};
use _impl::nn::trainer::GenomicTrainerCore;
use std::path::Path;
use std::io::{BufRead, BufReader};
use std::fs::File;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "models/checkpoints/gold_embryo.gaje";
    let topology_path = "models/core/topology_es.json";
    let dataset_path = "data/datasets/dataset_es.txt";
    let output_path = "models/checkpoints/gold_embryo_guided.gaje";
    let epochs = 5;
    let lr = 0.001;

    println!("🧬 GAJE Native Resonance Training (Phase 4.1)");
    println!("[*] Loading model from: {}", model_path);

    let loader = NativeLoader::new(model_path)?;
    let config = loader.load_config()?;
    let mut model = loader.load_llm()?;
    let tokenizer = loader.load_tokenizer()?;

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
        if line.trim().is_empty() { continue; }
        let tokens = tokenizer.encode(&line, false)?;
        dataset.push(tokens.into_iter().map(|t| t as usize).collect());
    }
    println!("📊 Dataset loaded: {} sequences.", dataset.len());

    let dataset_ref: Vec<Vec<usize>> = dataset;
    let trainer = GenomicTrainerCore::new(lr, 0.05);
    println!("[*] Starting Resonance phase...");
    
    trainer.fit(&mut model, &dataset_ref, epochs).map_err(|e| e.to_string())?;

    println!("[*] Saving refined model to: {}", output_path);
    save_genomic_model(output_path, &model, &config, Some(&tokenizer))?;

    println!("✨ Training completed successfully.");
    Ok(())
}
