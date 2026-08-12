use _impl::io::loader::{save_genomic_model, NativeLoader};
use _impl::nn::merger::merge_genomic_models;
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        println!(
            "Usage: gaje-merger <output.gaje> <model1.gaje> <model2.gaje> [<model3.gaje> ...]"
        );
        return Ok(());
    }

    let output_path = &args[1];
    let model_paths = &args[2..];

    let mut models = Vec::new();
    let mut first_config = None;
    let mut first_tokenizer = None;

    for path in model_paths {
        println!("[*] Cargando {}...", path);
        let loader = NativeLoader::new(path)?;
        let config = loader.load_config()?;
        let tokenizer = loader.load_tokenizer().ok();
        let model = loader.load_llm()?;

        if first_config.is_none() {
            first_config = Some(config);
            first_tokenizer = tokenizer;
        }

        models.push(model);
    }

    let merged = merge_genomic_models(&models).map_err(|e| e.to_string())?;

    println!("[*] Guardando modelo fusionado en {}...", output_path);
    save_genomic_model(
        output_path,
        &merged,
        &first_config.unwrap(),
        first_tokenizer.as_ref(),
    )?;

    println!("[+] ¡Fusión exitosa!");
    Ok(())
}
