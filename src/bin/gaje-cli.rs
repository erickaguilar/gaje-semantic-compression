use _impl::compute::doctor;
use _impl::compute::kernels;
use _impl::io::models_cmd;
use _impl::nn::repl::{self, ReplConfig};
use clap::{Args, Parser, Subcommand};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    name = "gaje-cli",
    author = "Erick Aguilar",
    version = "1.7.0-alpha",
    about = "🧬 GAJE HELIX — Motor de Compresión Semántica Genómica y Ejecución Nativa",
    long_about = "Framework soberano de inferencia ultrarrápida, memoria genética mmap zero-copy y compresión neuronal híbrida."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Ruta al modelo binario (.flat o .gaje) para modo directo
    #[arg(long, global = true)]
    model: Option<String>,

    /// Ejecutar inferencia de un solo disparo con un prompt
    #[arg(long, global = true)]
    prompt: Option<String>,

    /// Temperatura de muestreo [0.0 - 2.0]
    #[arg(long, global = true, default_value_t = 0.4)]
    temperature: f32,

    /// Penalización de repetición [1.0 - 2.0]
    #[arg(long, global = true, default_value_t = 1.15)]
    repetition_penalty: f32,

    /// Límite máximo de nuevos tokens generados
    #[arg(long, global = true, default_value_t = 256)]
    max_tokens: usize,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Inicia una sesión interactiva REPL en la terminal
    Chat(ChatArgs),

    /// Inicia el servidor HTTP nativo en Rust con streaming SSE y servicio de la Web UI
    Serve(ServeArgs),

    /// Ejecuta diagnósticos de hardware, extensiones SIMD y ancho de banda
    Doctor,

    /// Catálogo e inspección estructural de modelos planos (.flat)
    Models(ModelsArgs),

    /// Descarga automatizada de modelos mediante motor nativo multi-stream
    #[command(alias = "download")]
    Pull(PullArgs),

    /// Benchmark estandarizado de latencia (TTFT), velocidad y memoria
    #[command(alias = "bench")]
    Benchmark(BenchArgs),

    /// Exporta modelos (.gaje, .gguf, .flat) al formato plano de producción .flat v2
    ExportFlat(ExportFlatArgs),

    /// Construye y normaliza datasets de texto/JSONL para entrenamiento o DNI
    DatasetBuild(DatasetBuildArgs),

    /// Auditoría matemática de integridad, ausencia de NaNs y entropía de pesos
    Audit(AuditArgs),

    /// Gestión de épocas de memoria asociativa (.gmem v2)
    Epoch(EpochArgs),
}

#[derive(Args, Debug)]
struct ChatArgs {
    /// Archivo del modelo a ejecutar (.flat)
    #[arg(short, long)]
    model: Option<String>,

    /// Prompt para inferencia directa (no interactiva)
    #[arg(short, long)]
    prompt: Option<String>,

    /// Temperatura de muestreo
    #[arg(short, long, default_value_t = 0.4)]
    temperature: f32,

    /// Penalización por repetición
    #[arg(short, long, default_value_t = 1.15)]
    repetition_penalty: f32,

    /// Tokens máximos a generar
    #[arg(short, long, default_value_t = 256)]
    max_tokens: usize,

    /// Prompt del sistema
    #[arg(long, default_value = "Eres GAJE AI, un asistente genómico soberano, conciso y útil.")]
    system: String,
}

#[derive(Args, Debug)]
struct ServeArgs {
    /// Dirección IP de enlace de red (por defecto: 127.0.0.1)
    #[arg(short = 'H', long, default_value = "127.0.0.1")]
    host: String,

    /// Puerto TCP de escucha (por defecto: 8080)
    #[arg(short, long, default_value_t = 8080)]
    port: u16,

    /// Directorio de modelos (.flat)
    #[arg(long, default_value = "models")]
    models_dir: String,

    /// Directorio de recursos estáticos de la Web UI
    #[arg(long, default_value = "examples/ui/web_ui")]
    static_dir: String,

    /// Modelo inicial a precargar en memoria
    #[arg(short, long)]
    model: Option<String>,

    /// Modo ultra-ligero para móviles y edge (omite docs y grafos)
    #[arg(long)]
    chat_only: bool,
}

#[derive(Args, Debug)]
struct ModelsArgs {
    #[command(subcommand)]
    action: Option<ModelsSubcommand>,
}

#[derive(Subcommand, Debug)]
enum ModelsSubcommand {
    /// Lista todos los modelos disponibles en el directorio
    List {
        /// Directorio de búsqueda (por defecto: models/)
        #[arg(default_value = "models")]
        dir: String,
    },
    /// Muestra la cabecera y estructura interna de un archivo .flat
    Inspect {
        /// Ruta al archivo .flat
        file: String,
    },
    /// Verifica la integridad estructural y magic bytes de un modelo
    Verify {
        /// Ruta al archivo .flat
        file: String,
    },
    /// Incrusta un tokenizador GTOK en un archivo .flat existente
    InjectGtok {
        /// Ruta al archivo .flat
        file: String,
        /// Ruta al tokenizador .gtok (opcional, se auto-detecta si no se especifica)
        #[arg(short, long)]
        tokenizer: Option<String>,
    },
    /// Auto-inyecta GTOK en todos los modelos del catálogo que carezcan de él
    InjectAll {
        /// Directorio de búsqueda (por defecto: models/)
        #[arg(default_value = "models")]
        dir: String,
    },
}

#[derive(Args, Debug)]
struct PullArgs {
    /// Identificador del modelo (pico, nano, prime, ultra, repo HF o URL)
    target: String,

    /// Directorio de destino
    #[arg(short, long, default_value = "models")]
    out: String,

    /// Número de conexiones concurrentes
    #[arg(short, long, default_value_t = 8)]
    concurrency: usize,

    /// Tamaño mínimo por fragmento en MB
    #[arg(long, default_value_t = 2)]
    min_chunk: u64,
}

#[derive(Args, Debug)]
struct BenchArgs {
    /// Archivo del modelo a evaluar (.flat o .gaje)
    #[arg(short, long)]
    model: Option<String>,

    /// Prompt de evaluación
    #[arg(short, long, default_value = "Explica en pocas palabras qué es la compresión semántica genómica.")]
    prompt: String,

    /// Número de tokens a generar
    #[arg(short, long, default_value_t = 64)]
    tokens: usize,

    /// Archivo de texto o JSONL para cálculo de perplejidad
    #[arg(long)]
    corpus: Option<String>,
}

#[derive(Args, Debug)]
struct ExportFlatArgs {
    /// Archivo de entrada (.gaje, .gguf o .flat)
    input: String,

    /// Archivo de salida (.flat)
    #[arg(short, long)]
    output: String,

    /// Tokenizador a incrustar (opcional)
    #[arg(short, long)]
    tokenizer: Option<String>,

    /// Esquema de cuantización: 1=Q4_0 (default), 2=Q8_0, 3=Q2_0
    #[arg(long, default_value_t = 1)]
    quant_format: u32,
}

#[derive(Args, Debug)]
struct DatasetBuildArgs {
    /// Archivos de texto o jsonl de entrada
    inputs: Vec<String>,

    /// Archivo de salida normalizado
    #[arg(short, long)]
    output: String,

    /// Tokenizador para validación y filtrado (opcional)
    #[arg(short, long)]
    tokenizer: Option<String>,

    /// Longitud mínima de texto por línea
    #[arg(long, default_value_t = 10)]
    min_len: usize,
}

#[derive(Args, Debug)]
struct AuditArgs {
    /// Archivo del modelo a auditar (.flat)
    model: String,

    /// Auditar entropía y distribución de centroides
    #[arg(long, default_value_t = true)]
    entropy: bool,

    /// Verificar presencia de valores anómalos (NaN/Inf)
    #[arg(long, default_value_t = true)]
    check_nan: bool,
}

#[derive(Args, Debug)]
struct EpochArgs {
    /// Subcomando de época: list, rollback, merge, evolve
    action: String,

    /// Identificador del organismo
    #[arg(long, default_value = "default")]
    organism: String,

    /// Ruta al archivo de memoria .gmem
    #[arg(long, default_value = "data/memory/default.gmem")]
    path: String,

    /// ID de época para rollback
    #[arg(long, default_value_t = 0)]
    epoch_id: u64,

    /// ID de época origen (para merge)
    #[arg(long, default_value_t = 0)]
    source_epoch_id: u64,
}

fn resolve_default_model(model_opt: Option<String>) -> String {
    if let Some(m) = model_opt {
        return m;
    }
    // Buscar en rutas comunes
    let candidates = [
        "models/production/gaje_pico_135m.flat",
        "models/production/gaje_nano_1.5b.flat",
        "models/gaje_pico_135m.flat",
        "models/gaje_nano_1.5b.flat",
    ];
    for c in &candidates {
        if Path::new(c).exists() {
            return c.to_string();
        }
    }
    "models/production/gaje_pico_135m.flat".to_string()
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    unsafe {
        kernels::init_shuffle_table();
    }

    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        println!("\n[!] Interrupción detectada (Ctrl+C). Finalizando de forma segura...");
        r.store(false, Ordering::SeqCst);
    })
    .expect("Error configurando el manejador de señales");

    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Doctor) => {
            let report = doctor::run_doctor();
            doctor::print_doctor_report(&report);
            Ok(())
        }
        Some(Commands::Models(models_args)) => {
            match models_args.action.unwrap_or(ModelsSubcommand::List { dir: "models".to_string() }) {
                ModelsSubcommand::List { dir } => {
                    let models = models_cmd::list_models(Path::new(&dir))?;
                    models_cmd::print_models_table(&models);
                    Ok(())
                }
                ModelsSubcommand::Inspect { file } => {
                    models_cmd::inspect_model(Path::new(&file))?;
                    Ok(())
                }
                ModelsSubcommand::Verify { file } => {
                    models_cmd::verify_model(Path::new(&file))?;
                    Ok(())
                }
                ModelsSubcommand::InjectGtok { file, tokenizer } => {
                    let tok_ref = tokenizer.as_ref().map(|s| Path::new(s.as_str()));
                    models_cmd::inject_gtok(Path::new(&file), tok_ref)?;
                    Ok(())
                }
                ModelsSubcommand::InjectAll { dir } => {
                    models_cmd::inject_all_gtok(Path::new(&dir))?;
                    Ok(())
                }
            }
        }
        Some(Commands::Pull(pull_args)) => {
            let opts = _impl::io::downloader::DownloadOptions {
                concurrency: pull_args.concurrency,
                chunk_size_min: pull_args.min_chunk * 1024 * 1024,
                user_agent: "GAJE-Helix-Engine/1.7.0 (Rust; Native-Downloader)".to_string(),
            };
            println!("⚡ [GAJE CLI] Iniciando descarga nativa acelerada para: {}", pull_args.target);
            let stats = _impl::io::downloader::download_model(
                &pull_args.target,
                Some(Path::new(&pull_args.out)),
                Some(opts),
                Some(running),
            )?;
            println!("🎉 Descarga completada con éxito en {:?}", stats.destination);
            Ok(())
        }
        Some(Commands::Serve(serve_args)) => {
            let config = _impl::server::ServerConfig {
                host: serve_args.host,
                port: serve_args.port,
                models_dir: PathBuf::from(serve_args.models_dir),
                static_dir: PathBuf::from(serve_args.static_dir),
                initial_model: serve_args.model,
                chat_only: serve_args.chat_only,
            };
            _impl::server::run_server(config, running)?;
            Ok(())
        }
        Some(Commands::Chat(chat_args)) => {
            let model_path = resolve_default_model(chat_args.model);
            if let Some(prompt) = chat_args.prompt {
                run_single_prompt(&model_path, &prompt, chat_args.temperature, chat_args.repetition_penalty, chat_args.max_tokens)?;
            } else {
                let config = ReplConfig {
                    model_path,
                    temperature: chat_args.temperature,
                    repetition_penalty: chat_args.repetition_penalty,
                    max_tokens: chat_args.max_tokens,
                    system_prompt: chat_args.system,
                };
                repl::run_repl(config, running)?;
            }
            Ok(())
        }
        Some(Commands::Benchmark(bench_args)) => {
            let model_path = resolve_default_model(bench_args.model);
            _impl::io::cli_tools::benchmark_cmd(
                &model_path,
                &bench_args.prompt,
                bench_args.tokens,
                bench_args.corpus.as_deref(),
            )?;
            Ok(())
        }
        Some(Commands::ExportFlat(export_args)) => {
            _impl::io::cli_tools::export_flat_cmd(
                &export_args.input,
                &export_args.output,
                export_args.tokenizer.as_deref(),
                export_args.quant_format,
            )?;
            Ok(())
        }
        Some(Commands::DatasetBuild(dataset_args)) => {
            _impl::io::cli_tools::dataset_build_cmd(
                &dataset_args.inputs,
                &dataset_args.output,
                dataset_args.tokenizer.as_deref(),
                dataset_args.min_len,
            )?;
            Ok(())
        }
        Some(Commands::Audit(audit_args)) => {
            _impl::io::cli_tools::audit_cmd(
                &audit_args.model,
                audit_args.entropy,
                audit_args.check_nan,
            )?;
            Ok(())
        }
        Some(Commands::Epoch(epoch_args)) => {
            handle_epoch(&epoch_args)
        }
        None => {
            // Si el usuario pasó --model y --prompt directamente
            if let Some(prompt) = cli.prompt {
                let model_path = resolve_default_model(cli.model);
                run_single_prompt(&model_path, &prompt, cli.temperature, cli.repetition_penalty, cli.max_tokens)?;
            } else {
                // Iniciar REPL por defecto
                let model_path = resolve_default_model(cli.model);
                let config = ReplConfig {
                    model_path,
                    temperature: cli.temperature,
                    repetition_penalty: cli.repetition_penalty,
                    max_tokens: cli.max_tokens,
                    ..Default::default()
                };
                repl::run_repl(config, running)?;
            }
            Ok(())
        }
    }
}

fn run_single_prompt(
    model_path: &str,
    prompt: &str,
    temperature: f32,
    repetition_penalty: f32,
    max_tokens: usize,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("📦 Cargando modelo: {}...", model_path);
    let t0 = Instant::now();
    let (mut llm, tokenizer) = repl::load_model_and_tokenizer(model_path)?;
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("✅ Modelo listo en {:.2} ms\n", load_ms);

    let chat_prompt = format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", prompt);
    let prompt_tokens_u32 = tokenizer.encode(&chat_prompt, false).map_err(|e| e.to_string())?;
    let prompt_tokens: Vec<usize> = prompt_tokens_u32.into_iter().map(|t| t as usize).collect();

    let gen_t0 = Instant::now();
    let eos_ids = vec![2, 0];
    let generated_tokens = llm.generate_native_core(
        prompt_tokens,
        max_tokens,
        temperature,
        repetition_penalty,
        eos_ids,
    ).map_err(|e| format!("Error en inferencia: {}", e))?;

    let gen_u32: Vec<u32> = generated_tokens.into_iter().map(|t| t as u32).collect();
    let raw_reply = tokenizer.decode(&gen_u32, true).unwrap_or_default();

    let clean = raw_reply
        .replace("<|im_end|>", "")
        .replace("<|im_start|>", "")
        .replace("<|endoftext|>", "")
        .trim()
        .to_string();

    println!("{}\n", clean);
    let gen_time = gen_t0.elapsed().as_secs_f64();
    let tok_count = gen_u32.len();
    println!("\x1b[90m[{:.1} tok/s · {} tokens · {:.2}s]\x1b[0m", 
        tok_count as f64 / gen_time.max(0.001), tok_count, gen_time);
    Ok(())
}



fn handle_epoch(args: &EpochArgs) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut mgr = match _impl::compute::epoch_manager::EpochManager::new(&args.path, &args.organism, 512) {
        Ok(m) => m,
        Err(e) => return Err(format!("Error abriendo gestor de épocas: {}", e).into()),
    };

    match args.action.as_str() {
        "list" => {
            println!("📚 Épocas registradas para '{}':", args.organism);
            let epochs = mgr.list_epochs().map_err(|e| e.to_string())?;
            for ep in epochs {
                println!(
                    "  • Época #{}: {} (Fecha: {}, Padre: #{}, Estado: {})",
                    ep.epoch_id, ep.comment, ep.created_at, ep.parent_epoch, ep.verdict
                );
            }
        }
        "rollback" => {
            println!("⏪ Realizando rollback a la época #{}...", args.epoch_id);
            mgr.rollback_to(args.epoch_id).map_err(|e| e.to_string())?;
            println!("✅ Rollback exitoso.");
        }
        _ => {
            println!("Acción de época '{}' no reconocida. Usa 'list' o 'rollback'.", args.action);
        }
    }
    Ok(())
}
