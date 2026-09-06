use _impl::cli::models as models_cmd;
use _impl::cli::tools as cli_tools;
use _impl::compute::doctor;
use _impl::compute::kernels;
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
    #[command(alias = "repl", alias = "conversar")]
    Chat(ChatArgs),

    /// Inicia el servidor HTTP nativo en Rust con streaming SSE y servicio de la Web UI
    #[command(alias = "servidor", alias = "server", alias = "web")]
    Serve(ServeArgs),

    /// Ejecuta diagnósticos de hardware, extensiones SIMD y ancho de banda
    #[command(alias = "diag", alias = "diagnostico", alias = "info")]
    Doctor,

    /// Catálogo e inspección estructural de modelos (.gaje / .flat)
    #[command(alias = "modelos", alias = "ls")]
    Models(ModelsArgs),

    /// Descarga automatizada de modelos mediante motor nativo multi-stream
    #[command(alias = "download", alias = "descargar", alias = "get")]
    Pull(PullArgs),

    /// Benchmark estandarizado de latencia (TTFT), velocidad y memoria
    #[command(alias = "bench", alias = "medir", alias = "test-speed")]
    Benchmark(BenchArgs),

    /// Exporta modelos (.gaje, .gguf, .flat) al formato plano de producción .flat v2
    #[command(alias = "export", alias = "exportar")]
    ExportFlat(ExportFlatArgs),

    /// Construye y normaliza datasets de texto/JSONL para entrenamiento o DNI
    #[command(alias = "dataset", alias = "datos")]
    DatasetBuild(DatasetBuildArgs),

    /// Auditoría matemática de integridad, ausencia de NaNs y entropía de pesos
    #[command(alias = "auditar", alias = "check")]
    Audit(AuditArgs),

    /// Gestión de épocas de memoria asociativa (.gmem v2)
    #[command(alias = "memoria", alias = "memory")]
    Epoch(EpochArgs),

    /// Orquesta un enjambre de micro-agentes con StateGraph y Tree-of-Thoughts (ToT)
    #[command(alias = "enjambre", alias = "agentes")]
    Swarm(SwarmArgs),

    /// Da a luz a un organismo genómico nativo en 2-bits (Q2_0 / ADN)
    #[command(alias = "nacimiento", alias = "genesis")]
    Birth(BirthArgs),

    /// Entrena un organismo nacido (Q2_0) con el estimador Straight-Through cuaternario
    #[command(alias = "train", alias = "crianza", alias = "entrenar")]
    TrainBorn(TrainBornArgs),

    /// Destilación de conocimiento DNI online (Maestro -> Alumno)
    #[command(alias = "destilar", alias = "dni")]
    Distill(DistillArgs),

    /// Inyecta una mutación genómica controlada in-place en centroides (.gaje)
    #[command(alias = "mutar")]
    Mutate(MutateArgs),

    /// Inspecciona el linaje evolutivo y estado adaptativo de un modelo (.gaje)
    #[command(alias = "historial", alias = "lineage")]
    History(HistoryArgs),

    /// Muestra la guía rápida interactiva y mapa de comandos sin adivinar
    #[command(alias = "menu", alias = "comandos", alias = "guia", alias = "ayuda")]
    Guide,
}

#[derive(Args, Debug)]
struct MutateArgs {
    /// Archivo del modelo a mutar (.gaje o .flat)
    #[arg(short, long)]
    model: Option<String>,

    /// Nombre específico del tensor a mutar (opcional, por defecto: última capa / atención)
    #[arg(short, long)]
    tensor: Option<String>,

    /// Intensidad / sigma de la perturbación estocástica de centroides
    #[arg(short, long, default_value_t = 0.005)]
    intensity: f32,
}

#[derive(Args, Debug)]
struct HistoryArgs {
    /// Archivo del modelo a inspeccionar (.gaje o .flat)
    #[arg(short, long)]
    model: Option<String>,
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
    #[arg(
        long,
        default_value = "Eres GAJE AI, un asistente genómico soberano, conciso y útil."
    )]
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

    /// Suite de evaluación: 'quick', 'full', 'harness', 'reasoning'
    #[arg(long, default_value = "full")]
    suite: String,

    /// Prompt de evaluación personalizado (opcional)
    #[arg(short, long)]
    prompt: Option<String>,

    /// Número de tokens a generar por prueba
    #[arg(short, long, default_value_t = 64)]
    tokens: usize,

    /// Archivo de texto o JSONL para cálculo de perplejidad
    #[arg(long)]
    corpus: Option<String>,

    /// Formato de salida: 'console', 'markdown', 'json'
    #[arg(short, long, default_value = "console")]
    format: String,

    /// Ruta de archivo para exportar el reporte (ej. docs/reports/BENCHMARK_OFFICIAL.md)
    #[arg(short, long)]
    output: Option<String>,
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
    /// Subcomando de época: list, snapshot, rollback, promote, seal, merge, evolve
    action: String,

    /// Identificador del organismo
    #[arg(long, default_value = "default")]
    organism: String,

    /// Ruta o directorio raíz de memoria .gmem
    #[arg(long, alias = "root", default_value = "data/memory")]
    path: String,

    /// Dimensión vectorial del organismo
    #[arg(long, default_value_t = 512)]
    dim: usize,

    /// ID de época para rollback, promote o seal
    #[arg(long, alias = "epoch", default_value_t = 0)]
    epoch_id: u64,

    /// ID de época origen (para merge)
    #[arg(long, default_value_t = 0)]
    source_epoch_id: u64,

    /// Comentario descriptivo para snapshot
    #[arg(long, default_value = "Snapshot manual CLI")]
    comment: String,
}

#[derive(Args, Debug)]
struct SwarmArgs {
    /// Consulta o tarea a procesar por el enjambre
    #[arg(short, long)]
    prompt: Option<String>,

    /// Usar razonamiento en árbol Tree-of-Thoughts (MCTS)
    #[arg(long, default_value_t = true)]
    tot: bool,

    /// Número de iteraciones / presupuesto de evaluaciones MCTS
    #[arg(long, default_value_t = 16)]
    budget: usize,

    /// Profundidad máxima del árbol ToT
    #[arg(long, default_value_t = 3)]
    depth: usize,
}

#[derive(Args, Debug)]
struct BirthArgs {
    /// Nombre del organismo nacido (por defecto: max)
    #[arg(short, long, default_value = "max")]
    name: String,

    /// Dimensión oculta del modelo (por defecto: 256)
    #[arg(long, default_value_t = 256)]
    dim: usize,

    /// Número de capas transformer (por defecto: 8)
    #[arg(long, default_value_t = 8)]
    layers: usize,

    /// Número de cabezas de atención (por defecto: 4)
    #[arg(long, default_value_t = 4)]
    heads: usize,

    /// Dimensión intermedia FFN (por defecto: 768)
    #[arg(long, default_value_t = 768)]
    ffn_dim: usize,

    /// Tamaño de vocabulario (por defecto: 49152, compatible con SmolLM2/GAJE GTOK)
    #[arg(long, default_value_t = 49152)]
    vocab_size: usize,

    /// Ruta de salida del archivo .gaje
    #[arg(short, long)]
    output: Option<String>,

    /// Ruta opcional a un tokenizador binario (.gtok o .bin) para incrustar nativamente
    #[arg(short, long)]
    tokenizer: Option<String>,

    /// Vincular e inicializar hipocampo de memoria congénita (.gmem) desde el nacimiento
    #[arg(long)]
    with_memory: bool,

    /// Ruta opcional a un archivo de hechos iniciales (JSONL o texto) para el nicho documental
    #[arg(long)]
    memory_facts: Option<String>,
}

#[derive(Args, Debug)]
struct TrainBornArgs {
    /// Archivo del modelo genómico (.gaje o .flat)
    #[arg(short, long, default_value = "models/born/max_512_pro.gaje")]
    model: String,

    /// Ruta al dataset de entrenamiento (JSONL o texto)
    #[arg(
        short,
        long,
        default_value = "data/genesis_conversational_corpus.jsonl"
    )]
    dataset: String,

    /// Tasa de aprendizaje (Learning Rate)
    #[arg(long, default_value_t = 0.005)]
    lr: f32,

    /// Número de épocas de entrenamiento
    #[arg(short, long, default_value_t = 10)]
    epochs: usize,

    /// Decaimiento por capas (Layer-wise decay)
    #[arg(long, default_value_t = 0.95)]
    lr_decay: f32,

    /// Recorte de gradientes (Gradient Clipping)
    #[arg(long, default_value_t = 1.0)]
    gclip: f32,

    /// Ruta de guardado tras el entrenamiento
    #[arg(short, long)]
    output: Option<String>,

    /// Acelerar entrenamiento mediante GPU (WGPU / Vulkan)
    #[arg(long)]
    gpu: bool,

    /// Número de capas superiores a entrenar (Ladder training, e.g. 4 para entrenar solo bloques 8-11 + LM Head). Por defecto: todas
    #[arg(short = 'l', long = "train-layers")]
    train_layers: Option<usize>,
}

#[derive(Args, Debug)]
struct DistillArgs {
    /// Archivo del modelo maestro (.flat, .gaje o .gguf)
    #[arg(short, long)]
    teacher: String,

    /// Archivo del modelo alumno (.flat o .gaje)
    #[arg(short, long)]
    student: String,

    /// Dataset en formato JSONL o texto para destilación
    #[arg(short, long)]
    dataset: String,

    /// Épocas de destilación
    #[arg(short, long, default_value_t = 3)]
    epochs: usize,

    /// Tasa de aprendizaje (Learning Rate)
    #[arg(long, default_value_t = 0.001)]
    lr: f32,

    /// Ponderación de divergencia KL vs SFT (alpha: 0.0=solo CE, 1.0=solo KL)
    #[arg(long, default_value_t = 0.7)]
    alpha: f32,

    /// Temperatura de suavizado softmax para destilación
    #[arg(long, default_value_t = 1.0)]
    temperature: f32,

    /// Tokenizador opcional para el maestro
    #[arg(long)]
    teacher_tokenizer: Option<String>,

    /// Tokenizador opcional para el alumno
    #[arg(long)]
    student_tokenizer: Option<String>,

    /// Archivo de salida para el modelo refinado (.flat o .gaje)
    #[arg(short, long)]
    output: Option<String>,
}

fn resolve_default_model(model_opt: Option<String>) -> String {
    if let Some(m) = model_opt {
        return m;
    }
    // Buscar en rutas comunes
    let candidates = [
        "models/production/gaje_pico_135m.flat",
        "models/production/gaje_coder_3b.flat",
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
            match models_args.action.unwrap_or(ModelsSubcommand::List {
                dir: "models".to_string(),
            }) {
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
            println!(
                "⚡ [GAJE CLI] Iniciando descarga nativa acelerada para: {}",
                pull_args.target
            );
            let stats = _impl::io::downloader::download_model(
                &pull_args.target,
                Some(Path::new(&pull_args.out)),
                Some(opts),
                Some(running),
            )?;
            println!(
                "🎉 Descarga completada con éxito en {:?}",
                stats.destination
            );
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
                run_single_prompt(
                    &model_path,
                    &prompt,
                    chat_args.temperature,
                    chat_args.repetition_penalty,
                    chat_args.max_tokens,
                )?;
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
            cli_tools::benchmark_cmd(
                &model_path,
                &bench_args.suite,
                bench_args.prompt.as_deref(),
                bench_args.tokens,
                bench_args.corpus.as_deref(),
                &bench_args.format,
                bench_args.output.as_deref(),
            )?;
            Ok(())
        }
        Some(Commands::ExportFlat(export_args)) => {
            cli_tools::export_flat_cmd(
                &export_args.input,
                &export_args.output,
                export_args.tokenizer.as_deref(),
                export_args.quant_format,
            )?;
            Ok(())
        }
        Some(Commands::DatasetBuild(dataset_args)) => {
            cli_tools::dataset_build_cmd(
                &dataset_args.inputs,
                &dataset_args.output,
                dataset_args.tokenizer.as_deref(),
                dataset_args.min_len,
            )?;
            Ok(())
        }
        Some(Commands::Audit(audit_args)) => {
            cli_tools::audit_cmd(&audit_args.model, audit_args.entropy, audit_args.check_nan)?;
            Ok(())
        }
        Some(Commands::Epoch(epoch_args)) => handle_epoch(&epoch_args),
        Some(Commands::Swarm(swarm_args)) => handle_swarm(&swarm_args),
        Some(Commands::Birth(birth_args)) => handle_birth(&birth_args),
        Some(Commands::TrainBorn(train_args)) => handle_train_born(&train_args),
        Some(Commands::Distill(distill_args)) => handle_distill(&distill_args),
        Some(Commands::Mutate(mutate_args)) => handle_mutate(&mutate_args),
        Some(Commands::History(history_args)) => handle_history(&history_args),
        Some(Commands::Guide) => {
            print_cli_guide();
            Ok(())
        }
        None => {
            // Si el usuario pasó --model y --prompt directamente
            if let Some(prompt) = cli.prompt {
                let model_path = resolve_default_model(cli.model);
                run_single_prompt(
                    &model_path,
                    &prompt,
                    cli.temperature,
                    cli.repetition_penalty,
                    cli.max_tokens,
                )?;
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
    println!("✅ Modelo listo en {:.2} ms", load_ms);

    let mut context_prefix = String::new();
    if let Some(orch) = _impl::compute::island::IslandOrchestrator::try_load_paired_memory(
        model_path,
        llm.dim() as u32,
    ) {
        let q_vec = _impl::compute::island::IslandOrchestrator::vector_from_text(prompt, llm.dim());
        let matches = orch.retrieve_context(&q_vec, 2);
        let relevant: Vec<_> = matches
            .into_iter()
            .filter(|m| m.similarity >= 0.50)
            .collect();
        if !relevant.is_empty() {
            let facts_str = relevant
                .iter()
                .map(|m| m.text.as_str())
                .collect::<Vec<_>>()
                .join(" | ");
            println!("🧠 [Hipocampo] Inyectado contexto relevante: {}", facts_str);
            context_prefix = format!("[Conocimiento Hipocampal: {}]\n", facts_str);
        }
    }
    println!();

    let chat_prompt = if context_prefix.is_empty() {
        format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            prompt
        )
    } else {
        format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            context_prefix, prompt
        )
    };
    let prompt_tokens_u32 = tokenizer
        .encode(&chat_prompt, false)
        .map_err(|e| e.to_string())?;
    let prompt_tokens: Vec<usize> = prompt_tokens_u32.into_iter().map(|t| t as usize).collect();

    let gen_t0 = Instant::now();
    let eos_ids = vec![2, 0];
    let generated_tokens = llm
        .generate_native_core(
            prompt_tokens,
            max_tokens,
            temperature,
            repetition_penalty,
            eos_ids,
        )
        .map_err(|e| format!("Error en inferencia: {}", e))?;

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
    println!(
        "\x1b[90m[{:.1} tok/s · {} tokens · {:.2}s]\x1b[0m",
        tok_count as f64 / gen_time.max(0.001),
        tok_count,
        gen_time
    );
    Ok(())
}

fn handle_epoch(args: &EpochArgs) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let dim = args.dim as u32;
    let mut mgr =
        match _impl::compute::epoch_manager::EpochManager::new(&args.path, &args.organism, dim) {
            Ok(m) => m,
            Err(e) => return Err(format!("Error abriendo gestor de épocas: {}", e).into()),
        };

    match args.action.as_str() {
        "list" => {
            let epochs = mgr.list_epochs().map_err(|e| e.to_string())?;
            let active = mgr.active_epoch_id;
            println!(
                "📚 Épocas registradas para '{}': (Época Activa: {})",
                args.organism, active
            );
            for ep in epochs {
                let star = if ep.epoch_id == active {
                    format!("*{}", ep.epoch_id)
                } else {
                    format!(" {}", ep.epoch_id)
                };
                println!(
                    "  {} Época #{}: {} (Fecha: {}, Padre: #{}, Estado: {:?})",
                    star, ep.epoch_id, ep.comment, ep.created_at, ep.parent_epoch, ep.verdict
                );
            }
        }
        "snapshot" => {
            let mut orch = _impl::compute::island::IslandOrchestrator::new(dim);
            let new_id = mgr
                .create_snapshot(&mut orch, &args.comment, None)
                .map_err(|e| e.to_string())?;
            println!("✅ Creado snapshot: Época ID {}", new_id);
        }
        "rollback" => {
            println!("⏪ Realizando rollback a la época #{}...", args.epoch_id);
            mgr.rollback_to(args.epoch_id).map_err(|e| e.to_string())?;
            println!(
                "✅ Rollback exitoso. Época activa ahora es ID {}",
                args.epoch_id
            );
        }
        "promote" => {
            mgr.promote_epoch(args.epoch_id)
                .map_err(|e| e.to_string())?;
            println!("✅ Época ID {} promovida a activa.", args.epoch_id);
        }
        "seal" => {
            mgr.seal_epoch(args.epoch_id).map_err(|e| e.to_string())?;
            println!("✅ Época ID {} SELLADA (SEALED).", args.epoch_id);
        }
        _ => {
            println!(
                "Acción de época '{}' no reconocida. Usa 'list', 'snapshot', 'rollback', 'promote' o 'seal'.",
                args.action
            );
        }
    }
    Ok(())
}

fn handle_swarm(args: &SwarmArgs) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use _impl::compute::graph::*;
    use std::sync::Arc;

    let query = args.prompt.clone().unwrap_or_else(|| {
        "Deducir y sintetizar la relación óptima entre compresión genómica y latencia MCTS"
            .to_string()
    });

    println!("\n🧬 GAJE AGENTIC SWARM — Orquestador de Enjambre Multi-Agente");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📝 Consulta: \"{}\"", query);

    let mut graph = StateGraph::new();

    // Node 0: Factual Direct Specialist
    struct FactualNode;
    impl AgentNode for FactualNode {
        fn name(&self) -> &str {
            "micro_factual_135m"
        }
        fn process(&self, mut state: AgentState) -> Result<StepResult, String> {
            state.touch();
            state.response = Some(format!(
                "[135M Factual] Respuesta resuelta directamente: {}",
                state.user_query
            ));
            Ok(StepResult::End(state))
        }
    }
    let factual_idx = graph.add_node(Arc::new(FactualNode));

    // Node 1: Tool Node (Calculator Sandbox)
    let tool_node = ToolNode::new("math_calculator", factual_idx, |st| {
        Ok(format!("computed_sandbox({})", st.user_query))
    });
    let tool_idx = graph.add_node(Arc::new(tool_node));

    // Node 2: ToT Reasoner Node (MCTS Tree-of-Thoughts)
    let tot_node = ToTNode::new(
        "tot_mcts_reasoner",
        args.depth,
        args.budget,
        1.41,
        factual_idx,
    );
    let tot_idx = graph.add_node(Arc::new(tot_node));

    // Node 3: Swarm Router
    let router = SwarmRouterNode::new("swarm_router", factual_idx, tot_idx, 0.70)
        .add_intent_route(
            vec!["calcular".into(), "math".into(), "+".into()],
            SwarmIntent::ToolExecution,
            tool_idx,
        )
        .add_intent_route(
            vec![
                "deducir".into(),
                "analizar".into(),
                "relación".into(),
                "sintetizar".into(),
            ],
            SwarmIntent::DeepReasoning,
            tot_idx,
        );
    let router_idx = graph.add_node(Arc::new(router));

    let executor = SwarmExecutor::new(Arc::new(graph));
    let (state, hops, elapsed_ms) = executor
        .execute_profiled(router_idx, AgentState::with_query(query))
        .map_err(|e| format!("{:?}", e))?;

    println!("\n⚡ Telemetría del Grafo Agéntico:");
    println!(
        "  • Intención Detectada  : {}",
        state.intent.as_deref().unwrap_or("DirectFactual")
    );
    println!("  • Saltos en el Grafo   : {} pasos", hops);
    println!("  • Latencia de Ejecución: {:.2} ms", elapsed_ms);
    if !state.context.is_empty() {
        println!(
            "  • Contexto Acumulado   : {}",
            state.context.join("\n    ")
        );
    }
    if !state.tool_outputs.is_empty() {
        println!("  • Salidas de Tools     : {:?}", state.tool_outputs);
    }
    println!(
        "\n💬 Respuesta Final:\n{}\n",
        state.response.as_deref().unwrap_or("Sin respuesta")
    );

    Ok(())
}

fn handle_birth(args: &BirthArgs) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use _impl::io::config::ModelConfig;
    use _impl::io::flat_writer::save_genomic_flat_q;
    use _impl::nn::llm::birth::{create_born_organism, BornConfig};
    use std::path::Path;

    let output_path = args
        .output
        .clone()
        .unwrap_or_else(|| format!("models/born/{}.gaje", args.name));

    println!("\n🧬 GAJE PROTOCOLO DE GÉNESIS — Nacimiento de Organismo en 2-Bits (Q2_0)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  • Nombre del Organismo : {}", args.name);
    println!("  • Dimensión Oculta     : {}", args.dim);
    println!("  • Capas Neuronales     : {}", args.layers);
    println!("  • Cabezas de Atención  : {}", args.heads);
    println!("  • Dimensión FFN        : {}", args.ffn_dim);
    println!("  • Tamaño Vocabulario   : {} tokens", args.vocab_size);
    println!("  • Esquema de Pesos     : Q2_0 (2.0 bits/peso, Constelación Cuaternaria)");
    println!("  • Destino del Organismo: {}", output_path);

    let config = BornConfig {
        name: args.name.clone(),
        vocab_size: args.vocab_size,
        dim: args.dim,
        n_layers: args.layers,
        n_heads: args.heads,
        intermediate_dim: args.ffn_dim,
        eps: 1e-6,
        k_wta_ratio: 0.15,
    };

    println!("\n⏳ Inicializando genoma conforme y tensores cuaternarios...");
    let t0 = std::time::Instant::now();
    let organism = create_born_organism(config);
    let init_elapsed = t0.elapsed();
    println!("✅ Estructura celular creada en {:.2?}", init_elapsed);

    // Asegurar directorio padre
    if let Some(parent) = Path::new(&output_path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    let arch_config = _impl::io::config::ArchConfig {
        name: args.name.clone(),
        version: "2.0.0-born".to_string(),
        tokenizer_id: "gpt2".to_string(),
        rope_base: 10000.0,
        ffn_act: "silu".to_string(),
        use_genomic_norm: false,
        rope_style: "rope".to_string(),
        anchor_threshold: 0.1,
        ffn_anchor_threshold: 0.1,
        rna_threshold: 0.5,
        unpermute_weights: false,
        apply_smollm_rope_patch: false,
        tie_word_embeddings: false,
        dni: _impl::io::config::default_dni(),
        state: "born".to_string(),
    };

    let model_config = ModelConfig {
        config: arch_config,
        n_embd: args.dim,
        n_head: args.heads,
        n_head_kv: args.heads,
        n_blocks: args.layers,
        vocab_size: Some(args.vocab_size),
        eps: 1e-6,
    };

    println!("💾 Escribiendo archivo .gaje zero-copy...");
    let tokenizer_obj = if let Some(path) = &args.tokenizer {
        if path.ends_with(".json") {
            _impl::core::tokenizer::GajeTokenizer::from_file(path).ok()
        } else {
            None
        }
    } else {
        None
    };

    let t_write = std::time::Instant::now();
    save_genomic_flat_q(
        &output_path,
        &organism,
        &model_config,
        tokenizer_obj.as_ref(),
        3,
    )?;
    let write_elapsed = t_write.elapsed();

    let mut memory_info = None;
    if args.with_memory {
        let memory_dir = if output_path.ends_with(".gaje") {
            output_path.strip_suffix(".gaje").unwrap().to_string() + "_memory"
        } else {
            format!("{}_memory", output_path)
        };

        let mut orch = _impl::compute::island::IslandOrchestrator::new(args.dim as u32);
        let mut facts_count = 0;

        if let Some(facts_path) = &args.memory_facts {
            if let Ok(content) = std::fs::read_to_string(facts_path) {
                for (idx, line) in content.lines().enumerate() {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    let fact_text =
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
                            v.get("text")
                                .and_then(|t| t.as_str())
                                .unwrap_or(trimmed)
                                .to_string()
                        } else {
                            trimmed.to_string()
                        };

                    use std::hash::{Hash, Hasher};
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    fact_text.hash(&mut hasher);
                    let seed = hasher.finish();

                    let mut vec = vec![0.0f32; args.dim];
                    let mut norm_sq = 0.0f32;
                    for (i, val) in vec.iter_mut().enumerate() {
                        let pseudo = ((seed
                            .wrapping_add(i as u64)
                            .wrapping_mul(6364136223846793005))
                            >> 32) as i32;
                        let f = (pseudo as f32) / (i32::MAX as f32);
                        *val = f;
                        norm_sq += f * f;
                    }
                    let norm = norm_sq.sqrt().max(1e-8);
                    for val in vec.iter_mut() {
                        *val /= norm;
                    }

                    orch.add_memory(
                        _impl::compute::island::IslandNiche::Documental,
                        (idx + 1) as u64,
                        vec,
                        fact_text,
                    );
                    facts_count += 1;
                }
            }
        }

        orch.save_all(&memory_dir)?;
        memory_info = Some((memory_dir, facts_count));
    }

    let file_size = std::fs::metadata(&output_path)?.len();
    let size_mb = file_size as f64 / (1024.0 * 1024.0);

    println!("🎉 ¡Nacimiento completado con éxito!");
    println!("  • Archivo Genómico     : {}", output_path);
    println!("  • Tamaño en Disco      : {:.2} MB", size_mb);
    println!("  • Tiempo de Exportación: {:.2?}", write_elapsed);
    if let Some((mem_dir, count)) = memory_info {
        println!(
            "  • Hipocampo Congénito  : {} ({} hechos en nicho documental, D={})",
            mem_dir, count, args.dim
        );
    }
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    Ok(())
}

fn handle_train_born(args: &TrainBornArgs) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use _impl::io::flat_reader::GajeFlatFileReader;
    use _impl::io::flat_writer::save_genomic_flat_q;

    println!("\n🧬 GAJE PROTOCOLO DE CRIANZA — Entrenamiento Nativo STE en 2-Bits (Q2_0)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  • Modelo Organismo    : {}", args.model);
    println!("  • Dataset de Crianza  : {}", args.dataset);
    println!("  • Épocas              : {}", args.epochs);
    println!("  • Learning Rate       : {:.4}", args.lr);
    println!("  • Layer-wise Decay    : {:.2}", args.lr_decay);
    println!("  • Gradient Clip       : {:.2}", args.gclip);
    println!(
        "  • Acelerador          : {}",
        if args.gpu {
            "⚡ GPU (Vulkan / WGPU)"
        } else {
            "🖥️  CPU (AVX2/NEON Rayon)"
        }
    );

    println!("\n⏳ Abriendo archivo genómico y cargando modelo...");
    let (mut model, tokenizer) = repl::load_model_and_tokenizer(&args.model)?;
    let reader = GajeFlatFileReader::open(&args.model)?;
    let config = reader.load_config()?;

    println!(
        "✅ Modelo cargado ({} bloques, {} dim)",
        model.blocks.len(),
        config.n_embd
    );

    if args.gpu {
        let n_blocks = model.blocks.len();
        match model.offload_to_gpu(n_blocks) {
            Ok(assigned) => {
                println!(
                    "⚡ Acelerador GPU activado: {}/{} capas asignadas a GPU (Vulkan/WGPU)",
                    assigned, n_blocks
                );
            }
            Err(e) => {
                println!("⚠️ Advertencia GPU: {}. Continuando con fallback CPU.", e);
            }
        }
    }

    let mut memory_orch = _impl::compute::island::IslandOrchestrator::try_load_paired_memory(
        &args.model,
        config.n_embd as u32,
    )
    .or_else(|| {
        println!("🧠 Inicializando nuevo Hipocampo Congénito (.gmem) para el organismo...");
        Some(_impl::compute::island::IslandOrchestrator::new(
            config.n_embd as u32,
        ))
    });
    if let Some(ref orch) = memory_orch {
        let total = orch.documental.entries.len()
            + orch.episodic.entries.len()
            + orch.conversational.entries.len();
        println!(
            "🧠 Hipocampo Congénito activo: {} hechos vinculados a la crianza",
            total
        );
    }

    // Leer y parsear dataset
    println!("📖 Leyendo y tokenizando corpus...");
    let file = std::fs::File::open(&args.dataset)?;
    let lines = std::io::BufRead::lines(std::io::BufReader::new(file));
    let mut sequences: Vec<Vec<usize>> = Vec::new();
    let mut total_tokens = 0usize;
    let mut new_facts_added = 0usize;

    for line_res in lines {
        let line = line_res?;
        if line.trim().is_empty() {
            continue;
        }
        let text = if let Ok(val) = serde_json::from_str::<serde_json::Value>(&line) {
            val.get("text")
                .and_then(|v| v.as_str())
                .unwrap_or(&line)
                .to_string()
        } else {
            line
        };

        if let Some(ref mut orch) = memory_orch {
            let exists = orch.documental.entries.iter().any(|e| e.text == text);
            if !exists {
                let id = (orch.documental.entries.len() + 1) as u64;
                let vec = _impl::compute::island::IslandOrchestrator::vector_from_text(
                    &text,
                    config.n_embd,
                );
                orch.add_memory(
                    _impl::compute::island::IslandNiche::Documental,
                    id,
                    vec,
                    text.clone(),
                );
                new_facts_added += 1;
            }
        }

        if let Ok(tokens_u32) = tokenizer.encode(&text, false) {
            let mut tokens: Vec<usize> = tokens_u32.into_iter().map(|t| t as usize).collect();
            let vocab_limit = model.embeddings.out_features;
            tokens.retain(|&t| t < vocab_limit);
            if tokens.len() >= 2 {
                total_tokens += tokens.len();
                sequences.push(tokens);
            }
        }
    }

    if new_facts_added > 0 {
        println!(
            "🧠 [Hipocampo] +{} nuevos hechos integrados al nicho documental durante la lectura",
            new_facts_added
        );
    }

    println!(
        "📊 Corpus procesado: {} secuencias, {} tokens totales",
        sequences.len(),
        total_tokens
    );
    if sequences.is_empty() {
        return Err("El dataset no contiene secuencias válidas".into());
    }

    let total_blocks = model.blocks.len();
    let train_blocks = args.train_layers.unwrap_or(total_blocks).min(total_blocks);

    println!(
        "  • Capas en Crianza    : {}/{} bloques superiores + LM Head{}",
        train_blocks,
        total_blocks,
        if train_blocks < total_blocks { " (Ladder Training)" } else { " (Full Body)" }
    );

    println!("\n🔥 Iniciando bucle de entrenamiento STE cuaternario...");
    let t_start = std::time::Instant::now();
    let mut initial_loss = 0.0f32;
    let mut final_loss = 0.0f32;
    let n_seqs = sequences.len();

    use std::io::Write;
    for epoch in 1..=args.epochs {
        let t_ep = std::time::Instant::now();
        let mut ep_loss_sum = 0.0f32;
        let mut ep_toks = 0usize;

        for (seq_idx, seq) in sequences.iter().enumerate() {
            let loss = model.train_sequence_cached_layerwise_core(
                seq.clone(),
                args.lr,
                train_blocks,
                args.gclip,
                args.lr_decay,
                true,
                None,
            )?;
            let n_tok = seq.len().saturating_sub(1);
            ep_loss_sum += loss * n_tok as f32;
            ep_toks += n_tok;

            // Telemetría en vivo con vaciado explícito de búfer
            if (seq_idx + 1) % 10 == 0 || seq_idx + 1 == n_seqs {
                let elapsed = t_ep.elapsed().as_secs_f32().max(1e-4);
                let current_tps = (ep_toks as f32) / elapsed;
                let current_loss = ep_loss_sum / ep_toks.max(1) as f32;
                let remaining_seqs = n_seqs.saturating_sub(seq_idx + 1);
                let avg_time_per_seq = elapsed / (seq_idx + 1) as f32;
                let eta_secs = remaining_seqs as f32 * avg_time_per_seq;

                print!(
                    "\r  ⚡ [Época {:>2}/{}] Sec {:>3}/{} ({:>3.0}%) | Loss: {:.4} | {:.0} tok/s | ETA: {:>3.0}s   ",
                    epoch,
                    args.epochs,
                    seq_idx + 1,
                    n_seqs,
                    ((seq_idx + 1) as f32 / n_seqs as f32) * 100.0,
                    current_loss,
                    current_tps,
                    eta_secs,
                );
                std::io::stdout().flush().ok();
            }
        }

        let ep_avg_loss = ep_loss_sum / ep_toks.max(1) as f32;
        if epoch == 1 {
            initial_loss = ep_avg_loss;
        }
        final_loss = ep_avg_loss;

        let ep_dur = t_ep.elapsed().as_secs_f32().max(1e-4);
        let tps = (ep_toks as f32) / ep_dur;

        println!(
            "\r  • Época {:>2}/{} | Loss: {:.4} | {:.0} tok/s | Tiempo: {:.2}s                              ",
            epoch, args.epochs, ep_avg_loss, tps, ep_dur
        );
        std::io::stdout().flush().ok();
    }

    let total_time = t_start.elapsed();
    let delta_loss = initial_loss - final_loss;
    let red_pct = (delta_loss / initial_loss.max(1e-6)) * 100.0;

    println!("\n📈 Resumen de Crianza Genómica:");
    println!("  • Pérdida Inicial   : {:.4}", initial_loss);
    println!("  • Pérdida Final     : {:.4}", final_loss);
    println!(
        "  • Reducción Pérdida : {:.4} ({:.2}%)",
        delta_loss, red_pct
    );
    println!("  • Tiempo Total      : {:.2?}", total_time);

    let output_path = args.output.clone().unwrap_or_else(|| args.model.clone());
    println!("\n💾 Guardando organismo entrenado en: {}", output_path);
    save_genomic_flat_q(&output_path, &model, &config, None, 3)?;

    if let Some(ref mut orch) = memory_orch {
        let memory_dir = if output_path.ends_with(".gaje") {
            output_path.strip_suffix(".gaje").unwrap().to_string() + "_memory"
        } else if output_path.ends_with(".flat") {
            output_path.strip_suffix(".flat").unwrap().to_string() + "_memory"
        } else {
            format!("{}_memory", output_path)
        };
        orch.save_all(&memory_dir)?;
        let total = orch.documental.entries.len()
            + orch.episodic.entries.len()
            + orch.conversational.entries.len();
        println!(
            "🧠 Hipocampo Sincronizado: {} hechos consolidados en {}",
            total, memory_dir
        );
    }

    println!("✅ ¡Organismo y memoria guardados y listos para inferencia!");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    Ok(())
}

fn load_model_with_optional_tok(
    model_path: &str,
    custom_tok_path: Option<&str>,
) -> Result<
    (
        _impl::nn::llm::GenomicLLM,
        _impl::core::tokenizer::GajeTokenizer,
    ),
    Box<dyn std::error::Error + Send + Sync>,
> {
    use _impl::core::gtok::GtokNativeTokenizer;
    use _impl::core::tokenizer::GajeTokenizer;

    if let Some(tok_path) = custom_tok_path {
        let reader = _impl::io::flat_reader::GajeFlatFileReader::open(model_path)?;
        let model = reader.load_genomic()?;
        let tokenizer = if tok_path.ends_with(".bin") || tok_path.ends_with(".gtok") {
            let bytes = std::fs::read(tok_path)?;
            let gtok = GtokNativeTokenizer::from_bytes(&bytes)?;
            GajeTokenizer::from_gtok(gtok)
        } else {
            GajeTokenizer::from_file(tok_path).map_err(|e| e.to_string())?
        };
        Ok((model, tokenizer))
    } else {
        match repl::load_model_and_tokenizer(model_path) {
            Ok(pair) => Ok(pair),
            Err(e) => {
                if Path::new("data/gtok_human_4k.bin").exists() {
                    let bytes = std::fs::read("data/gtok_human_4k.bin")?;
                    let gtok = GtokNativeTokenizer::from_bytes(&bytes)?;
                    let reader = _impl::io::flat_reader::GajeFlatFileReader::open(model_path)?;
                    let model = reader.load_genomic()?;
                    Ok((model, GajeTokenizer::from_gtok(gtok)))
                } else if Path::new("models/core/tokenizer.json").exists() {
                    let reader = _impl::io::flat_reader::GajeFlatFileReader::open(model_path)?;
                    let model = reader.load_genomic()?;
                    let tokenizer = GajeTokenizer::from_file("models/core/tokenizer.json")
                        .map_err(|e| e.to_string())?;
                    Ok((model, tokenizer))
                } else {
                    Err(e.into())
                }
            }
        }
    }
}

fn handle_distill(args: &DistillArgs) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use _impl::io::flat_reader::GajeFlatFileReader;
    use _impl::io::flat_writer::save_genomic_flat_q;
    use _impl::nn::distiller::{CouncilOfTeachers, GenomicDistiller, Teacher};

    println!("\n🧬 GAJE PROTOCOLO DE DESTILACIÓN — DNI Online (Maestro ➔ Alumno)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  • Modelo Maestro     : {}", args.teacher);
    println!("  • Modelo Alumno      : {}", args.student);
    println!("  • Dataset Destilación: {}", args.dataset);
    println!("  • Épocas             : {}", args.epochs);
    println!("  • Learning Rate (lr) : {:.4}", args.lr);
    println!("  • Ponderación Alpha  : {:.2} (KL vs CE)", args.alpha);
    println!("  • Temperatura        : {:.2}", args.temperature);

    println!("\n⏳ Cargando modelo alumno...");
    let (mut student_model, student_tok) =
        load_model_with_optional_tok(&args.student, args.student_tokenizer.as_deref())?;
    let student_reader = GajeFlatFileReader::open(&args.student)?;
    let student_config = student_reader.load_config()?;
    let student_header = student_reader.header;

    println!(
        "✅ Alumno cargado ({} bloques, {} dim, vocab: {})",
        student_model.blocks.len(),
        student_config.n_embd,
        student_model.lm_head.out_features
    );

    println!("\n⏳ Cargando modelo maestro...");
    let teacher = if args.teacher.ends_with(".gguf") {
        #[cfg(feature = "native")]
        {
            let tok_path = args
                .teacher_tokenizer
                .as_deref()
                .unwrap_or("models/core/tokenizer.json");
            Teacher::new(
                "Maestro_GGUF".to_string(),
                &args.teacher,
                tok_path,
                &student_tok,
            )?
        }
        #[cfg(not(feature = "native"))]
        {
            return Err("Soporte GGUF requiere feature 'native'".into());
        }
    } else {
        let (teacher_model, teacher_tok) =
            load_model_with_optional_tok(&args.teacher, args.teacher_tokenizer.as_deref())?;
        Teacher::from_model(
            "Maestro_Flat".to_string(),
            teacher_model,
            teacher_tok,
            &student_tok,
        )
    };

    println!(
        "✅ Maestro listo (vocabulario mapeado, identidad: {})",
        teacher.is_identity_vocab
    );

    // Configurar consejo y destilador
    let mut council = CouncilOfTeachers::new();
    council.add_teacher(teacher);
    let mut distiller = GenomicDistiller::new(council, student_tok.clone());
    distiller.distill_weight = args.alpha;

    // Leer dataset de destilación
    println!("\n📖 Leyendo y preparando dataset de destilación...");
    let file = std::fs::File::open(&args.dataset)?;
    let lines = std::io::BufRead::lines(std::io::BufReader::new(file));
    let mut texts: Vec<String> = Vec::new();

    for line_res in lines {
        let line = line_res?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if let Ok(val) = serde_json::from_str::<serde_json::Value>(trimmed) {
            if let (Some(prompt), Some(resp)) = (
                val.get("prompt").and_then(|v| v.as_str()),
                val.get("response").and_then(|v| v.as_str()),
            ) {
                texts.push(format!(
                    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}<|im_end|>",
                    prompt, resp
                ));
            } else if let (Some(inst), Some(resp)) = (
                val.get("instruction").and_then(|v| v.as_str()),
                val.get("response").and_then(|v| v.as_str()),
            ) {
                texts.push(format!(
                    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}<|im_end|>",
                    inst, resp
                ));
            } else if let Some(t) = val.get("text").and_then(|v| v.as_str()) {
                texts.push(t.to_string());
            } else {
                texts.push(trimmed.to_string());
            }
        } else {
            texts.push(trimmed.to_string());
        }
    }

    println!("📊 Total de secuencias para destilación: {}", texts.len());
    if texts.is_empty() {
        return Err("El dataset no contiene secuencias válidas".into());
    }

    println!("\n🔥 Iniciando ciclo de destilación DNI...");
    let t_start = std::time::Instant::now();
    let mut initial_loss = 0.0f32;
    let mut final_loss = 0.0f32;

    for epoch in 1..=args.epochs {
        let t_ep = std::time::Instant::now();
        let mut ep_loss_sum = 0.0f32;
        let mut count = 0usize;

        for (idx, text) in texts.iter().enumerate() {
            match distiller.distill_step(&mut student_model, text, args.lr) {
                Ok(loss) => {
                    ep_loss_sum += loss;
                    count += 1;
                }
                Err(e) => {
                    if idx < 3 {
                        eprintln!("  [!] Advertencia en muestra {}: {}", idx, e);
                    }
                }
            }
        }

        let ep_avg_loss = if count > 0 {
            ep_loss_sum / count as f32
        } else {
            0.0
        };
        if epoch == 1 {
            initial_loss = ep_avg_loss;
        }
        final_loss = ep_avg_loss;

        let ep_dur = t_ep.elapsed().as_secs_f32().max(1e-4);
        println!(
            "  • Época {:>2}/{} | Loss: {:.4} | Muestras: {} | Tiempo: {:.2}s",
            epoch, args.epochs, ep_avg_loss, count, ep_dur
        );
    }

    let total_time = t_start.elapsed();
    let delta_loss = initial_loss - final_loss;
    let red_pct = if initial_loss > 1e-6 {
        (delta_loss / initial_loss) * 100.0
    } else {
        0.0
    };

    println!("\n📈 Resumen de Destilación DNI:");
    println!("  • Pérdida Inicial   : {:.4}", initial_loss);
    println!("  • Pérdida Final     : {:.4}", final_loss);
    println!(
        "  • Reducción Pérdida : {:.4} ({:.2}%)",
        delta_loss, red_pct
    );
    println!("  • Tiempo Total      : {:.2?}", total_time);

    let output_path = args.output.clone().unwrap_or_else(|| {
        if args.student.ends_with(".flat") {
            args.student.strip_suffix(".flat").unwrap().to_string() + "_distilled.flat"
        } else if args.student.ends_with(".gaje") {
            args.student.strip_suffix(".gaje").unwrap().to_string() + "_distilled.gaje"
        } else {
            format!("{}_distilled", args.student)
        }
    });

    println!("\n💾 Guardando alumno refinado en: {}", output_path);
    save_genomic_flat_q(
        &output_path,
        &student_model,
        &student_config,
        Some(&student_tok),
        student_header.quant_format,
    )?;

    println!("✅ ¡Modelo destilado guardado exitosamente!");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    Ok(())
}

fn handle_mutate(args: &MutateArgs) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model_path = resolve_default_model(args.model.clone());
    println!("\n🧬 GAJE ADAPTIVE — Mutación In-Place de Centroides (Codebook Tuning)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  • Modelo Objetivo   : {}", model_path);
    println!(
        "  • Tensor            : {}",
        args.tensor
            .as_deref()
            .unwrap_or("Automático (última capa / atención)")
    );
    println!("  • Intensidad (sigma): {}", args.intensity);

    let report = _impl::io::adaptive::mutate_model_centroids(
        Path::new(&model_path),
        args.tensor.as_deref(),
        args.intensity,
    )?;

    println!("\n✅ Mutación aplicada con éxito in-place:");
    println!("  • Tensor mutado       : {}", report.tensor_name);
    println!(
        "  • Centroides alterados: {} centroides float32",
        report.num_centroids
    );
    println!(
        "  • Muestra previa      : {:?}",
        report.old_centroids_sample
    );
    println!(
        "  • Muestra nueva       : {:?}",
        report.new_centroids_sample
    );
    println!("  • Mutación acumulada  : #{}", report.mutation_index);
    println!("  • Hash Ancestro       : {:#018x}", report.parent_hash);
    println!(
        "  • Hash Linaje Nuevo   : {:#018x}",
        report.new_lineage_hash
    );
    println!("\n💡 El modelo ha evolucionado directamente sobre su archivo sin duplicar gigabytes en disco.\n");
    Ok(())
}

fn handle_history(args: &HistoryArgs) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model_path = resolve_default_model(args.model.clone());
    let rep = _impl::io::adaptive::inspect_lineage(Path::new(&model_path))?;

    println!("\n🧬 GAJE LINEAGE — Árbol Genealógico y Estado Adaptativo");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  • Archivo del Modelo  : {}", rep.path);
    println!("  • Arquitectura        : {}", rep.arch_summary);
    println!("  • Cuantización        : {}", rep.quant_format);
    println!("  • Tensores Registrados: {}", rep.num_tensors);
    println!("  • Mutaciones Vivas    : {}", rep.num_mutations);
    println!(
        "  • Estado Adaptativo   : {}",
        if rep.has_adaptive_section {
            "ACTIVO (Organismo adaptado / en evolución)"
        } else {
            "PRÍSTINO (Modelo base congelado)"
        }
    );
    println!("  • Hash Padre (Parent) : {}", rep.parent_hash_hex);
    println!("  • Hash Linaje Actual  : {}", rep.current_hash_hex);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    Ok(())
}

fn print_cli_guide() {
    println!("\n🧬 GAJE HELIX — Guía Rápida y Mapa de Comandos Unificado");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Uso General: gaje-cli <COMANDO | ALIAS> [OPCIONES]");
    println!();
    println!("💬 1. INFERENCIA & DIÁLOGO:");
    println!("  • gaje-cli chat (alias: conversar, repl)");
    println!("      Inicia una sesión interactiva REPL en terminal con memoria y streaming.");
    println!("      Ej: gaje-cli chat -m models/born/max_512_pro.gaje");
    println!("  • gaje-cli serve (alias: servidor, server, web)");
    println!("      Lanza el servidor HTTP nativo con Web UI y streaming SSE.");
    println!("      Ej: gaje-cli serve --port 8080");
    println!("  • gaje-cli --prompt \"<texto>\" -m <modelo>");
    println!("      Inferencia directa de una sola consulta sin iniciar REPL.");
    println!();
    println!("🧬 2. GÉNESIS & CRIANZA GENÓMICA (2-Bits / Q2_0):");
    println!("  • gaje-cli birth (alias: nacimiento, genesis)");
    println!("      Da a luz a un organismo genómico nativo en el plano complejo ℂ.");
    println!("      Ej: gaje-cli birth --dim 512 --layers 12 -o models/born/nuevo.gaje");
    println!("  • gaje-cli train-born (alias: crianza, train, entrenar)");
    println!(
        "      Entrena un organismo nacido con el estimador Straight-Through cuaternario (STE)."
    );
    println!("      Ej: gaje-cli crianza -m models/born/max_512_pro.gaje -e 10 --gpu");
    println!("  • gaje-cli distill (alias: destilar, dni)");
    println!("      Destilación de conocimiento DNI Online (Maestro ➔ Alumno).");
    println!("      Ej: gaje-cli destilar -t models/production/gaje_nano_0_5b.flat -s models/born/max_512_pro.gaje -d data/genesis_conversational_corpus.jsonl");
    println!("  • gaje-cli mutate (alias: mutar)");
    println!(
        "      Inyecta mutación genómica controlada in-place en centroides sin duplicar archivo."
    );
    println!("  • gaje-cli history (alias: historial, lineage)");
    println!("      Inspecciona linaje evolutivo, árbol de mutaciones y estado adaptativo.");
    println!();
    println!("📦 3. GESTIÓN DE MODELOS & MEMORIA ASOCIATIVA (.gmem):");
    println!(
        "  • gaje-cli models (alias: modelos, ls) [list | inspect <archivo> | verify <archivo>]"
    );
    println!("      Inspecciona cabeceras binarias, verifica integridad y cataloga modelos.");
    println!("      Ej: gaje-cli modelos list");
    println!("      Ej: gaje-cli modelos inspect models/born/max_512_pro.gaje");
    println!("  • gaje-cli pull (alias: descargar, get, download)");
    println!("      Descarga modelos optimizados multi-stream desde Hugging Face.");
    println!("      Ej: gaje-cli descargar eaguilar/gaje-models/max_512_pro.gaje");
    println!("  • gaje-cli epoch (alias: memoria, memory) [status | consolidate | prune]");
    println!("      Inspecciona y consolida el Hipocampo Congénito .gmem.");
    println!("      Ej: gaje-cli memoria status -m models/born/max_512_pro.gaje");
    println!();
    println!("🚀 4. RENDIMIENTO, CALIBRACIÓN & AUDITORÍA:");
    println!("  • gaje-cli doctor (alias: diagnostico, diag, info)");
    println!(
        "      Diagnóstico integral de hardware: CPU, SIMD (AVX2/NEON), GPU Vulkan y memoria."
    );
    println!("  • gaje-cli benchmark (alias: bench, medir, test-speed)");
    println!("      Mide latencia TTFT, velocidad sostenida (tok/s) y consumo de memoria RAM.");
    println!("      Ej: gaje-cli bench -m models/born/max_512_pro.gaje");
    println!("  • gaje-cli audit (alias: auditar, check)");
    println!("      Auditoría matemática de ausencia de NaNs, entropía y ortogonalidad.");
    println!("  • gaje-cli dataset-build (alias: dataset, datos)");
    println!("      Construye y tokeniza datasets de texto/JSONL para entrenamiento.");
    println!("  • gaje-cli guide (alias: menu, comandos, guia, ayuda)");
    println!("      Muestra este mapa de comandos y ejemplos de uso.");
    println!(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    );
}
