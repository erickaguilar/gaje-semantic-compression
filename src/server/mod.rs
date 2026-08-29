pub mod api;
pub mod static_files;
pub mod streaming;

use crate::core::tokenizer::GajeTokenizer;
use crate::nn::llm::GenomicLLM;
use crate::nn::repl::load_model_and_tokenizer;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use tiny_http::{Header, Method, Response, Server, StatusCode};

pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub models_dir: PathBuf,
    pub static_dir: PathBuf,
    pub initial_model: Option<String>,
    pub chat_only: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            models_dir: PathBuf::from("models"),
            static_dir: PathBuf::from("examples/ui/web_ui"),
            initial_model: None,
            chat_only: false,
        }
    }
}

pub struct LoadedModel {
    pub name: String,
    pub llm: GenomicLLM,
    pub tokenizer: GajeTokenizer,
}

pub fn run_server(
    config: ServerConfig,
    running: Arc<AtomicBool>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let addr = format!("{}:{}", config.host, config.port);
    let server = match Server::http(&addr) {
        Ok(s) => s,
        Err(e) => {
            return Err(format!("No se pudo abrir el servidor HTTP en {}: {}", addr, e).into());
        }
    };

    println!("\n🧬 ===============================================================================");
    println!("🌐 GAJE HELIX — Servidor HTTP Nativo de Producción (Zero-Python Runtime)");
    println!("===============================================================================");
    println!("🚀 Escuchando en:         http://{}", addr);
    println!("📁 Directorio de Modelos: {:?}", config.models_dir);
    println!("🎨 Directorio Web UI:     {:?}", config.static_dir);
    if config.chat_only {
        println!("📱 Modo:                  Móvil / Ultra-Ligero (--chat-only activo)");
    }

    let active_model: Arc<RwLock<Option<LoadedModel>>> = Arc::new(RwLock::new(None));

    // Precargar modelo inicial si se especificó o si existe gaje_coder_3b o gaje_pico
    let model_to_load = config.initial_model.clone().unwrap_or_else(|| {
        let coder = config.models_dir.join("production/gaje_coder_3b.flat");
        let pico = config.models_dir.join("production/gaje_pico_135m.flat");
        if coder.exists() {
            coder.to_string_lossy().to_string()
        } else if pico.exists() {
            pico.to_string_lossy().to_string()
        } else {
            String::new()
        }
    });

    if !model_to_load.is_empty() && Path::new(&model_to_load).exists() {
        println!("📦 Precargando organismo activo: {}...", model_to_load);
        match load_model_and_tokenizer(&model_to_load) {
            Ok((llm, tokenizer)) => {
                let name = Path::new(&model_to_load)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                *active_model.write().unwrap() = Some(LoadedModel {
                    name,
                    llm,
                    tokenizer,
                });
                println!("✅ Organismo listo con mapeo mmap zero-copy.");
            }
            Err(e) => {
                eprintln!("⚠️ Aviso al precargar modelo: {}", e);
            }
        }
    }

    println!("\n✨ Servidor listo para recibir tráfico web y streaming SSE.\n");

    let server_arc = Arc::new(server);

    while running.load(Ordering::SeqCst) {
        let request = match server_arc.recv() {
            Ok(rq) => rq,
            Err(_) => break,
        };

        let url = request.url().to_string();
        let method = request.method().clone();

        // 1. CORS Preflight
        if method == Method::Options {
            let mut resp = Response::empty(StatusCode(204));
            resp.add_header(Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap());
            resp.add_header(Header::from_bytes(&b"Access-Control-Allow-Methods"[..], &b"GET, POST, OPTIONS"[..]).unwrap());
            resp.add_header(Header::from_bytes(&b"Access-Control-Allow-Headers"[..], &b"Content-Type"[..]).unwrap());
            let _ = request.respond(resp);
            continue;
        }

        // 2. Endpoints API
        if url == "/api/models" && method == Method::Get {
            let active_name = active_model.read().unwrap().as_ref().map(|m| m.name.clone());
            let json_val = api::get_available_models(&config.models_dir, active_name.as_deref());
            let body = json_val.to_string();
            let mut resp = Response::from_string(body).with_status_code(StatusCode(200));
            resp.add_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
            resp.add_header(Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap());
            let _ = request.respond(resp);
            continue;
        }

        if url == "/api/info" && method == Method::Get {
            let active_name = active_model.read().unwrap().as_ref().map(|m| m.name.clone());
            let json_val = api::get_runtime_info(active_name.as_deref());
            let body = json_val.to_string();
            let mut resp = Response::from_string(body).with_status_code(StatusCode(200));
            resp.add_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
            resp.add_header(Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap());
            let _ = request.respond(resp);
            continue;
        }

        if (url == "/api/chat/stream" || url.starts_with("/api/chat/stream")) && (method == Method::Post || method == Method::Get) {
            let mut guard = active_model.write().unwrap();
            if let Some(ref mut loaded) = *guard {
                let _ = streaming::handle_chat_stream_request(request, &mut loaded.llm, &loaded.tokenizer);
            } else {
                let err_json = serde_json::json!({ "error": "No hay ningún modelo cargado en el servidor." });
                let mut resp = Response::from_string(err_json.to_string()).with_status_code(StatusCode(503));
                resp.add_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
                let _ = request.respond(resp);
            }
            continue;
        }

        if url == "/api/chat" && method == Method::Post {
            let mut guard = active_model.write().unwrap();
            if let Some(ref mut loaded) = *guard {
                let mut body = String::new();
                let mut req = request;
                let _ = req.as_reader().read_to_string(&mut body);
                let chat_req: streaming::ChatRequest = serde_json::from_str(&body).unwrap_or(streaming::ChatRequest {
                    message: Some(body),
                    model: None,
                    history: None,
                    system_prompt: None,
                    max_tokens: Some(256),
                    temperature: Some(0.4),
                    top_p: Some(0.9),
                    repetition_penalty: Some(1.15),
                });

                let prompt = chat_req.message.unwrap_or_default();
                let chat_prompt = format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", prompt);
                let prompt_tokens = loaded.tokenizer.encode(&chat_prompt, false).unwrap_or_default();
                let prompt_tokens_usize: Vec<usize> = prompt_tokens.into_iter().map(|t| t as usize).collect();
                let eos_ids = vec![2, 0];
                let gen = loaded.llm.generate_native_core(prompt_tokens_usize, chat_req.max_tokens.unwrap_or(256), chat_req.temperature.unwrap_or(0.4), chat_req.repetition_penalty.unwrap_or(1.15), eos_ids).unwrap_or_default();
                let gen_u32: Vec<u32> = gen.into_iter().map(|t| t as u32).collect();
                let reply = loaded.tokenizer.decode(&gen_u32, true).unwrap_or_default();
                let clean = reply.replace("<|im_end|>", "").replace("<|im_start|>", "").replace("<|endoftext|>", "").trim().to_string();

                let json_resp = serde_json::json!({
                    "response": clean,
                    "status": "ok",
                    "model": loaded.name
                });
                let mut resp = Response::from_string(json_resp.to_string()).with_status_code(StatusCode(200));
                resp.add_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
                let _ = req.respond(resp);
            } else {
                let err_json = serde_json::json!({ "error": "No hay ningún modelo cargado." });
                let mut resp = Response::from_string(err_json.to_string()).with_status_code(StatusCode(503));
                resp.add_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
                let _ = request.respond(resp);
            }
            continue;
        }

        // 3. Servir Archivos Estáticos de la Web UI
        if let Some(resp) = static_files::serve_static_file(&config.static_dir, &url, config.chat_only) {
            let _ = request.respond(resp);
        } else {
            let resp = Response::from_string("404 Not Found").with_status_code(StatusCode(404));
            let _ = request.respond(resp);
        }
    }

    Ok(())
}
