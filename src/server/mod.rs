pub mod api;
pub mod static_files;
pub mod streaming;

use crate::core::tokenizer::GajeTokenizer;
use crate::nn::llm::GenomicLLM;
use crate::nn::repl::load_model_and_tokenizer;
use std::fs::File;
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
    pub path: PathBuf,
    pub llm: GenomicLLM,
    pub tokenizer: GajeTokenizer,
}

pub fn find_model_path(models_root: &Path, model_name: &str) -> Option<PathBuf> {
    let clean_name = Path::new(model_name).file_name()?.to_str()?;
    let search_dirs = [
        models_root.join("production"),
        models_root.join("born"),
        models_root.to_path_buf(),
    ];

    for dir in &search_dirs {
        if dir.exists() {
            let candidate = dir.join(clean_name);
            if candidate.exists() && candidate.is_file() {
                return Some(candidate);
            }
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_file() && path.file_name().map(|n| n == clean_name).unwrap_or(false)
                    {
                        return Some(path);
                    }
                }
            }
        }
    }
    None
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

    println!(
        "\n🧬 ==============================================================================="
    );
    println!("🌐 GAJE HELIX — Servidor HTTP Nativo de Producción (Zero-Python Runtime)");
    println!("===============================================================================");
    println!("🚀 Escuchando en:         http://{}", addr);
    println!("📁 Directorio de Modelos: {:?}", config.models_dir);
    println!("🎨 Directorio Web UI:     {:?}", config.static_dir);
    if config.chat_only {
        println!("📱 Modo:                  Móvil / Ultra-Ligero (--chat-only activo)");
    }

    let active_model: Arc<RwLock<Option<LoadedModel>>> = Arc::new(RwLock::new(None));

    // Precargar modelo inicial si se especificó o si existe max.gaje, gaje_coder_3b o gaje_pico
    let model_to_load = config.initial_model.clone().unwrap_or_else(|| {
        let max_born = config.models_dir.join("born/max.gaje");
        let coder = config.models_dir.join("production/gaje_coder_3b.flat");
        let pico = config.models_dir.join("production/gaje_pico_135m.flat");
        if max_born.exists() {
            max_born.to_string_lossy().to_string()
        } else if coder.exists() {
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
                    path: PathBuf::from(&model_to_load),
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
        let request = match server_arc.recv_timeout(std::time::Duration::from_millis(200)) {
            Ok(Some(rq)) => rq,
            Ok(None) => continue, // Timeout cada 200ms para chequear la señal de terminación (Ctrl+C)
            Err(_) => break,
        };

        let url = request.url().to_string();
        let method = request.method().clone();

        // 1. CORS Preflight
        if method == Method::Options {
            let mut resp = Response::empty(StatusCode(204));
            resp.add_header(
                Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap(),
            );
            resp.add_header(
                Header::from_bytes(
                    &b"Access-Control-Allow-Methods"[..],
                    &b"GET, POST, OPTIONS"[..],
                )
                .unwrap(),
            );
            resp.add_header(
                Header::from_bytes(&b"Access-Control-Allow-Headers"[..], &b"Content-Type"[..])
                    .unwrap(),
            );
            let _ = request.respond(resp);
            continue;
        }

        // 2. Endpoints API
        if url == "/api/models" && method == Method::Get {
            let active_name = active_model
                .read()
                .unwrap()
                .as_ref()
                .map(|m| m.name.clone());
            let json_val = api::get_available_models(&config.models_dir, active_name.as_deref());
            let body = json_val.to_string();
            let mut resp = Response::from_string(body).with_status_code(StatusCode(200));
            resp.add_header(
                Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap(),
            );
            resp.add_header(
                Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap(),
            );
            let _ = request.respond(resp);
            continue;
        }

        if url == "/api/info" && method == Method::Get {
            let active_name = active_model
                .read()
                .unwrap()
                .as_ref()
                .map(|m| m.name.clone());
            let json_val = api::get_runtime_info(active_name.as_deref());
            let body = json_val.to_string();
            let mut resp = Response::from_string(body).with_status_code(StatusCode(200));
            resp.add_header(
                Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap(),
            );
            resp.add_header(
                Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap(),
            );
            let _ = request.respond(resp);
            continue;
        }

        if url == "/api/memory" && method == Method::Get {
            let active_guard = active_model.read().unwrap();
            let (active_path, active_dim) = if let Some(ref m) = *active_guard {
                (Some(m.path.to_string_lossy().to_string()), m.llm.dim())
            } else {
                (None, 384)
            };
            let json_val = api::get_memory_info(active_path.as_deref(), active_dim);
            let body = json_val.to_string();
            let mut resp = Response::from_string(body).with_status_code(StatusCode(200));
            resp.add_header(
                Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap(),
            );
            resp.add_header(
                Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap(),
            );
            let _ = request.respond(resp);
            continue;
        }

        if url == "/api/load_model" && method == Method::Post {
            let mut body = String::new();
            let mut req = request;
            let _ = req.as_reader().read_to_string(&mut body);
            let req_data: serde_json::Value = serde_json::from_str(&body).unwrap_or_default();
            let requested_name = req_data.get("model").and_then(|v| v.as_str()).unwrap_or("");

            if let Some(model_path) = find_model_path(&config.models_dir, requested_name) {
                // Liberar el modelo previo antes de abrir el nuevo para evitar solapamiento de memoria en RAM
                *active_model.write().unwrap() = None;

                println!("🧬 [Carga Dinámica] Cargando modelo: {:?}", model_path);
                match load_model_and_tokenizer(&model_path.to_string_lossy()) {
                    Ok((llm, tokenizer)) => {
                        let name = model_path
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .to_string();
                        *active_model.write().unwrap() = Some(LoadedModel {
                            name: name.clone(),
                            path: model_path,
                            llm,
                            tokenizer,
                        });
                        let json_resp = serde_json::json!({ "status": "ok", "model": name });
                        let mut resp = Response::from_string(json_resp.to_string())
                            .with_status_code(StatusCode(200));
                        resp.add_header(
                            Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..])
                                .unwrap(),
                        );
                        resp.add_header(
                            Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..])
                                .unwrap(),
                        );
                        let _ = req.respond(resp);
                    }
                    Err(e) => {
                        let err_json = serde_json::json!({ "error": format!("Error al cargar modelo: {}", e) });
                        let mut resp = Response::from_string(err_json.to_string())
                            .with_status_code(StatusCode(500));
                        resp.add_header(
                            Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..])
                                .unwrap(),
                        );
                        let _ = req.respond(resp);
                    }
                }
            } else {
                let err_json = serde_json::json!({ "error": format!("Modelo '{}' no encontrado", requested_name) });
                let mut resp =
                    Response::from_string(err_json.to_string()).with_status_code(StatusCode(404));
                resp.add_header(
                    Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap(),
                );
                let _ = req.respond(resp);
            }
            continue;
        }

        if url == "/api/unload_model" && method == Method::Post {
            *active_model.write().unwrap() = None;
            let json_resp = serde_json::json!({ "status": "ok", "unloaded": true });
            let mut resp =
                Response::from_string(json_resp.to_string()).with_status_code(StatusCode(200));
            resp.add_header(
                Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap(),
            );
            resp.add_header(
                Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap(),
            );
            let _ = request.respond(resp);
            continue;
        }

        // 3. Servir Modelos Binarios para Descarga o WASM (`/models/*`) con streaming zero-copy
        if url.starts_with("/models/") && method == Method::Get {
            let rel_path = url
                .trim_start_matches("/models/")
                .split('?')
                .next()
                .unwrap_or("");
            if let Some(target_path) = find_model_path(&config.models_dir, rel_path) {
                if let Ok(f) = File::open(&target_path) {
                    let mut resp = Response::from_file(f).with_status_code(StatusCode(200));
                    resp.add_header(
                        Header::from_bytes(&b"Content-Type"[..], &b"application/octet-stream"[..])
                            .unwrap(),
                    );
                    resp.add_header(
                        Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap(),
                    );
                    let _ = request.respond(resp);
                    continue;
                }
            }
            let resp = Response::from_string(format!("Modelo '{}' no encontrado", rel_path))
                .with_status_code(StatusCode(404));
            let _ = request.respond(resp);
            continue;
        }

        if (url == "/api/chat/stream" || url.starts_with("/api/chat/stream"))
            && (method == Method::Post || method == Method::Get)
        {
            let mut guard = active_model.write().unwrap();
            if let Some(ref mut loaded) = *guard {
                let _ = streaming::handle_chat_stream_request(
                    request,
                    &mut loaded.llm,
                    &loaded.tokenizer,
                );
            } else {
                let err_json =
                    serde_json::json!({ "error": "No hay ningún modelo cargado en el servidor." });
                let mut resp =
                    Response::from_string(err_json.to_string()).with_status_code(StatusCode(503));
                resp.add_header(
                    Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap(),
                );
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
                let chat_req: streaming::ChatRequest =
                    serde_json::from_str(&body).unwrap_or(streaming::ChatRequest {
                        message: Some(body),
                        model: None,
                        history: None,
                        system_prompt: None,
                        max_tokens: Some(256),
                        temperature: Some(0.3),
                        top_p: Some(0.9),
                        repetition_penalty: Some(1.15),
                    });

                let prompt = chat_req.message.unwrap_or_default();
                let sys_prompt = chat_req.system_prompt.unwrap_or_else(|| {
                    "Tu nombre es GAJE. Eres un asistente de inteligencia artificial avanzado, servicial, conciso y preciso.".to_string()
                });
                let chat_prompt = format!(
                    "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                    sys_prompt, prompt
                );
                let prompt_tokens = loaded
                    .tokenizer
                    .encode(&chat_prompt, false)
                    .unwrap_or_default();
                let prompt_tokens_usize: Vec<usize> =
                    prompt_tokens.into_iter().map(|t| t as usize).collect();
                let eos_ids = vec![0, 2, 151643, 151644, 151645];
                let gen = loaded
                    .llm
                    .generate_native_core(
                        prompt_tokens_usize,
                        chat_req.max_tokens.unwrap_or(256),
                        chat_req.temperature.unwrap_or(0.4),
                        chat_req.repetition_penalty.unwrap_or(1.15),
                        eos_ids,
                    )
                    .unwrap_or_default();
                let gen_u32: Vec<u32> = gen.into_iter().map(|t| t as u32).collect();
                let reply = loaded.tokenizer.decode(&gen_u32, true).unwrap_or_default();
                let clean = reply
                    .replace("<|im_end|>", "")
                    .replace("<|im_start|>", "")
                    .replace("<|endoftext|>", "")
                    .trim()
                    .to_string();

                let json_resp = serde_json::json!({
                    "response": clean,
                    "status": "ok",
                    "model": loaded.name
                });
                let mut resp =
                    Response::from_string(json_resp.to_string()).with_status_code(StatusCode(200));
                resp.add_header(
                    Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap(),
                );
                let _ = req.respond(resp);
            } else {
                let err_json = serde_json::json!({ "error": "No hay ningún modelo cargado." });
                let mut resp =
                    Response::from_string(err_json.to_string()).with_status_code(StatusCode(503));
                resp.add_header(
                    Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap(),
                );
                let _ = request.respond(resp);
            }
            continue;
        }

        // 4. Servir Archivos Estáticos de la Web UI
        if let Some(resp) =
            static_files::serve_static_file(&config.static_dir, &url, config.chat_only)
        {
            let _ = request.respond(resp);
        } else {
            let resp = Response::from_string("404 Not Found").with_status_code(StatusCode(404));
            let _ = request.respond(resp);
        }
    }

    println!("\n🛑 [GAJE-SERVER] Servidor HTTP finalizado de forma limpia.");
    Ok(())
}
