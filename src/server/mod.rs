pub mod api;
pub mod mcp;
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
    pub memory: crate::compute::island::IslandOrchestrator,
    pub memory_threshold: f32,
    pub memory_uses_whitening: bool,
    pub memory_mu: Option<Vec<f32>>,
    pub memory_whitening_missing: bool,
    pub memory_dir: PathBuf,
}

impl LoadedModel {
    pub fn memory_config(
        &self,
        use_memory: bool,
        custom_threshold: Option<f32>,
    ) -> crate::compute::island::MemoryConfig {
        crate::compute::island::MemoryConfig {
            use_memory,
            threshold: custom_threshold.unwrap_or(self.memory_threshold),
            uses_whitening: self.memory_uses_whitening,
            mu_vector: self.memory_mu.clone(),
            whitening_missing: self.memory_whitening_missing,
            max_tokens_context: 128,
        }
    }
}

pub fn calibrate_memory_threshold(name_or_path: &str, dim: usize) -> f32 {
    crate::compute::island::configure_model_memory(Path::new(name_or_path), dim).0
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
        let coder_gaje = config.models_dir.join("production/gaje_coder_3b.gaje");
        let coder_flat = config.models_dir.join("production/gaje_coder_3b.flat");
        let pico_gaje = config.models_dir.join("production/gaje_pico_135m.gaje");
        let pico_flat = config.models_dir.join("production/gaje_pico_135m.flat");
        if max_born.exists() {
            max_born.to_string_lossy().to_string()
        } else if coder_gaje.exists() {
            coder_gaje.to_string_lossy().to_string()
        } else if coder_flat.exists() {
            coder_flat.to_string_lossy().to_string()
        } else if pico_gaje.exists() {
            pico_gaje.to_string_lossy().to_string()
        } else if pico_flat.exists() {
            pico_flat.to_string_lossy().to_string()
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
                let dim = llm.dim() as u32;
                let path = PathBuf::from(&model_to_load);
                let (memory_threshold, memory_uses_whitening, memory_mu, memory_whitening_missing) =
                    crate::compute::island::configure_model_memory(&path, dim as usize);
                let (memory, memory_dir) = crate::compute::island::IslandOrchestrator::load_paired_for_model(
                    &path,
                    dim,
                    memory_threshold,
                );
                *active_model.write().unwrap() = Some(LoadedModel {
                    name,
                    path,
                    llm,
                    tokenizer,
                    memory,
                    memory_threshold,
                    memory_uses_whitening,
                    memory_mu,
                    memory_whitening_missing,
                    memory_dir,
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
                    &b"GET, POST, DELETE, OPTIONS"[..],
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
            let json_val = if let Some(ref m) = *active_guard {
                let mu_path = crate::compute::island::resolve_mu_vector_path(&m.path);
                api::get_loaded_memory_info(
                    &m.memory,
                    &m.memory_dir,
                    m.memory_threshold,
                    m.memory_uses_whitening,
                    m.memory_whitening_missing,
                    mu_path.as_deref(),
                )
            } else {
                api::get_memory_info(None, 384)
            };
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

        if url == "/api/memory/remember" && method == Method::Post {
            let mut body = String::new();
            let mut req = request;
            let _ = req.as_reader().read_to_string(&mut body);

            // T5: Límite de payload > 8192 bytes
            if body.len() > 8192 {
                let mut resp = Response::from_string(
                    serde_json::json!({
                        "error": "Payload excede el límite máximo de 8192 bytes"
                    }).to_string()
                ).with_status_code(StatusCode(400));
                resp.add_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
                resp.add_header(Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap());
                let _ = req.respond(resp);
                continue;
            }

            let req_data: serde_json::Value = match serde_json::from_str(&body) {
                Ok(v) => v,
                Err(_) => {
                    let mut resp = Response::from_string(
                        serde_json::json!({ "error": "JSON malformado" }).to_string()
                    ).with_status_code(StatusCode(400));
                    resp.add_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
                    resp.add_header(Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap());
                    let _ = req.respond(resp);
                    continue;
                }
            };

            let text = req_data.get("text").and_then(|v| v.as_str()).unwrap_or("").trim();
            if text.is_empty() {
                let mut resp = Response::from_string(
                    serde_json::json!({ "error": "El campo 'text' es obligatorio" }).to_string()
                ).with_status_code(StatusCode(400));
                resp.add_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
                resp.add_header(Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap());
                let _ = req.respond(resp);
                continue;
            }

            // T6: Anti path traversal
            let raw_niche = req_data.get("niche").and_then(|v| v.as_str()).unwrap_or("auto");
            if text.contains("..") || raw_niche.contains("..") {
                let mut resp = Response::from_string(
                    serde_json::json!({ "error": "Secuencia no permitida ('..') detectada" }).to_string()
                ).with_status_code(StatusCode(400));
                resp.add_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
                resp.add_header(Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap());
                let _ = req.respond(resp);
                continue;
            }

            let target_niche = match raw_niche.to_lowercase().as_str() {
                "documental" | "document" => crate::compute::island::IslandNiche::Documental,
                "episodic" => crate::compute::island::IslandNiche::Episodic,
                "conversational" => crate::compute::island::IslandNiche::Conversational,
                "auto" => crate::compute::island::IslandNiche::Documental,
                _ => {
                    let mut resp = Response::from_string(
                        serde_json::json!({
                            "error": "Nicho inválido. Valores permitidos: 'documental', 'episodic', 'conversational', 'auto'"
                        }).to_string()
                    ).with_status_code(StatusCode(400));
                    resp.add_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
                    resp.add_header(Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap());
                    let _ = req.respond(resp);
                    continue;
                }
            };

            let mut active_guard = active_model.write().unwrap();
            let m = match active_guard.as_mut() {
                Some(m) => m,
                None => {
                    let mut resp = Response::from_string(
                        serde_json::json!({ "error": "No hay ningún modelo activo cargado en el servidor" }).to_string()
                    ).with_status_code(StatusCode(400));
                    resp.add_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
                    resp.add_header(Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap());
                    let _ = req.respond(resp);
                    continue;
                }
            };

            // Whitening estricto: rechazo con 400 si falta el vector de calibración
            if m.memory_uses_whitening && m.memory_whitening_missing {
                let mut resp = Response::from_string(
                    serde_json::json!({
                        "error": "El modelo activo requiere vector de blanqueamiento (.mu.bin) para proyectar recuerdos homogéneos y no fue encontrado. Operación cancelada para prevenir contaminación del espacio latente."
                    }).to_string()
                ).with_status_code(StatusCode(400));
                resp.add_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
                resp.add_header(Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap());
                let _ = req.respond(resp);
                continue;
            }

            let expected_dim = m.llm.dim();
            let raw_vec = match m.llm.embed_text(text, &m.tokenizer) {
                Ok(v) => v,
                Err(e) => {
                    let mut resp = Response::from_string(
                        serde_json::json!({ "error": format!("Error extrayendo embedding: {}", e) }).to_string()
                    ).with_status_code(StatusCode(500));
                    resp.add_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
                    resp.add_header(Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap());
                    let _ = req.respond(resp);
                    continue;
                }
            };

            // Aplicar blanqueamiento si está configurado y mu_vector está presente
            let final_vector = match (&m.memory_mu, m.memory_uses_whitening) {
                (Some(mu), true) if mu.len() == expected_dim => {
                    let mut centered = vec![0.0f32; expected_dim];
                    let mut norm_sq = 0.0f32;
                    for i in 0..expected_dim {
                        let diff = raw_vec[i] - mu[i];
                        centered[i] = diff;
                        norm_sq += diff * diff;
                    }
                    let norm = norm_sq.sqrt().max(1e-8);
                    for val in centered.iter_mut() {
                        *val /= norm;
                    }
                    centered
                }
                _ => raw_vec,
            };

            let id = m.memory.generate_next_id();
            m.memory.add_memory(target_niche, id, final_vector, text.to_string());
            if let Err(e) = m.memory.save_all(&m.memory_dir.to_string_lossy()) {
                let mut resp = Response::from_string(
                    serde_json::json!({ "error": format!("Error persistiendo memoria en disco: {}", e) }).to_string()
                ).with_status_code(StatusCode(500));
                resp.add_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
                resp.add_header(Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap());
                let _ = req.respond(resp);
                continue;
            }

            let total_facts = m.memory.documental.entries.len()
                + m.memory.episodic.entries.len()
                + m.memory.conversational.entries.len();

            let resp_json = serde_json::json!({
                "status": "ok",
                "id": id,
                "niche": target_niche.as_str(),
                "total_facts": total_facts,
                "text": text,
            });
            let mut resp = Response::from_string(resp_json.to_string()).with_status_code(StatusCode(200));
            resp.add_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
            resp.add_header(Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap());
            let _ = req.respond(resp);
            continue;
        }

        if url == "/api/memory/entry" && method == Method::Delete {
            let mut body = String::new();
            let mut req = request;
            let _ = req.as_reader().read_to_string(&mut body);

            let req_data: serde_json::Value = match serde_json::from_str(&body) {
                Ok(v) => v,
                Err(_) => {
                    let mut resp = Response::from_string(
                        serde_json::json!({ "error": "JSON malformado" }).to_string()
                    ).with_status_code(StatusCode(400));
                    resp.add_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
                    resp.add_header(Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap());
                    let _ = req.respond(resp);
                    continue;
                }
            };

            let id = match req_data.get("id").and_then(|v| v.as_u64()) {
                Some(id) => id,
                None => {
                    let mut resp = Response::from_string(
                        serde_json::json!({ "error": "El campo 'id' numérico es obligatorio" }).to_string()
                    ).with_status_code(StatusCode(400));
                    resp.add_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
                    resp.add_header(Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap());
                    let _ = req.respond(resp);
                    continue;
                }
            };

            let niche_opt = req_data.get("niche").and_then(|v| v.as_str()).and_then(|s| {
                match s.to_lowercase().as_str() {
                    "documental" => Some(crate::compute::island::IslandNiche::Documental),
                    "episodic" => Some(crate::compute::island::IslandNiche::Episodic),
                    "conversational" => Some(crate::compute::island::IslandNiche::Conversational),
                    _ => None,
                }
            });

            let mut active_guard = active_model.write().unwrap();
            let m = match active_guard.as_mut() {
                Some(m) => m,
                None => {
                    let mut resp = Response::from_string(
                        serde_json::json!({ "error": "No hay ningún modelo activo cargado en el servidor" }).to_string()
                    ).with_status_code(StatusCode(400));
                    resp.add_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
                    resp.add_header(Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap());
                    let _ = req.respond(resp);
                    continue;
                }
            };

            let removed = m.memory.remove_memory(niche_opt, id);
            if !removed {
                let mut resp = Response::from_string(
                    serde_json::json!({ "error": format!("Recuerdo con id {} no encontrado", id) }).to_string()
                ).with_status_code(StatusCode(404));
                resp.add_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
                resp.add_header(Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap());
                let _ = req.respond(resp);
                continue;
            }

            if let Err(e) = m.memory.save_all(&m.memory_dir.to_string_lossy()) {
                let mut resp = Response::from_string(
                    serde_json::json!({ "error": format!("Error persistiendo memoria tras borrado: {}", e) }).to_string()
                ).with_status_code(StatusCode(500));
                resp.add_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
                resp.add_header(Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap());
                let _ = req.respond(resp);
                continue;
            }

            let total_facts = m.memory.documental.entries.len()
                + m.memory.episodic.entries.len()
                + m.memory.conversational.entries.len();

            let resp_json = serde_json::json!({
                "status": "ok",
                "removed": true,
                "id": id,
                "total_facts": total_facts,
            });
            let mut resp = Response::from_string(resp_json.to_string()).with_status_code(StatusCode(200));
            resp.add_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
            resp.add_header(Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap());
            let _ = req.respond(resp);
            continue;
        }

        if url == "/api/memory/consolidate" && method == Method::Post {
            let mut body = String::new();
            let mut req = request;
            let _ = req.as_reader().read_to_string(&mut body);

            let req_data: serde_json::Value = if body.trim().is_empty() {
                serde_json::json!({})
            } else {
                match serde_json::from_str(&body) {
                    Ok(v) => v,
                    Err(_) => {
                        let mut resp = Response::from_string(
                            serde_json::json!({ "error": "JSON malformado" }).to_string()
                        ).with_status_code(StatusCode(400));
                        resp.add_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
                        resp.add_header(Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap());
                        let _ = req.respond(resp);
                        continue;
                    }
                }
            };

            let dedup_threshold = req_data.get("dedup_threshold")
                .and_then(|v| v.as_f64())
                .map(|f| f as f32)
                .unwrap_or(0.97);

            if dedup_threshold < 0.5 || dedup_threshold > 1.0 {
                let mut resp = Response::from_string(
                    serde_json::json!({ "error": "El threshold de consolidación debe estar en el rango [0.5, 1.0]" }).to_string()
                ).with_status_code(StatusCode(400));
                resp.add_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
                resp.add_header(Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap());
                let _ = req.respond(resp);
                continue;
            }

            let mut active_guard = active_model.write().unwrap();
            let m = match active_guard.as_mut() {
                Some(m) => m,
                None => {
                    let mut resp = Response::from_string(
                        serde_json::json!({ "error": "No hay ningún modelo activo cargado en el servidor" }).to_string()
                    ).with_status_code(StatusCode(400));
                    resp.add_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
                    resp.add_header(Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap());
                    let _ = req.respond(resp);
                    continue;
                }
            };

            let stats = m.memory.consolidate_memory(dedup_threshold);
            if let Err(e) = m.memory.save_all(&m.memory_dir.to_string_lossy()) {
                let mut resp = Response::from_string(
                    serde_json::json!({ "error": format!("Error persistiendo memoria tras consolidación: {}", e) }).to_string()
                ).with_status_code(StatusCode(500));
                resp.add_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
                resp.add_header(Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap());
                let _ = req.respond(resp);
                continue;
            }

            let total_facts = m.memory.documental.entries.len()
                + m.memory.episodic.entries.len()
                + m.memory.conversational.entries.len();

            let resp_json = serde_json::json!({
                "status": "ok",
                "removed": stats.duplicates_pruned,
                "dedup_threshold": dedup_threshold,
                "total_facts": total_facts,
                "stats": {
                    "episodic_transferred": stats.episodic_transferred,
                    "conversational_transferred": stats.conversational_transferred,
                    "duplicates_pruned": stats.duplicates_pruned,
                    "total_documental_entries": stats.total_documental_entries,
                }
            });
            let mut resp = Response::from_string(resp_json.to_string()).with_status_code(StatusCode(200));
            resp.add_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
            resp.add_header(Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap());
            let _ = req.respond(resp);
            continue;
        }

        if url == "/api/memory/query" && method == Method::Post {
            let mut body = String::new();
            let mut req = request;
            let _ = req.as_reader().read_to_string(&mut body);

            if body.len() > 8192 {
                let mut resp = Response::from_string(
                    serde_json::json!({
                        "error": "Payload excede el límite máximo de 8192 bytes"
                    }).to_string()
                ).with_status_code(StatusCode(400));
                resp.add_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
                resp.add_header(Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap());
                let _ = req.respond(resp);
                continue;
            }

            let req_data: serde_json::Value = match serde_json::from_str(&body) {
                Ok(v) => v,
                Err(_) => {
                    let mut resp = Response::from_string(
                        serde_json::json!({ "error": "JSON malformado" }).to_string()
                    ).with_status_code(StatusCode(400));
                    resp.add_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
                    resp.add_header(Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap());
                    let _ = req.respond(resp);
                    continue;
                }
            };

            let text = req_data
                .get("text")
                .or_else(|| req_data.get("query"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim();
            if text.is_empty() {
                let mut resp = Response::from_string(
                    serde_json::json!({ "error": "El campo 'text' o 'query' es obligatorio" }).to_string()
                ).with_status_code(StatusCode(400));
                resp.add_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
                resp.add_header(Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap());
                let _ = req.respond(resp);
                continue;
            }

            if text.contains("..") {
                let mut resp = Response::from_string(
                    serde_json::json!({ "error": "Secuencia no permitida ('..') detectada" }).to_string()
                ).with_status_code(StatusCode(400));
                resp.add_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
                resp.add_header(Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap());
                let _ = req.respond(resp);
                continue;
            }

            let top_k = req_data.get("top_k")
                .and_then(|v| v.as_u64())
                .unwrap_or(5)
                .clamp(1, 20) as usize;

            let niche_filter = req_data.get("niche").and_then(|v| v.as_str());

            let active_guard = active_model.read().unwrap();
            let m = match active_guard.as_ref() {
                Some(m) => m,
                None => {
                    let mut resp = Response::from_string(
                        serde_json::json!({ "error": "No hay ningún modelo activo cargado en el servidor" }).to_string()
                    ).with_status_code(StatusCode(400));
                    resp.add_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
                    resp.add_header(Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap());
                    let _ = req.respond(resp);
                    continue;
                }
            };

            let expected_dim = m.llm.dim();
            let raw_vec = match m.llm.embed_text(text, &m.tokenizer) {
                Ok(v) => v,
                Err(e) => {
                    let mut resp = Response::from_string(
                        serde_json::json!({ "error": format!("Error extrayendo embedding: {}", e) }).to_string()
                    ).with_status_code(StatusCode(500));
                    resp.add_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
                    resp.add_header(Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap());
                    let _ = req.respond(resp);
                    continue;
                }
            };

            let query_vec = match (&m.memory_mu, m.memory_uses_whitening) {
                (Some(mu), true) if mu.len() == expected_dim => {
                    let mut centered = vec![0.0f32; expected_dim];
                    let mut norm_sq = 0.0f32;
                    for i in 0..expected_dim {
                        let diff = raw_vec[i] - mu[i];
                        centered[i] = diff;
                        norm_sq += diff * diff;
                    }
                    let norm = norm_sq.sqrt().max(1e-8);
                    for val in centered.iter_mut() {
                        *val /= norm;
                    }
                    centered
                }
                _ => raw_vec,
            };

            let all_matches = m.memory.retrieve_context(&query_vec, top_k);
            let filtered_matches: Vec<_> = all_matches.into_iter()
                .filter(|m_item| {
                    if let Some(target) = niche_filter {
                        if target != "auto" && !target.is_empty() {
                            return m_item.niche.as_str().eq_ignore_ascii_case(target);
                        }
                    }
                    true
                })
                .take(top_k)
                .map(|m_item| {
                    serde_json::json!({
                        "id": m_item.id,
                        "niche": m_item.niche.as_str(),
                        "similarity": m_item.similarity,
                        "text": m_item.text,
                    })
                })
                .collect();

            let resp_json = serde_json::json!({
                "status": "ok",
                "query": text,
                "count": filtered_matches.len(),
                "matches": &filtered_matches,
                "results": &filtered_matches,
                "whitened": m.memory_uses_whitening && !m.memory_whitening_missing,
            });
            let mut resp = Response::from_string(resp_json.to_string()).with_status_code(StatusCode(200));
            resp.add_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
            resp.add_header(Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap());
            let _ = req.respond(resp);
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
                        let dim = llm.dim() as u32;
                        let (memory_threshold, memory_uses_whitening, memory_mu, memory_whitening_missing) =
                            crate::compute::island::configure_model_memory(&model_path, dim as usize);
                        let (memory, memory_dir) = crate::compute::island::IslandOrchestrator::load_paired_for_model(
                            &model_path,
                            dim,
                            memory_threshold,
                        );
                        {
                            let mut guard = active_model.write().unwrap();
                            let old = guard.take();
                            drop(old); // Forzar munmap inmediato del modelo previo
                            *guard = Some(LoadedModel {
                                name: name.clone(),
                                path: model_path,
                                llm,
                                tokenizer,
                                memory,
                                memory_threshold,
                                memory_uses_whitening,
                                memory_mu,
                                memory_whitening_missing,
                                memory_dir,
                            });
                        }
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
                        resp.add_header(
                            Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..])
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
                resp.add_header(
                    Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap(),
                );
                let _ = req.respond(resp);
            }
            continue;
        }

        if url == "/api/unload_model" && method == Method::Post {
            {
                let mut guard = active_model.write().unwrap();
                let old = guard.take();
                drop(old); // Forzar munmap inmediato del modelo activo
            }
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
                    loaded,
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
                        use_memory: Some(false),
                        memory_threshold: None,
                    });

                let user_msg = chat_req.message.as_deref().unwrap_or("");
                let sys_prompt = chat_req.system_prompt.as_deref().unwrap_or(
                    "Tu nombre es GAJE. Eres un asistente de inteligencia artificial avanzado, servicial, conciso y preciso.",
                );

                let template = streaming::detect_chat_template_from_tokenizer(&loaded.tokenizer);
                let mem_config = loaded.memory_config(
                    chat_req.use_memory.unwrap_or(false),
                    chat_req.memory_threshold,
                );
                let (chat_prompt, telemetry) = crate::compute::island::prepare_prompt_with_memory(
                    user_msg,
                    chat_req.history.as_deref(),
                    sys_prompt,
                    template,
                    &loaded.llm,
                    &loaded.tokenizer,
                    Some(&loaded.memory),
                    &mem_config,
                );

                let prompt_tokens = loaded
                    .tokenizer
                    .encode(&chat_prompt, false)
                    .unwrap_or_default();
                let prompt_tokens_usize: Vec<usize> =
                    prompt_tokens.into_iter().map(|t| t as usize).collect();
                let mut eos_ids = loaded
                    .tokenizer
                    .get_stop_tokens()
                    .into_iter()
                    .map(|t| t as usize)
                    .collect::<Vec<_>>();
                if eos_ids.is_empty() {
                    eos_ids = vec![0, 2, 151643, 151644, 151645];
                }
                let gen = loaded
                    .llm
                    .generate_native_core(
                        prompt_tokens_usize,
                        chat_req.max_tokens.unwrap_or(256),
                        chat_req.temperature.unwrap_or(0.3),
                        chat_req.repetition_penalty.unwrap_or(1.15),
                        eos_ids,
                    )
                    .unwrap_or_default();
                let gen_u32: Vec<u32> = gen.into_iter().map(|t| t as u32).collect();
                let reply = loaded.tokenizer.decode(&gen_u32, true).unwrap_or_default();
                let clean = streaming::clean_special_tokens(&reply).trim().to_string();

                let json_resp = serde_json::json!({
                    "response": clean,
                    "status": "ok",
                    "model": loaded.name,
                    "memory": &telemetry,
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
