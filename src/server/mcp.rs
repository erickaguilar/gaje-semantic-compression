//! Servidor MCP (Model Context Protocol) nativo para GAJE Helix.
//!
//! Implementa la especificación MCP sobre stdio usando JSON-RPC 2.0.
//! Actúa como shim liviano de alto rendimiento hacia el servidor HTTP local de GAJE Helix.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use std::time::Duration;

#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Option<Value>,
}

#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    pub jsonrpc: &'static str,
    pub id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl JsonRpcResponse {
    fn success(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: Some(result),
            error: None,
        }
    }

    fn error(id: Value, code: i32, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
                data: None,
            }),
        }
    }
}

/// Ejecuta el bucle de eventos del servidor MCP sobre stdio.
pub fn run_mcp_server(server_url: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let base_url = server_url.trim_end_matches('/').to_string();
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(5))
        .build();

    eprintln!("🧬 [GAJE MCP] Servidor iniciado sobre stdio.");
    eprintln!("🔗 [GAJE MCP] Puente HTTP hacia: {}", base_url);

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("⚠️ [GAJE MCP] Error leyendo stdin: {}", e);
                break;
            }
        };

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let req: JsonRpcRequest = match serde_json::from_str(trimmed) {
            Ok(r) => r,
            Err(e) => {
                let err_resp = JsonRpcResponse::error(
                    Value::Null,
                    -32700,
                    format!("Parse error: {}", e),
                );
                let out = serde_json::to_string(&err_resp)? + "\n";
                stdout.write_all(out.as_bytes())?;
                stdout.flush()?;
                continue;
            }
        };

        if req.jsonrpc != "2.0" {
            let err_resp = JsonRpcResponse::error(
                req.id.unwrap_or(Value::Null),
                -32600,
                "Invalid Request: 'jsonrpc' must be '2.0'",
            );
            let out = serde_json::to_string(&err_resp)? + "\n";
            stdout.write_all(out.as_bytes())?;
            stdout.flush()?;
            continue;
        }

        // Si es una notificación (sin id), procesar sin responder
        let req_id = match req.id {
            Some(id) => id,
            None => {
                // Notificaciones MCP conocidas
                if req.method == "notifications/initialized" {
                    eprintln!("✅ [GAJE MCP] Handshake completado por el cliente.");
                }
                continue;
            }
        };

        let response = match req.method.as_str() {
            "initialize" => handle_initialize(req_id, req.params),
            "ping" => JsonRpcResponse::success(req_id, json!({})),
            "tools/list" => handle_tools_list(req_id),
            "tools/call" => handle_tools_call(req_id, req.params, &agent, &base_url),
            other => JsonRpcResponse::error(
                req_id,
                -32601,
                format!("Method not found: {}", other),
            ),
        };

        let out = serde_json::to_string(&response)? + "\n";
        stdout.write_all(out.as_bytes())?;
        stdout.flush()?;
    }

    eprintln!("🧬 [GAJE MCP] Servidor finalizado.");
    Ok(())
}

fn handle_initialize(id: Value, _params: Option<Value>) -> JsonRpcResponse {
    let result = json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "tools": {}
        },
        "serverInfo": {
            "name": "gaje-mcp",
            "version": "1.7.4"
        }
    });
    JsonRpcResponse::success(id, result)
}

fn handle_tools_list(id: Value) -> JsonRpcResponse {
    let tools = json!({
        "tools": [
            {
                "name": "gmem_query",
                "description": "Busca recuerdos y conocimientos semánticos en la memoria asociativa (.gmem) de GAJE Helix mediante similitud de embeddings neuronales y centrado de media (μ) + L2.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Texto o consulta conceptual a buscar en la memoria asociativa"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Número máximo de recuerdos más relevantes a recuperar (por defecto 5)",
                            "default": 5
                        },
                        "niche": {
                            "type": "string",
                            "enum": ["episodic", "documental", "conversational"],
                            "description": "Nicho ecológico de memoria donde buscar (opcional, busca en todos si se omite)"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "gmem_store",
                "description": "Almacena de forma persistente y atómica un hecho, contexto o memoria semántica en la memoria asociativa (.gmem) de GAJE Helix.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Contenido textual del hecho o memoria a almacenar"
                        },
                        "niche": {
                            "type": "string",
                            "enum": ["episodic", "documental", "conversational"],
                            "description": "Nicho ecológico de memoria (por defecto 'episodic')",
                            "default": "episodic"
                        },
                        "tags": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Etiquetas o metadatos categóricos opcionales asociados a la memoria"
                        }
                    },
                    "required": ["text"]
                }
            }
        ]
    });
    JsonRpcResponse::success(id, tools)
}

fn handle_tools_call(
    id: Value,
    params: Option<Value>,
    agent: &ureq::Agent,
    base_url: &str,
) -> JsonRpcResponse {
    let params_val = match params {
        Some(p) => p,
        None => return JsonRpcResponse::error(id, -32602, "Missing params"),
    };

    let tool_name = match params_val.get("name").and_then(|v| v.as_str()) {
        Some(name) => name,
        None => return JsonRpcResponse::error(id, -32602, "Missing 'name' in tools/call params"),
    };

    let arguments = params_val
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));

    match tool_name {
        "gmem_query" => call_gmem_query(id, arguments, agent, base_url),
        "gmem_store" => call_gmem_store(id, arguments, agent, base_url),
        other => JsonRpcResponse::success(
            id,
            json!({
                "content": [
                    {
                        "type": "text",
                        "text": format!("Error: Herramienta desconocida '{}'", other)
                    }
                ],
                "isError": true
            }),
        ),
    }
}

fn call_gmem_query(
    id: Value,
    arguments: Value,
    agent: &ureq::Agent,
    base_url: &str,
) -> JsonRpcResponse {
    let query_str = match arguments
        .get("query")
        .or_else(|| arguments.get("text"))
        .and_then(|q| q.as_str())
    {
        Some(q) if !q.trim().is_empty() => q,
        _ => return JsonRpcResponse::error(id, -32602, "Missing required string argument 'query' or 'text'"),
    };

    let top_k = arguments
        .get("top_k")
        .and_then(|k| k.as_u64())
        .unwrap_or(5);

    let niche = arguments
        .get("niche")
        .and_then(|n| n.as_str())
        .map(|s| s.to_string());

    let mut body = json!({
        "query": query_str,
        "text": query_str,
        "top_k": top_k
    });

    if let Some(n) = niche {
        body["niche"] = json!(n);
    }

    let url = format!("{}/api/memory/query", base_url);
    let body_str = serde_json::to_string(&body).unwrap_or_default();
    let resp = match agent
        .post(&url)
        .set("Content-Type", "application/json")
        .send_string(&body_str)
    {
        Ok(r) => r,
        Err(ureq::Error::Status(code, resp)) => {
            let err_text = resp.into_string().unwrap_or_default();
            return JsonRpcResponse::success(
                id,
                json!({
                    "content": [
                        {
                            "type": "text",
                            "text": format!("Error del servidor GAJE (HTTP {}): {}", code, err_text)
                        }
                    ],
                    "isError": true
                }),
            );
        }
        Err(ureq::Error::Transport(err)) => {
            return JsonRpcResponse::error(
                id,
                -32000,
                format!(
                    "Servidor GAJE no disponible en {}: {}. Asegúrate de ejecutar 'gaje-cli serve'.",
                    base_url, err
                ),
            );
        }
    };

    let resp_str = match resp.into_string() {
        Ok(s) => s,
        Err(e) => {
            return JsonRpcResponse::error(
                id,
                -32603,
                format!("Error leyendo cuerpo de respuesta de GAJE: {}", e),
            );
        }
    };

    let data: Value = match serde_json::from_str(&resp_str) {
        Ok(d) => d,
        Err(e) => {
            return JsonRpcResponse::error(
                id,
                -32603,
                format!("Respuesta inválida del servidor GAJE: {}", e),
            );
        }
    };

    let count = data.get("count").and_then(|c| c.as_u64()).unwrap_or(0);
    let whitened = data
        .get("whitened")
        .and_then(|w| w.as_bool())
        .unwrap_or(false);

    let mut formatted = format!(
        "Recuerdos recuperados: {} (Centrado de media μ + L2: {})\n\n",
        count,
        if whitened { "Activo" } else { "Inactivo" }
    );

    if let Some(results) = data.get("matches").or_else(|| data.get("results")).and_then(|r| r.as_array()) {
        if results.is_empty() {
            formatted.push_str("No se encontraron recuerdos relevantes.");
        } else {
            for (i, item) in results.iter().enumerate() {
                let mem_id = item.get("id").and_then(|v| v.as_u64()).unwrap_or(0);
                let niche_name = item
                    .get("niche")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                let sim = item
                    .get("similarity")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                let text = item
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");

                formatted.push_str(&format!(
                    "[{}] (ID: {}, Nicho: {}, Similitud: {:.4})\n{}\n\n",
                    i + 1,
                    mem_id,
                    niche_name,
                    sim,
                    text
                ));
            }
        }
    }

    JsonRpcResponse::success(
        id,
        json!({
            "content": [
                {
                    "type": "text",
                    "text": formatted.trim_end()
                }
            ],
            "isError": false
        }),
    )
}

fn call_gmem_store(
    id: Value,
    arguments: Value,
    agent: &ureq::Agent,
    base_url: &str,
) -> JsonRpcResponse {
    let text = match arguments
        .get("text")
        .or_else(|| arguments.get("content"))
        .and_then(|t| t.as_str())
    {
        Some(t) if !t.trim().is_empty() => t,
        _ => return JsonRpcResponse::error(id, -32602, "Missing required string argument 'text'"),
    };

    let niche = arguments
        .get("niche")
        .and_then(|n| n.as_str())
        .unwrap_or("episodic");

    let tags = arguments
        .get("tags")
        .cloned()
        .unwrap_or_else(|| json!([]));

    let body = json!({
        "text": text,
        "niche": niche,
        "tags": tags
    });

    let url = format!("{}/api/memory/remember", base_url);
    let body_str = serde_json::to_string(&body).unwrap_or_default();
    let resp = match agent
        .post(&url)
        .set("Content-Type", "application/json")
        .send_string(&body_str)
    {
        Ok(r) => r,
        Err(ureq::Error::Status(code, resp)) => {
            let err_text = resp.into_string().unwrap_or_default();
            return JsonRpcResponse::success(
                id,
                json!({
                    "content": [
                        {
                            "type": "text",
                            "text": format!("Error del servidor GAJE (HTTP {}): {}", code, err_text)
                        }
                    ],
                    "isError": true
                }),
            );
        }
        Err(ureq::Error::Transport(err)) => {
            return JsonRpcResponse::error(
                id,
                -32000,
                format!(
                    "Servidor GAJE no disponible en {}: {}. Asegúrate de ejecutar 'gaje-cli serve'.",
                    base_url, err
                ),
            );
        }
    };

    let resp_str = match resp.into_string() {
        Ok(s) => s,
        Err(e) => {
            return JsonRpcResponse::error(
                id,
                -32603,
                format!("Error leyendo cuerpo de respuesta de GAJE: {}", e),
            );
        }
    };

    let data: Value = match serde_json::from_str(&resp_str) {
        Ok(d) => d,
        Err(e) => {
            return JsonRpcResponse::error(
                id,
                -32603,
                format!("Respuesta inválida del servidor GAJE: {}", e),
            );
        }
    };

    let mem_id = data.get("id").and_then(|v| v.as_u64()).unwrap_or(0);
    let n_name = data.get("niche").and_then(|v| v.as_str()).unwrap_or(niche);
    let total = data.get("total_facts").or_else(|| data.get("total_entries")).and_then(|v| v.as_u64()).unwrap_or(0);
    let persisted = data.get("persisted").and_then(|v| v.as_bool()).unwrap_or(true);

    let formatted = format!(
        "✅ Hecho almacenado exitosamente en la memoria asociativa GAJE.\nID: {}\nNicho: {}\nTotal en nicho: {}\nPersistencia atómica en disco: {}",
        mem_id,
        n_name,
        total,
        if persisted { "Confirmada" } else { "Pendiente" }
    );

    JsonRpcResponse::success(
        id,
        json!({
            "content": [
                {
                    "type": "text",
                    "text": formatted
                }
            ],
            "isError": false
        }),
    )
}
