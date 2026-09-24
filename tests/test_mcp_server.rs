use serde_json::Value;
use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use tiny_http::{Header, Response, Server, StatusCode};

fn send_mcp_payload_with_url(input: &str, server_url: &str) -> Vec<Value> {
    let mut child = Command::new("./target/debug/gaje-cli")
        .arg("mcp")
        .arg("--server-url")
        .arg(server_url)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Fallo al ejecutar target/debug/gaje-cli mcp");

    {
        let stdin = child.stdin.as_mut().expect("No se pudo obtener stdin");
        stdin
            .write_all(input.as_bytes())
            .expect("Fallo escribiendo a stdin");
    }

    let output = child.wait_with_output().expect("Fallo esperando al proceso");
    let stdout_str = String::from_utf8_lossy(&output.stdout);

    stdout_str
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| serde_json::from_str::<Value>(line).expect("Respuesta no es JSON válido"))
        .collect()
}

fn send_mcp_payload(input: &str) -> Vec<Value> {
    send_mcp_payload_with_url(input, "http://127.0.0.1:54321")
}

#[test]
fn test_mcp_initialize() {
    let payload = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test-harness","version":"1.0"}}}"#;
    let responses = send_mcp_payload(&format!("{}\n", payload));

    assert_eq!(responses.len(), 1);
    let resp = &responses[0];
    assert_eq!(resp["jsonrpc"], "2.0");
    assert_eq!(resp["id"], 1);

    let result = &resp["result"];
    assert_eq!(result["protocolVersion"], "2024-11-05");
    assert_eq!(result["serverInfo"]["name"], "gaje-mcp");
    assert_eq!(result["capabilities"]["tools"], serde_json::json!({}));
}

#[test]
fn test_mcp_tools_list() {
    let payload = "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/list\"}\n";
    let responses = send_mcp_payload(payload);

    assert_eq!(responses.len(), 1);
    let resp = &responses[0];
    assert_eq!(resp["id"], 2);

    let tools = resp["result"]["tools"]
        .as_array()
        .expect("tools debe ser un array");
    assert_eq!(tools.len(), 2);

    let tool_names: Vec<&str> = tools
        .iter()
        .map(|t| t["name"].as_str().expect("nombre de tool"))
        .collect();

    assert!(tool_names.contains(&"gmem_query"));
    assert!(tool_names.contains(&"gmem_store"));

    // Verificar que los nichos son los tres canónicos: episodic, documental, conversational
    let query_tool = tools.iter().find(|t| t["name"] == "gmem_query").unwrap();
    let niches = query_tool["inputSchema"]["properties"]["niche"]["enum"]
        .as_array()
        .unwrap();
    let niche_strs: Vec<&str> = niches.iter().map(|n| n.as_str().unwrap()).collect();
    assert_eq!(niche_strs, vec!["episodic", "documental", "conversational"]);
}

#[test]
fn test_mcp_offline_server_error() {
    let payload = r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"gmem_query","arguments":{"query":"test query"}}}"#;
    let responses = send_mcp_payload(&format!("{}\n", payload));

    assert_eq!(responses.len(), 1);
    let resp = &responses[0];
    assert_eq!(resp["id"], 3);

    // Debe ser un error JSON-RPC con código -32000
    assert_eq!(resp["error"]["code"], -32000);
    let msg = resp["error"]["message"]
        .as_str()
        .expect("mensaje de error");
    assert!(msg.contains("Servidor GAJE no disponible"));
}

#[test]
fn test_mcp_invalid_request() {
    let payload = "{\"jsonrpc\":\"1.0\",\"id\":4,\"method\":\"ping\"}\n";
    let responses = send_mcp_payload(payload);

    assert_eq!(responses.len(), 1);
    let resp = &responses[0];
    assert_eq!(resp["id"], 4);
    assert_eq!(resp["error"]["code"], -32600);
}

#[test]
fn test_mcp_malformed_json() {
    let payload = "ESTO NO ES UN JSON\n";
    let responses = send_mcp_payload(payload);

    assert_eq!(responses.len(), 1);
    let resp = &responses[0];
    assert_eq!(resp["error"]["code"], -32700);
}

#[test]
fn test_mcp_e2e_full_cycle_query_and_store() {
    // 1. Levantar servidor local en puerto dinámico
    let server = Server::http("127.0.0.1:0").expect("Fallo creando servidor tiny_http");
    let port = server.server_addr().to_ip().unwrap().port();
    let server_url = format!("http://127.0.0.1:{}", port);

    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();

    let server_handle = thread::spawn(move || {
        while r.load(Ordering::SeqCst) {
            if let Ok(Some(mut req)) = server.recv_timeout(Duration::from_millis(50)) {
                let url = req.url().to_string();
                let method = req.method().clone();

                if url == "/api/memory/remember" && method == tiny_http::Method::Post {
                    let mut body = String::new();
                    let _ = req.as_reader().read_to_string(&mut body);
                    let val: Value = serde_json::from_str(&body).unwrap_or(serde_json::json!({}));
                    let text = val["text"].as_str().unwrap_or("");
                    assert!(text.contains("Canberra"));

                    let resp_body = serde_json::json!({
                        "status": "stored",
                        "id": 101,
                        "niche": "documental",
                        "total_entries": 1,
                        "persisted": true
                    });
                    let mut resp = Response::from_string(resp_body.to_string())
                        .with_status_code(StatusCode(200));
                    resp.add_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
                    let _ = req.respond(resp);
                } else if url == "/api/memory/query" && method == tiny_http::Method::Post {
                    let mut body = String::new();
                    let _ = req.as_reader().read_to_string(&mut body);
                    let val: Value = serde_json::from_str(&body).unwrap_or(serde_json::json!({}));
                    // Validar que se recibió query o text
                    let q = val["query"].as_str().or_else(|| val["text"].as_str()).unwrap_or("");
                    assert!(q.contains("Francia") || q.contains("capital"));

                    let resp_body = serde_json::json!({
                        "count": 1,
                        "whitened": true,
                        "results": [
                            {
                                "id": 101,
                                "niche": "documental",
                                "similarity": 0.9412,
                                "text": "París es la capital histórica de Francia."
                            }
                        ]
                    });
                    let mut resp = Response::from_string(resp_body.to_string())
                        .with_status_code(StatusCode(200));
                    resp.add_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap());
                    let _ = req.respond(resp);
                } else {
                    let _ = req.respond(Response::from_string("Not Found").with_status_code(StatusCode(404)));
                }
            }
        }
    });

    // 2. Ejecutar gmem_store a través del shim MCP
    let store_payload = r#"{"jsonrpc":"2.0","id":10,"method":"tools/call","params":{"name":"gmem_store","arguments":{"text":"Canberra es la capital de Australia","niche":"documental"}}}"#;
    let store_responses = send_mcp_payload_with_url(&format!("{}\n", store_payload), &server_url);

    assert_eq!(store_responses.len(), 1);
    let store_resp = &store_responses[0];
    assert_eq!(store_resp["id"], 10);
    assert_eq!(store_resp["result"]["isError"], false);
    let store_text = store_resp["result"]["content"][0]["text"].as_str().unwrap();
    assert!(store_text.contains("Hecho almacenado exitosamente"));
    assert!(store_text.contains("ID: 101"));

    // 3. Ejecutar gmem_query a través del shim MCP
    let query_payload = r#"{"jsonrpc":"2.0","id":11,"method":"tools/call","params":{"name":"gmem_query","arguments":{"text":"capital de Francia","top_k":3}}}"#;
    let query_responses = send_mcp_payload_with_url(&format!("{}\n", query_payload), &server_url);

    assert_eq!(query_responses.len(), 1);
    let query_resp = &query_responses[0];
    assert_eq!(query_resp["id"], 11);
    assert_eq!(query_resp["result"]["isError"], false);
    let query_text = query_resp["result"]["content"][0]["text"].as_str().unwrap();
    assert!(query_text.contains("París es la capital histórica de Francia."));
    assert!(query_text.contains("Centrado de media μ + L2: Activo"));
    assert!(query_text.contains("0.9412"));

    // Detener servidor de prueba
    running.store(false, Ordering::SeqCst);
    let _ = server_handle.join();
}
