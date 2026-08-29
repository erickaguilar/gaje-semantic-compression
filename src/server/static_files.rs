use std::fs::File;
use std::io::Read;
use std::path::Path;
use tiny_http::{Header, Response, StatusCode};

pub fn get_mime_type(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()).unwrap_or("") {
        "html" | "htm" => "text/html; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "js" | "mjs" => "application/javascript; charset=utf-8",
        "svg" => "image/svg+xml",
        "wasm" => "application/wasm",
        "json" => "application/json",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "ico" => "image/x-icon",
        "flat" | "gaje" | "gmem" => "application/octet-stream",
        _ => "application/octet-stream",
    }
}

pub fn serve_static_file(
    static_root: &Path,
    url_path: &str,
    chat_only: bool,
) -> Option<Response<std::io::Cursor<Vec<u8>>>> {
    let clean_path = url_path.split('?').next().unwrap_or("/");
    let path_no_slash = clean_path.trim_start_matches('/');

    let relative_path = if path_no_slash.is_empty() || path_no_slash == "index.html" {
        "index.html"
    } else {
        path_no_slash
    };

    // Modo chat-only: restringir páginas secundarias
    if chat_only && (relative_path.contains("docs") || relative_path.contains("architecture")) {
        return Some(
            Response::from_string("Acceso restringido en modo --chat-only")
                .with_status_code(StatusCode(404)),
        );
    }

    let target_path = static_root.join(relative_path);

    // Evitar Path Traversal
    if let Ok(canon_root) = static_root.canonicalize() {
        if let Ok(canon_target) = target_path.canonicalize() {
            if !canon_target.starts_with(&canon_root) {
                return Some(
                    Response::from_string("Acceso denegado")
                        .with_status_code(StatusCode(403)),
                );
            }
        }
    }

    if !target_path.exists() || target_path.is_dir() {
        // Solo aplicar SPA fallback si NO tiene extensión de archivo
        let has_extension = target_path.extension().is_some();
        if !has_extension {
            let fallback_index = static_root.join("index.html");
            if fallback_index.exists() {
                if let Ok(mut f) = File::open(&fallback_index) {
                    let mut buffer = Vec::new();
                    if f.read_to_end(&mut buffer).is_ok() {
                        let mut resp = Response::from_data(buffer).with_status_code(StatusCode(200));
                        resp.add_header(
                            Header::from_bytes(&b"Content-Type"[..], &b"text/html; charset=utf-8"[..])
                                .unwrap(),
                        );
                        resp.add_header(
                            Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..])
                                .unwrap(),
                        );
                        return Some(resp);
                    }
                }
            }
        }
        return None;
    }

    let mut file = match File::open(&target_path) {
        Ok(f) => f,
        Err(_) => return None,
    };

    let mut buffer = Vec::new();
    if file.read_to_end(&mut buffer).is_err() {
        return None;
    }

    let mime = get_mime_type(&target_path);
    let mut resp = Response::from_data(buffer).with_status_code(StatusCode(200));
    resp.add_header(Header::from_bytes(&b"Content-Type"[..], mime.as_bytes()).unwrap());
    resp.add_header(Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap());
    resp.add_header(Header::from_bytes(&b"Cache-Control"[..], &b"no-cache"[..]).unwrap());

    Some(resp)
}
