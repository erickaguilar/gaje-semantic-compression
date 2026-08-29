use std::fs::File;
use std::io::Read;
use std::path::Path;
use tiny_http::{Header, Response, StatusCode};

#[cfg(feature = "native")]
use rust_embed::RustEmbed;

#[cfg(feature = "native")]
#[derive(RustEmbed)]
#[folder = "examples/ui/web_ui/"]
pub struct EmbeddedAssets;

pub fn get_mime_type_str(extension: &str) -> &'static str {
    match extension {
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

pub fn get_mime_type(path: &Path) -> &'static str {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    get_mime_type_str(ext)
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

    // Bloquear scripts de backend y rutas no públicas
    if relative_path.ends_with(".py")
        || relative_path.ends_with(".pyc")
        || relative_path.contains("eggs")
        || relative_path.contains("legacy_web")
    {
        return Some(
            Response::from_string("Recurso no permitido").with_status_code(StatusCode(403)),
        );
    }

    // Restringir páginas secundarias excluidas en el binario autónomo
    if relative_path.contains("docs") || relative_path.contains("architecture") {
        // Solo servir si existe explícitamente en el disco (modo dev)
        if !static_root.join(relative_path).exists() {
            return Some(
                Response::from_string("Página no disponible en el binario autónomo de chat.")
                    .with_status_code(StatusCode(404)),
            );
        }
    }

    // 1. Prioridad: Intentar servir desde el disco local si la carpeta existe (Modo Dev / Hot-Reload)
    if static_root.exists() {
        let target_path = static_root.join(relative_path);

        // Evitar Path Traversal
        let is_safe = match (static_root.canonicalize(), target_path.canonicalize()) {
            (Ok(canon_root), Ok(canon_target)) => canon_target.starts_with(&canon_root),
            _ => false,
        };

        if is_safe && target_path.exists() && target_path.is_file() {
            if let Ok(mut file) = File::open(&target_path) {
                let mut buffer = Vec::new();
                if file.read_to_end(&mut buffer).is_ok() {
                    let mime = get_mime_type(&target_path);
                    let mut resp = Response::from_data(buffer).with_status_code(StatusCode(200));
                    resp.add_header(
                        Header::from_bytes(&b"Content-Type"[..], mime.as_bytes()).unwrap(),
                    );
                    resp.add_header(
                        Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap(),
                    );
                    resp.add_header(
                        Header::from_bytes(&b"Cache-Control"[..], &b"no-cache"[..]).unwrap(),
                    );
                    return Some(resp);
                }
            }
        }
    }

    // 2. Fallback Autónomo: Servir desde memoria embebida compilada en el binario
    #[cfg(feature = "native")]
    {
        use rust_embed::RustEmbed;

        if let Some(embedded_file) = <EmbeddedAssets as RustEmbed>::get(relative_path) {
            let ext = Path::new(relative_path)
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("");
            let mime = get_mime_type_str(ext);
            let mut resp =
                Response::from_data(embedded_file.data.to_vec()).with_status_code(StatusCode(200));
            resp.add_header(Header::from_bytes(&b"Content-Type"[..], mime.as_bytes()).unwrap());
            resp.add_header(
                Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap(),
            );
            resp.add_header(
                Header::from_bytes(&b"Cache-Control"[..], &b"public, max-age=3600"[..]).unwrap(),
            );
            return Some(resp);
        }

        // SPA Fallback a index.html embebido si la ruta no tiene extensión
        if !relative_path.contains('.') {
            if let Some(index_file) = <EmbeddedAssets as RustEmbed>::get("index.html") {
                let mut resp =
                    Response::from_data(index_file.data.to_vec()).with_status_code(StatusCode(200));
                resp.add_header(
                    Header::from_bytes(&b"Content-Type"[..], &b"text/html; charset=utf-8"[..])
                        .unwrap(),
                );
                resp.add_header(
                    Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap(),
                );
                return Some(resp);
            }
        }
    }

    None
}
