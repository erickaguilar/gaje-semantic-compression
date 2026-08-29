//! Motor nativo de descargas multi-stream concurrentes estilo DNF / hf_transfer para modelos GAJE.
//!
//! Implementa:
//! - Petición inicial HEAD/Range para detección de tamaño total y soporte de HTTP 206 Partial Content.
//! - Pre-asignación zero-copy del archivo en disco (`File::set_len`) para eliminar fragmentación.
//! - Descarga multi-stream particionada en N hilos con Rayon.
//! - Escritura concurrente por offsets exactos en el archivo parcial.
//! - Telemetría en tiempo real con velocidad (MB/s), ETA y barra de progreso (`indicatif`).
//! - Descarga atómica con archivo temporal `.part` y reemplazo atómico para evitar corrupción.

#[cfg(feature = "native")]
use indicatif::{ProgressBar, ProgressStyle};
#[cfg(feature = "native")]
use std::fs::{File, OpenOptions};
#[cfg(feature = "native")]
use std::io::{Read, Seek, SeekFrom, Write};
#[cfg(feature = "native")]
use std::path::{Path, PathBuf};
#[cfg(feature = "native")]
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
#[cfg(feature = "native")]
use std::sync::Arc;
#[cfg(feature = "native")]
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct DownloadOptions {
    pub concurrency: usize,
    pub chunk_size_min: u64,
    pub user_agent: String,
}

impl Default for DownloadOptions {
    fn default() -> Self {
        Self {
            concurrency: 8,
            chunk_size_min: 2 * 1024 * 1024, // 2 MB mínimo por chunk
            user_agent: "GAJE-Helix-Engine/1.7.0 (Rust; Native-Downloader)".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DownloadStats {
    pub total_bytes: u64,
    pub elapsed_secs: f64,
    pub speed_mb_s: f64,
    pub destination: PathBuf,
}

/// Resuelve un identificador de modelo o URL directa al endpoint real de Hugging Face.
pub fn resolve_model_url(model_identifier: &str) -> (String, String) {
    if model_identifier.starts_with("http://") || model_identifier.starts_with("https://") {
        let filename = model_identifier
            .split('?')
            .next()
            .unwrap_or(model_identifier)
            .split('/')
            .last()
            .unwrap_or("model.flat")
            .to_string();
        return (model_identifier.to_string(), filename);
    }

    let (repo_id, filename) = match model_identifier {
        "gaje_pico_135m.flat" | "gaje_pico_135m" | "pico" => {
            ("erickaguilar/gaje-pico-135m", "gaje_pico_135m.flat")
        }
        "gaje_nano_0_5b.flat" | "gaje_nano_0_5b" | "nano_0_5b" | "qwen_0_5b" | "0.5b" | "qwen2.5-0.5b" => {
            ("eaguilar/gaje-nano-0.5b", "gaje_nano_0_5b.flat")
        }
        "gaje_nano_1.5b.flat" | "gaje_nano_1.5b" | "nano" => {
            ("erickaguilar/gaje-nano-1.5b", "gaje_nano_1.5b.flat")
        }
        "gaje_prime_3b.flat" | "gaje_prime_3b" | "prime" => {
            ("erickaguilar/gaje-prime-3b", "gaje_prime_3b.flat")
        }
        "gaje_ultra_7b.flat" | "gaje_ultra_7b" | "ultra" => {
            ("erickaguilar/gaje-ultra-7b", "gaje_ultra_7b.flat")
        }
        "deepseek_r1_1.5b.flat" | "deepseek_r1_1.5b" | "deepseek-r1" | "r1" | "r1-1.5b" => {
            ("eaguilar/gaje-models", "deepseek_r1_distill_qwen_1.5b.flat")
        }
        "deepseek_r1_7b.flat" | "deepseek_r1_7b" | "deepseek-r1-7b" | "r1-7b" => {
            ("eaguilar/gaje-models", "deepseek_r1_distill_qwen_7b.flat")
        }
        "gaje_gemma_2b.flat" | "gaje_gemma_2b" | "gemma_2b" | "gemma-2b" | "gemma-2" | "gemma" => {
            ("eaguilar/gaje-models", "gaje_gemma_2b.flat")
        }
        custom => {
            if custom.contains('/') {
                let parts: Vec<&str> = custom.split('/').collect();
                let last = parts.last().copied().unwrap_or("model.flat");
                let fname = if last.ends_with(".flat") || last.ends_with(".gaje") {
                    last
                } else {
                    "model.flat"
                };
                (custom, fname)
            } else {
                let fname = if custom.ends_with(".flat") || custom.ends_with(".gaje") {
                    custom
                } else {
                    "model.flat"
                };
                let repo = format!("erickaguilar/{}", custom.trim_end_matches(".flat"));
                return (
                    format!("https://huggingface.co/{}/resolve/main/{}", repo, fname),
                    fname.to_string(),
                );
            }
        }
    };

    let url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        repo_id, filename
    );
    (url, filename.to_string())
}

#[cfg(feature = "native")]
/// Descarga un archivo a máxima velocidad utilizando streams paralelos HTTP Range.
pub fn download_model(
    model_identifier: &str,
    target_dir: Option<&Path>,
    options: Option<DownloadOptions>,
    running: Option<Arc<AtomicBool>>,
) -> Result<DownloadStats, Box<dyn std::error::Error + Send + Sync>> {
    let (url, filename) = resolve_model_url(model_identifier);
    let out_dir = target_dir.unwrap_or_else(|| Path::new("models"));
    std::fs::create_dir_all(out_dir)?;
    let destination = out_dir.join(filename);

    download_file_direct(&url, &destination, options, running)
}

#[cfg(feature = "native")]
/// Descarga un archivo desde una URL directa hacia una ruta de destino local.
pub fn download_file_direct(
    url: &str,
    destination: &Path,
    options: Option<DownloadOptions>,
    running: Option<Arc<AtomicBool>>,
) -> Result<DownloadStats, Box<dyn std::error::Error + Send + Sync>> {
    let opts = options.unwrap_or_default();
    let agent = ureq::AgentBuilder::new()
        .user_agent(&opts.user_agent)
        .timeout(std::time::Duration::from_secs(45))
        .build();

    println!("⚡ [GAJE-Downloader] Conectando con: {}", url);

    // 1. Petición inicial HEAD / probe para determinar Content-Length y soporte de Range
    let head_resp = agent.head(url).call();
    let (content_length, supports_range) = match head_resp {
        Ok(resp) => {
            let len = resp
                .header("content-length")
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(0);
            let accept_ranges = resp
                .header("accept-ranges")
                .map(|v| v.to_lowercase().contains("bytes"))
                .unwrap_or(false);
            (len, accept_ranges)
        }
        Err(_) => {
            // Fallback con GET Range 0-0 si HEAD no está permitido por el CDN
            let probe = agent.get(url).set("Range", "bytes=0-0").call();
            match probe {
                Ok(resp) => {
                    let len = resp
                        .header("content-range")
                        .and_then(|cr| cr.split('/').last())
                        .and_then(|v| v.parse::<u64>().ok())
                        .unwrap_or(0);
                    let is_partial = resp.status() == 206;
                    (len, is_partial)
                }
                Err(e) => {
                    return Err(format!("Error conectando con el servidor remoto: {}", e).into())
                }
            }
        }
    };

    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let part_file_path = destination.with_extension("flat.part");
    let start_time = Instant::now();

    println!(
        "📦 [GAJE-Downloader] Tamaño: {:.2} MB | Range HTTP 206: {}",
        content_length as f64 / (1024.0 * 1024.0),
        if supports_range {
            "Soportado (Multi-Stream Activo)"
        } else {
            "No soportado (1 Stream Lineal)"
        }
    );

    // 2. Si no soporta Range, el archivo es < 4 MB o concurrency es 1 -> Descarga lineal
    if !supports_range || content_length < 4 * 1024 * 1024 || opts.concurrency <= 1 {
        let mut file = File::create(&part_file_path)?;
        let resp = agent.get(url).call()?;
        let mut reader = resp.into_reader();

        let pb = ProgressBar::new(content_length);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:38.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, ETA: {eta})")
                .unwrap_or_else(|_| ProgressStyle::default_bar())
                .progress_chars("━╸━"),
        );

        let mut buf = [0u8; 64 * 1024];
        let mut downloaded = 0u64;
        loop {
            if let Some(ref r) = running {
                if !r.load(Ordering::Relaxed) {
                    let _ = std::fs::remove_file(&part_file_path);
                    return Err("Descarga cancelada por el usuario".into());
                }
            }
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            file.write_all(&buf[..n])?;
            downloaded += n as u64;
            pb.set_position(downloaded);
        }
        pb.finish_with_message("Descarga completada");
    } else {
        // 3. Descarga Multi-Stream Paralela estilo DNF / hf_transfer
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&part_file_path)?;

        // Pre-asignación en disco zero-copy
        file.set_len(content_length)?;
        drop(file);

        let num_workers = opts
            .concurrency
            .min((content_length / opts.chunk_size_min).max(1) as usize);
        let chunk_size = (content_length + num_workers as u64 - 1) / num_workers as u64;

        let pb = ProgressBar::new(content_length);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:38.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, ETA: {eta}) [Streams: {msg}]")
                .unwrap_or_else(|_| ProgressStyle::default_bar())
                .progress_chars("━╸━"),
        );
        pb.set_message(format!("{}", num_workers));

        let pb_arc = Arc::new(pb);
        let downloaded_bytes = Arc::new(AtomicU64::new(0));

        use rayon::prelude::*;
        let chunks: Vec<(usize, u64, u64)> = (0..num_workers)
            .map(|i| {
                let start = i as u64 * chunk_size;
                let end = (start + chunk_size - 1).min(content_length - 1);
                (i, start, end)
            })
            .collect();

        let errors: Vec<_> = chunks
            .into_par_iter()
            .map(|(_worker_id, start, end)| -> Result<(), String> {
                let mut chunk_file = OpenOptions::new()
                    .write(true)
                    .open(&part_file_path)
                    .map_err(|e| format!("Error abriendo archivo parcial: {}", e))?;

                chunk_file
                    .seek(SeekFrom::Start(start))
                    .map_err(|e| format!("Error en seek: {}", e))?;

                let range_header = format!("bytes={}-{}", start, end);
                let resp = agent
                    .get(url)
                    .set("Range", &range_header)
                    .call()
                    .map_err(|e| format!("Error en petición Range {}: {}", range_header, e))?;

                let mut reader = resp.into_reader();
                let mut buf = vec![0u8; 128 * 1024]; // Buffer de 128 KB
                let mut remaining = end - start + 1;

                while remaining > 0 {
                    if let Some(ref r) = running {
                        if !r.load(Ordering::Relaxed) {
                            return Err("Descarga cancelada".into());
                        }
                    }
                    let to_read = (buf.len() as u64).min(remaining) as usize;
                    let n = reader
                        .read(&mut buf[..to_read])
                        .map_err(|e| format!("Error leyendo socket: {}", e))?;
                    if n == 0 {
                        break;
                    }

                    chunk_file
                        .write_all(&buf[..n])
                        .map_err(|e| format!("Error escribiendo a disco: {}", e))?;

                    remaining -= n as u64;
                    downloaded_bytes.fetch_add(n as u64, Ordering::Relaxed);
                    pb_arc.inc(n as u64);
                }

                Ok(())
            })
            .filter_map(|r| r.err())
            .collect();

        if !errors.is_empty() {
            let _ = std::fs::remove_file(&part_file_path);
            return Err(format!("Fallo en descarga multi-stream: {}", errors.join("; ")).into());
        }

        pb_arc.finish_with_message("Descarga OK");
    }

    // Reemplazo atómico
    if destination.exists() {
        std::fs::remove_file(destination)?;
    }
    std::fs::rename(&part_file_path, destination)?;

    let elapsed = start_time.elapsed().as_secs_f64();
    let speed_mb_s = if elapsed > 0.0 {
        (content_length as f64 / (1024.0 * 1024.0)) / elapsed
    } else {
        0.0
    };

    println!(
        "\n✅ [GAJE-Downloader] Modelo guardado en: {:?}\n   Velocidad Promedio: {:.2} MB/s | Tiempo: {:.2}s",
        destination, speed_mb_s, elapsed
    );

    Ok(DownloadStats {
        total_bytes: content_length,
        elapsed_secs: elapsed,
        speed_mb_s,
        destination: destination.to_path_buf(),
    })
}
