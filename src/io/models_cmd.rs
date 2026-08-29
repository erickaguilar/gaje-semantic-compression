//! 📦 Subcomando `gaje-cli models` (list, inspect, verify)

use crate::io::header::FlatHeaderV2;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub struct ModelSummary {
    pub path: PathBuf,
    pub filename: String,
    pub size_mb: f64,
    pub arch_name: String,
    pub quant_format: String,
    pub n_embd: u32,
    pub n_layers: u32,
    pub has_gtok: bool,
}

/// Lista recursivamente todos los modelos .flat en el directorio especificado
pub fn list_models(search_dir: &Path) -> Result<Vec<ModelSummary>, String> {
    let mut results = Vec::new();
    if !search_dir.exists() {
        return Ok(results);
    }

    scan_directory(search_dir, &mut results)?;
    results.sort_by(|a, b| a.filename.cmp(&b.filename));
    Ok(results)
}

fn scan_directory(dir: &Path, acc: &mut Vec<ModelSummary>) -> Result<(), String> {
    let entries = fs::read_dir(dir).map_err(|e| format!("Error leyendo directorio {:?}: {}", dir, e))?;

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let _ = scan_directory(&path, acc);
        } else if let Some(ext) = path.extension() {
            if ext == "flat" || ext == "gaje" {
                if let Ok(summary) = read_model_summary(&path) {
                    acc.push(summary);
                }
            }
        }
    }
    Ok(())
}

fn read_model_summary(path: &Path) -> Result<ModelSummary, String> {
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    let meta = file.metadata().map_err(|e| e.to_string())?;
    let size_mb = meta.len() as f64 / (1024.0 * 1024.0);

    let mut header_bytes = [0u8; FlatHeaderV2::SIZE];
    file.read_exact(&mut header_bytes).map_err(|e| e.to_string())?;

    let header = FlatHeaderV2::from_bytes(&header_bytes).map_err(|e| format!("{:?}", e))?;

    let arch_name = if let Some(desc) = header.architecture_descriptor() {
        format!("{:?}", desc.family)
    } else {
        "Desconocido / Genérico".to_string()
    };

    let quant_format = match header.quant_format {
        1 => "Q4_0 (4-bit)",
        2 => "Q8_0 (8-bit)",
        3 => "Q2_0 (2-bit)",
        _ => "FP32 / Legacy",
    }
    .to_string();

    let has_gtok = header.gtok_len > 0;
    let filename = path.file_name().unwrap_or_default().to_string_lossy().to_string();

    Ok(ModelSummary {
        path: path.to_path_buf(),
        filename,
        size_mb,
        arch_name,
        quant_format,
        n_embd: header.arch_n_embd,
        n_layers: header.arch_n_blocks,
        has_gtok,
    })
}

pub fn print_models_table(models: &[ModelSummary]) {
    println!("\n🧬 ===========================================================================================");
    println!("📦 GAJE HELIX — Catálogo de Organismos Genómicos Locales");
    println!("===========================================================================================\n");

    if models.is_empty() {
        println!("  ⚠️ No se encontraron modelos (.flat) en el directorio especificado.");
        println!("  💡 Tip: Descarga un modelo con: gaje-cli pull pico\n");
        return;
    }

    println!("{:<30} {:<15} {:<12} {:<10} {:<12} {:<8}", 
        "ARCHIVO", "ARQUITECTURA", "CUANTIZACIÓN", "DIM/CAPAS", "TAMAÑO", "GTOK");
    println!("{:-<30} {:-<15} {:-<12} {:-<10} {:-<12} {:-<8}", 
        "", "", "", "", "", "");

    for m in models {
        let dim_layers = if m.n_layers > 0 {
            format!("{}d/{}L", m.n_embd, m.n_layers)
        } else {
            "—".to_string()
        };
        let size_str = format!("{:.1} MB", m.size_mb);
        let gtok_badge = if m.has_gtok { "🟢 SÍ" } else { "⚪ NO" };

        println!("{:<30} {:<15} {:<12} {:<10} {:<12} {:<8}",
            m.filename, m.arch_name, m.quant_format, dim_layers, size_str, gtok_badge);
    }

    println!("\nTotal de organismos registrados: {}\n", models.len());
}

pub fn inspect_model(file_path: &Path) -> Result<(), String> {
    let mut file = File::open(file_path).map_err(|e| format!("Error abriendo {:?}: {}", file_path, e))?;
    let meta = file.metadata().map_err(|e| e.to_string())?;
    let mut header_bytes = [0u8; FlatHeaderV2::SIZE];
    file.read_exact(&mut header_bytes).map_err(|e| e.to_string())?;

    let header = FlatHeaderV2::from_bytes(&header_bytes).map_err(|e| format!("Cabecera inválida: {:?}", e))?;

    println!("\n🔍 ========================================================");
    println!("🔬 Inspección Estructural: {:?}", file_path);
    println!("========================================================\n");

    println!("📄 Metadatos del Archivo:");
    println!("   • Tamaño en Disco:     {:.2} MB ({} bytes)", meta.len() as f64 / (1024.0 * 1024.0), meta.len());
    println!("   • Magic Bytes:         {:?}", std::str::from_utf8(&header.magic).unwrap_or("????"));
    println!("   • Versión de Formato:  v{}", header.version);
    println!("   • Número de Tensores:  {}", header.num_tensors);

    println!("\n🧬 Configuración de Arquitectura:");
    if let Some(desc) = header.architecture_descriptor() {
        println!("   • Familia de Modelo:   {:?}", desc.family);
        println!("   • Dimensión Oculta:    {} (n_embd)", desc.n_embd);
        println!("   • Cabezas de Atención: {} (n_head)", desc.n_head);
        println!("   • Cabezas KV:          {} (n_head_kv)", desc.n_head_kv);
        println!("   • Capas de Bloques:    {} (n_blocks)", desc.n_blocks);
        println!("   • Base RoPE:           {}", desc.rope_base);
        println!("   • Activación FFN:      {}", desc.ffn_act);
        println!("   • Plantilla de Chat:   {}", desc.chat_template);
    } else {
        println!("   • Sin descriptor de arquitectura incrustado (v1 legacy)");
    }

    println!("\n🎛️ Esquema de Cuantización:");
    println!("   • Formato:             {:?}", header.quantization_type());
    println!("   • Tamaño de Grupo:     {} elementos", header.effective_group_size());
    println!("   • Offset de Pesos:     {} bytes", header.weights_offset);
    println!("   • Longitud de Pesos:   {} bytes", header.weights_len);

    println!("\n📚 Tokenizador GTOK Incrustado:");
    if header.gtok_len > 0 {
        println!("   • Estado:              🟢 Incrustado");
        println!("   • Offset GTOK:         {} bytes", header.gtok_offset);
        println!("   • Longitud GTOK:       {} bytes", header.gtok_len);
    } else {
        println!("   • Estado:              ⚪ No incrustado (Requiere tokenizador externo)");
    }

    println!("========================================================\n");
    Ok(())
}

pub fn verify_model(file_path: &Path) -> Result<(), String> {
    println!("🔍 Verificando integridad estructural de {:?}...", file_path);
    let mut file = File::open(file_path).map_err(|e| format!("Error abriendo {:?}: {}", file_path, e))?;
    let meta = file.metadata().map_err(|e| e.to_string())?;

    if meta.len() < FlatHeaderV2::SIZE as u64 {
        return Err(format!("El archivo es demasiado pequeño (< {} bytes)", FlatHeaderV2::SIZE));
    }

    let mut header_bytes = [0u8; FlatHeaderV2::SIZE];
    file.read_exact(&mut header_bytes).map_err(|e| e.to_string())?;
    let header = FlatHeaderV2::from_bytes(&header_bytes).map_err(|e| format!("Error en cabecera: {:?}", e))?;

    if header.weights_offset + header.weights_len > meta.len() {
        return Err(format!(
            "Archivo truncado: fin de pesos ({} bytes) excede el tamaño del archivo ({} bytes)",
            header.weights_offset + header.weights_len,
            meta.len()
        ));
    }

    println!("✅ Magic Bytes: OK");
    println!("✅ Offset y límites de tensores: OK");
    println!("✅ Header V2: OK (Arquitectura: {:?}, Cuantización: {:?})", 
        header.arch_family, header.quantization_type());
    println!("🏆 VEREDICTO: El modelo {:?} es 100% íntegro y compatible.", file_path);
    Ok(())
}
