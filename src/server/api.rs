use crate::cli::models as models_cmd;
use crate::compute::doctor;
use serde_json::json;
use std::path::Path;

pub fn get_runtime_info(loaded_model_name: Option<&str>) -> serde_json::Value {
    let report = doctor::run_doctor();

    let simd_str = if report.simd_avx512f {
        "AVX-512F"
    } else if report.simd_avx2 && report.simd_fma {
        "AVX2/FMA/SSE4.2"
    } else if report.simd_neon {
        "ARM NEON"
    } else {
        "SIMD Estándar"
    };

    let island_config = json!({
        "memory_type": ".gmem (Zero-Copy Native Rust)",
        "retrieval_latency_ms": 0.50,
        "context_budget": 512,
        "pills": ["Episódica", "Documental", "Conversación"]
    });

    let has_gpu = !report.gpu_backend.contains("No disponible");

    json!({
        "engine_version": "1.7.0-alpha (Native Rust)",
        "python_version": "None (Native Single-Binary)",
        "architecture": report.os_info,
        "cpu": "CPU Host SIMD Multi-Core",
        "cores": report.cpu_cores,
        "simd": simd_str,
        "os": report.os_info,
        "software": format!("Rust 2021 ({}) - Zero Python Runtime", simd_str),
        "hardware": format!("{} Cores - {} (Memoria: {:.1} GB/s)", report.cpu_cores, report.os_info, report.mmap_speed_gb_s),
        "island": island_config,
        "auto_load_model": true,
        "active_model": loaded_model_name.unwrap_or("Ninguno"),
        "gpu": {
            "name": report.gpu_backend,
            "available": has_gpu
        }
    })
}

fn model_quality(name: &str) -> f32 {
    let n = name.to_lowercase();
    if n == "max.gaje" || n.ends_with("max.gaje") || n == "max" {
        200.0
    } else if n.contains("3b") || n.contains("pro") || n.contains("qwen2_5_3b") {
        100.0
    } else if n.contains("deepseek") || n.contains("r1") {
        80.0
    } else if n.contains("gaje") && n.ends_with(".gaje") {
        75.0
    } else if n.contains("0_5b") || n.contains("turbo") {
        50.0
    } else if n.contains("smollm") || n.contains("135") || n.contains("nano") {
        30.0
    } else {
        0.0
    }
}

pub fn get_available_models(
    models_dir: &Path,
    loaded_model_name: Option<&str>,
) -> serde_json::Value {
    let models_res = models_cmd::list_models(models_dir).unwrap_or_default();

    let mut model_list = Vec::new();
    for m in models_res {
        let is_loaded = loaded_model_name.map(|n| n == m.filename).unwrap_or(false);
        let date_str = std::fs::metadata(&m.path)
            .and_then(|meta| meta.modified())
            .map(|time| {
                let dt: chrono::DateTime<chrono::Local> = time.into();
                dt.format("%Y-%m-%d %H:%M").to_string()
            })
            .unwrap_or_else(|_| "2026-08-28 00:00".to_string());

        let mem_dir = crate::compute::island::IslandOrchestrator::resolve_memory_dir(&m.path);
        let has_memory = mem_dir.is_dir();

        let quality = model_quality(&m.filename);
        model_list.push((
            quality,
            json!({
                "name": m.filename,
                "path": m.path.to_string_lossy(),
                "size_bytes": (m.size_mb * 1024.0 * 1024.0) as u64,
                "size_mb": m.size_mb,
                "architecture": m.arch_name,
                "quantization": m.quant_format,
                "n_embd": m.n_embd,
                "n_layers": m.n_layers,
                "has_gtok": m.has_gtok,
                "has_memory": has_memory,
                "ram_mb": if is_loaded { m.size_mb } else { 0.0 },
                "date": date_str
            }),
        ));
    }

    // Ordenar descendentemente por calidad/prioridad de modelo
    model_list.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let sorted_models: Vec<serde_json::Value> = model_list.into_iter().map(|(_, v)| v).collect();

    json!({ "models": sorted_models })
}

pub fn get_loaded_memory_info(
    memory: &crate::compute::island::IslandOrchestrator,
    mem_dir: &std::path::Path,
    threshold: f32,
) -> serde_json::Value {
    let doc_count = memory.documental.entries.len();
    let epi_count = memory.episodic.entries.len();
    let conv_count = memory.conversational.entries.len();
    let total = doc_count + epi_count + conv_count;

    let sample_facts: Vec<_> = memory
        .documental
        .entries
        .iter()
        .take(5)
        .map(|e| e.text.clone())
        .collect();

    json!({
        "status": if total > 0 { "connected" } else { "empty" },
        "dim": memory.dim,
        "total_facts": total,
        "memory_dir": mem_dir.to_string_lossy(),
        "memory_threshold": threshold,
        "entropy_gap_threshold": memory.entropy_gap_threshold,
        "kwta_ratio": memory.kwta_ratio,
        "niches": {
            "documental": doc_count,
            "episodic": epi_count,
            "conversational": conv_count,
        },
        "niche_weights": memory.niche_weights,
        "sample_facts": sample_facts
    })
}

pub fn get_memory_info(model_path: Option<&str>, dim: usize) -> serde_json::Value {
    if let Some(path) = model_path {
        if let Some(orch) =
            crate::compute::island::IslandOrchestrator::try_load_paired_memory(path, dim as u32)
        {
            let p = std::path::Path::new(path);
            let mem_dir = crate::compute::island::IslandOrchestrator::resolve_memory_dir(p);
            return get_loaded_memory_info(&orch, &mem_dir, 0.65);
        }
    }

    json!({
        "status": "none",
        "total_facts": 0,
        "niches": {
            "documental": 0,
            "episodic": 0,
            "conversational": 0
        }
    })
}
