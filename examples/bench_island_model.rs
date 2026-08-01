//! # 🧪 Benchmark Isolado del Island Model e Índice `.gmem`
//!
//! Este benchmark mide con precisión de microsegundos:
//! 1. Ingesta de 10,000 entradas de memoria en las 3 islas.
//! 2. Latencia de búsqueda vectorizada `retrieve_context` (Top-K por CosSim).
//! 3. Latencia de ensamblado `build_augmented_prompt` con presupuesto estricto de tokens.
//! 4. Tiempo total de overhead por consulta de usuario.

use _impl::compute::island::{IslandNiche, IslandOrchestrator};
use std::hint::black_box;
use std::time::Instant;

fn main() {
    println!("================================================================");
    println!("🧪 BENCHMARK DE RENDIMIENTO: ISLAND MODEL & MEMORIA .GMEM");
    println!("================================================================");

    let dim = 128; // Dimensión de embedding ultraligero DNI
    let total_entries = 10_000;
    let mut orchestrator = IslandOrchestrator::new(dim as u32);

    println!(
        "📥 1. Ingestando {} memorias en las 3 islas...",
        total_entries
    );
    let t0 = Instant::now();

    for i in 0..total_entries {
        let niche = match i % 3 {
            0 => IslandNiche::Episodic,
            1 => IslandNiche::Documental,
            _ => IslandNiche::Conversational,
        };

        // Generar un vector determinista normado
        let mut vec = vec![0.0f32; dim];
        vec[i % dim] = 1.0;
        let text = format!("Memoria registrada en isla {:?} id={}", niche, i);

        orchestrator.add_memory(niche, i as u64, vec, text);
    }

    let dur_ingest = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "  ✅ Ingesta completada en {:.2} ms ({:.2} µs/entrada)",
        dur_ingest,
        dur_ingest * 1000.0 / total_entries as f64
    );

    // 2. Medir Latencia de Retrieval en 1,000 consultas
    println!("\n🔍 2. Ejecutando 1,000 consultas semánticas de retrieval (Top-2 por isla)...");
    let queries_count = 1_000;
    let mut dummy_sink = 0usize;

    let t0 = Instant::now();
    for q in 0..queries_count {
        let mut query_vec = vec![0.0f32; dim];
        query_vec[q % dim] = 0.95;
        query_vec[(q + 1) % dim] = 0.05;

        let results = orchestrator.retrieve_context(&query_vec, 2);
        dummy_sink += black_box(results.len());
    }
    let dur_search = t0.elapsed().as_secs_f64() * 1000.0;
    let us_per_query = dur_search * 1000.0 / queries_count as f64;

    println!(
        "  📊 Latencia Total Retrieval (1,000 queries): {:.2} ms",
        dur_search
    );
    println!(
        "  ⚡ Latencia Promedio por Consulta        : {:.2} µs ({:.4} ms)",
        us_per_query,
        us_per_query / 1000.0
    );

    // 3. Medir Latencia de Prompt Augmentation con presupuesto de tokens usando resultados previos
    println!("\n📝 3. Midiendo ensamblado de Augmented Prompt desde resultados recuperados...");
    let prompt_count = 1_000;
    let t0 = Instant::now();

    for q in 0..prompt_count {
        let mut query_vec = vec![0.0f32; dim];
        query_vec[q % dim] = 0.95;
        let matches = orchestrator.retrieve_context(&query_vec, 2);
        let augmented = orchestrator.build_augmented_prompt_from_matches(
            "Explica el concepto de la memoria .gmem",
            &matches,
            128,
        );
        dummy_sink += black_box(augmented.len());
    }
    let dur_e2e = t0.elapsed().as_secs_f64() * 1000.0;
    let us_per_e2e = dur_e2e * 1000.0 / prompt_count as f64;

    println!(
        "  📊 Latencia Total E2E (1,000 prompts completados): {:.2} ms",
        dur_e2e
    );
    println!(
        "  ⚡ Latencia Promedio E2E (Retrieval + Augmentation): {:.2} µs ({:.4} ms)",
        us_per_e2e,
        us_per_e2e / 1000.0
    );

    black_box(dummy_sink);

    println!("\n================================================================");
    println!("🏆 EVALUACIÓN DE OVERHEAD PARA EL ENGINE GAJE:");
    println!(
        "  - Latencia E2E Total por Mensaje Usuario: {:.2} µs ({:.4} ms)",
        us_per_e2e,
        us_per_e2e / 1000.0
    );
    if us_per_e2e / 1000.0 < 2.0 {
        println!("  ✅ CERTIFICACIÓN: Overhead < 2.0 ms (Cero impacto perceptible en TTFT)");
    } else {
        println!("  ⚠️ ALERTA: Overhead > 2.0 ms");
    }
    println!("================================================================");
}
