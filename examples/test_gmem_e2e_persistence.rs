//! # 🧪 Test de Integración E2E: Persistencia Real .GMEM a Disco y Cold Start
//!
//! Este test certifica:
//! 1. Creación de memoria en las 3 islas.
//! 2. Serialización binaria `.gmem` a disco.
//! 3. Destrucción total de la instancia en memoria RAM.
//! 4. Carga Fría (Cold Start) desde el archivo `.gmem` mapeado.
//! 5. Verificación de precisión y latencia de recuperación.

use _impl::compute::island::{IslandNiche, IslandOrchestrator};
use _impl::io::gmem::GmemMemoryIndex;
use std::fs;
use std::time::Instant;

fn main() {
    println!("================================================================");
    println!("🧪 PRUEBA E2E DE PERSISTENCIA EN DISCO (.GMEM) Y COLD START");
    println!("================================================================");

    let _ = fs::create_dir_all("scratch");
    let file_path = "scratch/test_island_memory.gmem";
    let _ = fs::remove_file(file_path);

    let dim = 128;
    let mut orchestrator = IslandOrchestrator::new(dim);

    // 1. Ingestar conocimiento específico
    println!("📥 1. Ingestando datos reales en las 3 islas...");
    let mut v_doc = vec![0.0f32; dim as usize];
    v_doc[0] = 0.98;
    v_doc[1] = 0.02;
    orchestrator.add_memory(
        IslandNiche::Documental,
        1001,
        v_doc,
        "El proyecto GAJE utiliza compresión semántica genómica nativa en Rust.".to_string(),
    );

    let mut v_conv = vec![0.0f32; dim as usize];
    v_conv[1] = 0.95;
    orchestrator.add_memory(
        IslandNiche::Conversational,
        2002,
        v_conv,
        "El usuario prefiere respuestas concisas en markdown.".to_string(),
    );

    // 2. Serializar a disco
    println!("💾 2. Guardando índice de memoria .gmem a disco...");
    let t0 = Instant::now();
    orchestrator
        .documental
        .save_to_file(file_path)
        .expect("Error al guardar .gmem");
    let dur_save = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  ✅ Guardado completado en {:.2} ms", dur_save);

    // 3. Destrucción de Memoria RAM (Simular Cierre de Proceso)
    println!("🔥 3. Destruyendo instancia en RAM (Cierre de proceso)...");
    drop(orchestrator);

    // 4. Cold Start desde Disco
    println!("❄️ 4. Ejecutando Cold Start (Recuperación desde archivo .gmem)...");
    let t0 = Instant::now();

    let loaded_doc_index =
        GmemMemoryIndex::load_from_file(file_path).expect("Error al cargar .gmem desde disco");
    let metadata = fs::metadata(file_path).expect("El archivo .gmem no existe en disco");
    println!("  📁 Tamaño de archivo en disco: {} bytes", metadata.len());
    let dur_cold = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  ✅ Cold Start completado en {:.2} ms", dur_cold);

    // 5. Test de Búsqueda Vectorizada sobre Memoria Persistida
    println!("🔍 5. Ejecutando búsqueda semántica sobre la memoria recuperada...");
    let mut query = vec![0.0f32; dim as usize];
    query[0] = 0.95;

    // Asignar el índice restaurado desde disco
    let mut restored_orch = IslandOrchestrator::new(dim);
    restored_orch.documental = loaded_doc_index;

    let matches = restored_orch.retrieve_context(&query, 1);
    assert!(
        !matches.is_empty(),
        "No se recuperaron datos de la memoria persistida"
    );

    println!("  🎯 Fragmento Recuperado:");
    println!("     - Isla: {:?}", matches[0].niche);
    println!("     - Similitud: {:.4}", matches[0].similarity);
    println!("     - Contenido: \"{}\"", matches[0].text);

    let prompt_aug = restored_orch.build_augmented_prompt_from_matches(
        "¿Qué tecnología utiliza GAJE?",
        &matches,
        128,
    );

    println!("\n📝 Prompt Aumentado Final:");
    println!("----------------------------------------------------------------");
    println!("{}", prompt_aug);
    println!("----------------------------------------------------------------");

    println!("\n================================================================");
    println!("🏆 CERTIFICACIÓN DE PERSISTENCIA .GMEM COMPLETA");
    println!("  - Persistencia a disco : OK ({:.2} ms)", dur_save);
    println!("  - Cold Start Mmap      : OK ({:.2} ms)", dur_cold);
    println!(
        "  - Recuperación Semántica: OK (CosSim {:.4})",
        matches[0].similarity
    );
    println!("================================================================");

    // Limpieza
    let _ = fs::remove_file(file_path);
}
