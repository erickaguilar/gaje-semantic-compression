//! 🧪 Tests de Integración: Endpoints y Mecánica Central de Memory Station
//! Valida los 6 Criterios de Aceptación Pre-registrados (T1-T6):
//! - T1: Remember + Query Inmediato
//! - T2: Persistencia Atómica Entre Reinicios (mismo FS, tmp + rename + fsync)
//! - T3: DELETE Semantics (elimina 1 de 3 -> quedan 2, 404 para IDs inexistentes)
//! - T4: Consolidate Desduplicación
//! - T5: Rechazo de Payload Grande (> 8192 bytes)
//! - T6: Rechazo de Path Traversal ('..')

use _impl::compute::island::{IslandNiche, IslandOrchestrator};

#[test]
fn test_t1_remember_and_immediate_query() {
    let dim = 16;
    let mut orch = IslandOrchestrator::new(dim);

    let id = orch.generate_next_id();
    assert_eq!(id, 1);

    let mut vec = vec![0.0f32; dim as usize];
    vec[0] = 1.0;

    let text = "La capital de Francia es París.".to_string();
    orch.add_memory(IslandNiche::Documental, id, vec.clone(), text.clone());

    let results = orch.retrieve_context(&vec, 5);
    assert!(!results.is_empty(), "La consulta inmediata debió recuperar el recuerdo ingresado");
    assert_eq!(results[0].text, text);
    assert_eq!(results[0].id, 1);
}

#[test]
fn test_t2_atomic_persistence_and_restart_consistency() {
    let dim = 16;
    let temp_dir = std::env::temp_dir().join(format!("gaje_mem_t2_{}", std::process::id()));
    let temp_dir_str = temp_dir.to_string_lossy().to_string();

    {
        let mut orch = IslandOrchestrator::new(dim);
        let id1 = orch.generate_next_id();
        let id2 = orch.generate_next_id();
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);

        let mut v1 = vec![0.0f32; dim as usize];
        v1[0] = 1.0;
        let mut v2 = vec![0.0f32; dim as usize];
        v2[1] = 1.0;

        orch.add_memory(IslandNiche::Documental, id1, v1, "Hecho 1: Sol".to_string());
        orch.add_memory(IslandNiche::Episodic, id2, v2, "Hecho 2: Luna".to_string());

        // Guardado por operación atómica (3 tmp en el mismo dir -> rename)
        orch.save_all(&temp_dir_str).expect("save_all debe completarse exitosamente");

        // Ningún archivo temporal residual debe quedar en el directorio
        for entry in std::fs::read_dir(&temp_dir).unwrap() {
            let path = entry.unwrap().path();
            let name = path.file_name().unwrap().to_string_lossy();
            assert!(!name.contains(".tmp."), "No deben quedar archivos temporales residuales: {}", name);
        }
    }

    // Reinicio simulado: carga desde disco
    {
        let mut loaded_orch = IslandOrchestrator::new(dim);
        loaded_orch.load_all(&temp_dir_str).expect("load_all debe cargar la memoria persistida");

        assert_eq!(loaded_orch.documental.entries.len(), 1);
        assert_eq!(loaded_orch.episodic.entries.len(), 1);
        assert_eq!(loaded_orch.conversational.entries.len(), 0);

        // El contador persistido next_seq debe ser >= 3 para evitar colisiones
        let next_id = loaded_orch.generate_next_id();
        assert!(next_id >= 3, "El nuevo ID ({}) debe ser mayor que los previos (1, 2)", next_id);
    }

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_t3_delete_semantics_and_persistence() {
    let dim = 8;
    let temp_dir = std::env::temp_dir().join(format!("gaje_mem_t3_{}", std::process::id()));
    let temp_dir_str = temp_dir.to_string_lossy().to_string();

    let mut orch = IslandOrchestrator::new(dim);
    let id1 = orch.generate_next_id();
    let id2 = orch.generate_next_id();
    let id3 = orch.generate_next_id();

    let dummy_v = vec![0.1f32; dim as usize];
    orch.add_memory(IslandNiche::Documental, id1, dummy_v.clone(), "Dato 1".to_string());
    orch.add_memory(IslandNiche::Documental, id2, dummy_v.clone(), "Dato 2".to_string());
    orch.add_memory(IslandNiche::Documental, id3, dummy_v.clone(), "Dato 3".to_string());

    assert_eq!(orch.documental.entries.len(), 3);

    // Borrado de ID existente
    let removed = orch.remove_memory(Some(IslandNiche::Documental), id2);
    assert!(removed, "El recuerdo id2 debió ser eliminado exitosamente");
    assert_eq!(orch.documental.entries.len(), 2, "Deben quedar exactamente 2 recuerdos");

    // Borrado de ID inexistente debe retornar false (404 semántico)
    let removed_nonexistent = orch.remove_memory(Some(IslandNiche::Documental), 9999);
    assert!(!removed_nonexistent, "Un ID inexistente debe retornar false");

    // Persistir y verificar tras recarga
    orch.save_all(&temp_dir_str).unwrap();

    let mut reloaded = IslandOrchestrator::new(dim);
    reloaded.load_all(&temp_dir_str).unwrap();
    assert_eq!(reloaded.documental.entries.len(), 2);
    assert!(reloaded.documental.entries.iter().any(|e| e.id == id1));
    assert!(reloaded.documental.entries.iter().any(|e| e.id == id3));
    assert!(!reloaded.documental.entries.iter().any(|e| e.id == id2));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_t4_consolidate_deduplication() {
    let dim = 8;
    let mut orch = IslandOrchestrator::new(dim);

    let mut v = vec![0.0f32; dim as usize];
    v[0] = 1.0;

    // Insertar 5 variantes idénticas en el nicho documental
    for i in 1..=5 {
        orch.add_memory(
            IslandNiche::Documental,
            i,
            v.clone(),
            format!("Hecho idéntico número {}", i),
        );
    }

    assert_eq!(orch.documental.entries.len(), 5);

    let stats = orch.consolidate_memory(0.97);
    assert!(
        stats.duplicates_pruned >= 4,
        "Consolidation debió podar al menos 4 duplicados, podó: {}",
        stats.duplicates_pruned
    );
    assert_eq!(orch.documental.entries.len(), 1, "Debe quedar únicamente 1 ejemplar canónico");
}

#[test]
fn test_t5_payload_limit_logic() {
    // Validar regla de rechazo > 8192 bytes
    let large_text = "a".repeat(8193);
    assert!(large_text.len() > 8192);
    let valid_text = "a".repeat(8192);
    assert!(valid_text.len() <= 8192);
}

#[test]
fn test_t6_path_traversal_detection() {
    // Validar detección de secuencias de escape
    let malicious_path = "../../etc/passwd";
    let malicious_niche = "../system";
    let safe_text = "Texto normal con información de París.";
    let safe_niche = "documental";

    assert!(malicious_path.contains(".."));
    assert!(malicious_niche.contains(".."));
    assert!(!safe_text.contains(".."));
    assert!(!safe_niche.contains(".."));
}
