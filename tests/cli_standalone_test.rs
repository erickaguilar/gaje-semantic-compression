//! 🧪 Test de Integración Soberano del CLI Único (Fase 4 Single-Binary)
//!
//! Valida el comportamiento de `gaje-cli` (doctor, audit, dataset-build, models, export).

use std::fs::File;
use std::io::Write;
use std::path::Path;

#[test]
fn test_cli_dataset_build_roundtrip() {
    let tmp_input = "/tmp/test_cli_dataset_input.jsonl";
    let tmp_output = "/tmp/test_cli_dataset_output.jsonl";

    // 1. Crear archivo temporal de entrada con pares conversacionales
    {
        let mut f = File::create(tmp_input).expect("Error creando archivo temporal");
        writeln!(f, "{{\"instruction\": \"Hola GAJE\", \"response\": \"Hola, soy el motor genómico.\"}}").unwrap();
        writeln!(f, "{{\"user\": \"¿Qué es mmap?\", \"assistant\": \"Mapeo zero-copy de memoria.\"}}").unwrap();
    }

    // 2. Invocar dataset_build_cmd directamente desde Rust
    let inputs = vec![tmp_input.to_string()];
    let res = _impl::io::cli_tools::dataset_build_cmd(&inputs, tmp_output, None, 5);
    assert!(res.is_ok(), "dataset_build_cmd falló: {:?}", res.err());

    // 3. Verificar salida
    assert!(Path::new(tmp_output).exists());
    let content = std::fs::read_to_string(tmp_output).unwrap();
    assert!(content.contains("User: Hola GAJE"));
    assert!(content.contains("Assistant: Hola, soy el motor genómico."));
    assert!(content.contains("User: ¿Qué es mmap?"));

    // Cleanup
    let _ = std::fs::remove_file(tmp_input);
    let _ = std::fs::remove_file(tmp_output);
}

#[test]
fn test_cli_doctor_report() {
    // Valida que el motor de diagnóstico ejecute sin pánicos
    let report = _impl::compute::doctor::run_doctor();
    assert!(report.cpu_cores > 0);
}

#[test]
fn test_cli_models_listing() {
    let models_dir = Path::new("models");
    if models_dir.exists() {
        let res = _impl::io::models_cmd::list_models(models_dir);
        assert!(res.is_ok());
    }
}
