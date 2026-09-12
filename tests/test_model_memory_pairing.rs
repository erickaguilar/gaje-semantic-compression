use _impl::compute::island::IslandOrchestrator;
use _impl::server::calibrate_memory_threshold;
use _impl::server::api::get_loaded_memory_info;
use std::path::Path;

#[test]
fn test_resolve_memory_dir() {
    let p_laser = Path::new("models/born/max_laser.gaje");
    let resolved = IslandOrchestrator::resolve_memory_dir(p_laser);
    assert_eq!(resolved, Path::new("models/born/max_laser_memory"));
    assert!(resolved.is_dir(), "max_laser_memory debe existir en disco");

    let p_new = Path::new("models/born/organism_x.gaje");
    let resolved_new = IslandOrchestrator::resolve_memory_dir(p_new);
    assert_eq!(resolved_new, Path::new("models/born/organism_x_memory"));
}

#[test]
fn test_calibrate_memory_threshold_empirical() {
    // Modelos basados en max / dim <= 384 (Weighted pooling nativo: tau* = 0.42)
    assert_eq!(calibrate_memory_threshold("max.gaje", 256), 0.42);
    assert_eq!(calibrate_memory_threshold("models/born/max_laser.gaje", 384), 0.42);

    // Modelos basados en Qwen2.5 (Con whitening .mu.bin: tau* = 0.33)
    assert_eq!(calibrate_memory_threshold("models/production/qwen2_5_0_5b_q2_0.gaje", 896), 0.33);

    // Modelos SmolLM / Pico (Con whitening .mu.bin: tau* = 0.27)
    assert_eq!(calibrate_memory_threshold("models/production/gaje_pico_135m.gaje", 576), 0.27);
}

#[test]
fn test_load_paired_for_model_max_laser() {
    let p_laser = Path::new("models/born/max_laser.gaje");
    let threshold = calibrate_memory_threshold(p_laser.to_str().unwrap(), 384);
    let (orch, mem_dir) = IslandOrchestrator::load_paired_for_model(p_laser, 384, threshold);

    assert_eq!(mem_dir, Path::new("models/born/max_laser_memory"));
    assert_eq!(orch.dim, 384);
    assert_eq!(orch.min_similarity, 0.42);

    // Documental debe tener al menos 50 hechos
    assert!(
        orch.documental.entries.len() >= 50,
        "max_laser_memory debe contener >= 50 hechos documentales, encontrados: {}",
        orch.documental.entries.len()
    );

    // Validar JSON de telemetría API
    let json_info = get_loaded_memory_info(&orch, &mem_dir, threshold);
    assert_eq!(json_info["status"], "connected");
    assert_eq!(json_info["dim"], 384);
    assert!(json_info["total_facts"].as_u64().unwrap() >= 50);
    assert!((json_info["memory_threshold"].as_f64().unwrap() - 0.42).abs() < 1e-4);
}
