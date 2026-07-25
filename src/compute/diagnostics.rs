use std::sync::atomic::{AtomicU64, Ordering};

/// 📊 Diagnostics: Seguimiento de métricas de eficiencia en tiempo real.
pub struct Diagnostics {
    pub total_calculations: AtomicU64,
    pub skipped_calculations: AtomicU64,
}

static GLOBAL_DIAGNOSTICS: Diagnostics = Diagnostics {
    total_calculations: AtomicU64::new(0),
    skipped_calculations: AtomicU64::new(0),
};

/// Registra cálculos totales y omitidos para el reporte de Sparsity.
pub fn report_sparsity(total: u64, skipped: u64) {
    GLOBAL_DIAGNOSTICS
        .total_calculations
        .fetch_add(total, Ordering::Relaxed);
    GLOBAL_DIAGNOSTICS
        .skipped_calculations
        .fetch_add(skipped, Ordering::Relaxed);
}

/// Obtiene el porcentaje de Sparsity (Escasez) actual.
pub fn get_sparsity_report() -> f32 {
    let total = GLOBAL_DIAGNOSTICS
        .total_calculations
        .load(Ordering::Relaxed);
    let skipped = GLOBAL_DIAGNOSTICS
        .skipped_calculations
        .load(Ordering::Relaxed);

    if total == 0 {
        return 0.0;
    }
    (skipped as f32 / total as f32) * 100.0
}

/// Resetea los contadores de diagnóstico.
pub fn reset_diagnostics() {
    GLOBAL_DIAGNOSTICS
        .total_calculations
        .store(0, Ordering::Relaxed);
    GLOBAL_DIAGNOSTICS
        .skipped_calculations
        .store(0, Ordering::Relaxed);
}
