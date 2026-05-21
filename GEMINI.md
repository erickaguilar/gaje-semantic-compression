# 🧬 GAJE-Flow: Protocolo de Desarrollo y Estabilidad (v0.9.0)

Este archivo define las reglas de flujo de trabajo, la arquitectura del repositorio y los estándares técnicos para el proyecto **DNA Semantic Compression**. Es de cumplimiento obligatorio para todas las sesiones de desarrollo asistido.

## 1. Arquitectura del Repositorio (Organización v0.8.0)

El repositorio sigue una estructura lógica estricta. PROHIBIDO crear archivos en la raíz que no sean de configuración esencial.

- **`/docs`**: Documentación segmentada (`guides/`, `plans/`, `reports/`, `meta/`, `research/`).
- **`/scripts`**: Utilidades de entrenamiento y mantenimiento (`maintenance/`).
- **`/data`**: Centralización de datos generados (`training/`, `experiments/`, `datasets/`).
- **`/examples`**: Demos categorizadas (`core_demos/`, `visual_demos/`, `legacy_research/`).
- **`/tests`**: Suite de validación (`unit/`, `integration/`, `metrics/`, `training/`).
- **`/benchmarks`**: Evaluación de rendimiento con logs centralizados en `benchmarks/logs/`.
- **`/src` & `/python`**: Núcleo nativo (Rust) y lógica de investigación/puente (Python).

## 2. Reglas de Oro para el Agente (Gemini CLI)

1.  **Aislamiento y Ramas:** Ante cambios significativos, usa ramas `feature/` o `fix/`. La rama `develop` es sagrada.
2.  **Validación Pre-Merge:** Obligatorio ejecutar:
    - `cargo build --release` (Verificación nativa).
    - `pytest tests/integration/test_integration_v060.py` (Estabilidad funcional).
3.  **Mandato de Estabilidad de Memoria:**
    - Prohibidas las pre-asignaciones masivas de tensores `f32` en el `forward`.
    - Priorizar el uso de punteros y memoria compartida (`Arc<Vec<u8>>`).
4.  **Mantenimiento de la Estructura:** Cualquier archivo nuevo debe ser ubicado en su subdirectorio correspondiente según la arquitectura definida en la sección 1.

## 3. Estado Técnico y Metas (v0.9.0)

- **Soberanía Nativa Alcanzada:** El motor es 100% Rust. (Ver `docs/plans/MASTER_STRATEGY_v1.0.md`).
- **Arquitectura SoA:** Implementada para máximo throughput SIMD NEON.
- **Rendimiento Validado:** >200,000 registros/seg en búsqueda asimétrica (ADC) en ARM.
- **Compresión Extrema:** 2 bits por peso con refinamiento epigenético (4/6-bit).

## 4. Estilo de Commits

Seguir el estándar de **Conventional Commits**:
- `feat(scope):`, `fix(scope):`, `perf(scope):`, `docs(scope):`, `chore(scope):`.

---
*Este protocolo es vinculante y actualiza todas las versiones previas.*
