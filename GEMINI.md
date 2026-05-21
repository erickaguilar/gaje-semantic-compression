# 🧬 GAJE-Flow: Protocolo de Desarrollo y Estabilidad (v0.7.0)

Este archivo define las reglas de flujo de trabajo, la arquitectura del repositorio y los estándares técnicos para el proyecto **DNA Semantic Compression**. Es de cumplimiento obligatorio para todas las sesiones de desarrollo asistido.

## 1. Arquitectura del Repositorio (Organización v0.7.0)

El repositorio sigue una estructura lógica estricta. PROHIBIDO crear archivos en la raíz que no sean de configuración esencial.

- **`/docs`**: Documentación segmentada (`guides/`, `plans/`, `reports/`, `meta/`, `research/`).
- **`/scripts`**: Utilidades de entrenamiento, destilación y mantenimiento (`maintenance/`).
- **`/data`**: Centralización de datos generados (`training/`, `experiments/`, `datasets/`).
- **`/examples`**: Demos categorizadas (`core_demos/`, `visual_demos/`, `legacy_research/`).
- **`/tests`**: Suite de validación (`unit/`, `integration/`, `metrics/`, `training/`).
- **`/benchmarks`**: Evaluación de rendimiento con logs centralizados en `benchmarks/logs/`.
- **`/src` & `/python`**: Núcleo nativo (Rust) y lógica de investigación (Python).

## 2. Reglas de Oro para el Agente (Gemini CLI)

1.  **Aislamiento y Ramas:** Ante cambios significativos, usa ramas `feature/` o `fix/`. La rama `develop` es sagrada.
2.  **Validación Pre-Merge:** Obligatorio ejecutar:
    - `cargo build --release` (Verificación nativa).
    - `pytest tests/integration/test_integration_v060.py` (Estabilidad funcional).
3.  **Mandato de Estabilidad de Memoria:**
    - Prohibidas las pre-asignaciones masivas de tensores `f32` en el `forward`.
    - Priorizar el uso de punteros y memoria compartida (`Arc<Vec<u8>>`).
4.  **Mantenimiento de la Estructura:** Cualquier archivo nuevo debe ser ubicado en su subdirectorio correspondiente según la arquitectura definida en la sección 1.

## 3. Estado Técnico y Metas (v0.7.0)

- **Soberanía Nativa:** Transición activa hacia Rust 100% (Ver `docs/plans/NATIVE_SOVEREIGNTY_PLAN.md`).
- **Rendimiento Validado:** >200,000 registros/seg en búsqueda asimétrica (ADC) en ARM/NEON.
- **Precisión:** Recall@10 > 82% en vectores de 768d (SBERT).
- **Compresión:** 2 bits por peso (16x reducción de RAM).

## 4. Estilo de Commits

Seguir el estándar de **Conventional Commits**:
- `feat(scope):`, `fix(scope):`, `perf(scope):`, `docs(scope):`, `chore(scope):`.

---
*Este protocolo es vinculante y actualiza todas las versiones previas.*
