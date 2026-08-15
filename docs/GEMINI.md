# 🧬 GAJE-Flow: Protocolo de Desarrollo y Estabilidad (v1.0.0-alpha: Silver Adult)

Este archivo define las reglas de flujo de trabajo, la arquitectura del repositorio y los estándares técnicos para el proyecto **DNA Semantic Compression**. Es de cumplimiento obligatorio para todas las sesiones de desarrollo asistido.

## 1. Arquitectura del Repositorio (Organización v0.9.0)

El repositorio sigue una estructura lógica estricta. PROHIBIDO crear archivos en la raíz que no sean de configuración esencial.

- **`/docs`**: Documentación segmentada (`guides/`, `plans/`, `reports/`, `meta/`, `research/`). Archivos obsoletos se mueven a `docs/archive/`.
- **`/scripts`**: Utilidades de mantenimiento (`maintenance/`).
- **`/data`**: Centralización de datos (`training/`, `experiments/`, `datasets/`).
- **`/examples`**: Demos categorizadas (`core_demos/`, `visual_demos/`). Uso obligatorio de `silver_adult_anchored.gaje` para validación.
- **`/tests`**: Suite de validación (`unit/`, `integration/`, `metrics/`).
- **`/benchmarks`**: Evaluación de rendimiento con logs en `benchmarks/logs/`.
- **`/src` & `/python`**: Núcleo nativo (Rust) y puente de investigación (Python).

## 2. Reglas de Oro para el Agente (Gemini CLI)

1.  **Aislamiento y Ramas:** Ante cambios significativos, usa ramas `feature/` o `fix/`. La rama `develop` es sagrada.
2.  **Validación Pre-Merge:** Obligatorio ejecutar:
    - `cargo build --release` (Verificación nativa).
    - `pytest tests/integration/test_integration_v060.py` (Estabilidad funcional).
3.  **Mandato de Estabilidad de Memoria:**
    - Prohibidas las pre-asignaciones masivas de tensores `f32` en el `forward`.
    - Priorizar el uso de punteros y memoria compartida (`Arc<Vec<u8>>`).
4.  **Soberanía del Tooling (Anti-Python):**
    - PROHIBIDO crear nuevos scripts de utilidad en Python. Cualquier funcionalidad administrativa debe ser un subcomando en `gaje-cli`.
5.  **Mantenimiento de la Estructura:** Cualquier archivo nuevo debe ser ubicado en su subdirectorio correspondiente. No ensuciar la raíz.
6.  **Soberanía Nativa y Colisiones:** PROHIBIDO crear carpetas en `python/gaje/` que colisionen con nombres de módulos nativos (ej. `_impl`). Usar prefijos `_legacy_` si es necesario.
7.  **Robustez Rust:** El código nativo debe usar siempre validación de límites (`.get().unwrap_or()`) para prevenir pánicos por discrepancias epigenéticas.
8.  **Benchmarking Interactivo:** Las herramientas deben aceptar entrada de texto y tener timeouts para no bloquear el flujo de trabajo.

## 3. Estado Técnico y Estándar Empírico (Operación Rescate)

El proyecto se encuentra en estado **Alpha Real**. La infraestructura base compila, pero la validación semántica falla (PPL ~572).

*   **Mandato de Verdad Empírica:** ESTÁ PROHIBIDO declarar cualquier fase o característica como "completada" basándose únicamente en que el código compila.
*   **Certificación Requerida:** Toda declaración de éxito debe cumplir los umbrales definidos en `docs/meta/EMPIRICAL_TRUTH_STATE.md`.
*   **Fundamentación Matemática:** Antes de intentar reducir la PPL, es OBLIGATORIO consultar `docs/research/FORMALIZATION_LAYER.md` para entender el equilibrio Lagrangiano requerido entre movilidad (energía cinética) y precisión (potencial semántico).
*   **Prioridad Actual:** Congelamiento de nuevas características (Island Model, RAG) hasta superar la Certificación Semántica (Nivel 2: lograr PPL < 15.0).
*   **Lectura Obligatoria:** Antes de iniciar desarrollo, consultar `docs/meta/EMPIRICAL_TRUTH_STATE.md` para conocer la realidad matemática y funcional del modelo.

## 4. Estilo de Commits

Seguir el estándar de **Conventional Commits**:
- `feat(scope):`, `fix(scope):`, `perf(scope):`, `docs(scope):`, `chore(scope):`.

## 5. Ciclo de Desarrollo Integral (SDD -> BDD -> TDD)

1.  **SDD:** Diseño de arquitectura y contratos técnicos.
2.  **BDD:** Escenarios *Given-When-Then*. Ver [docs/bdd/BDD_GUIDE.md](docs/bdd/BDD_GUIDE.md).
3.  **TDD:** Implementación iterativa Red-Green-Refactor.

## 6. Pilares Arquitectónicos de Fase Circular

- **Stability Anchors:** Inyección estratégica de precisión para evitar la fragmentación semántica.
- **K-WTA Lateral Inhibition:** Filtrado de ruido mediante competencia temporal.
- **Direct Neural Ingestion (DNI):** Capacidad de inyección de conocimiento directo en el genoma.

## 7. Próximos Pasos (Q3 2026)

- **Island Model:** Implementación de evolución distribuida por nichos semánticos.
- **Native Semantic RAG:** Recuperación de información integrada directamente en los kernels de Rust.

---
*Este protocolo es vinculante y actualiza todas las versiones previas (Protocolo v1.3).*
