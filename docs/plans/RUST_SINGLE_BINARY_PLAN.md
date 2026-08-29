# Plan: Reducción de dependencia de Python a favor de Rust single-binary

> **Estado:** propuesto (no implementado).
> **Objetivo:** prescindir de Python lo máximo posible, dejando a `gaje-cli` como el
> ejecutable autónomo del motor (inferencia, entrenamiento, evolución, DNI, import).

## 1. Diagnóstico actual (verificado en el código)

### 1.1 El motor ya es 100% Rust
- `gaje-cli` es un **single-binary funcional y puro**: `cargo build --bin gaje-cli`
  (sin feature `python`) compila sin errores.
- El binario usa directamente `GajeTokenizer`, `GenomicLLM`, `NativeLoader`,
  `GGUFLoader` y las funciones de `compute/math` (`gaje-cli.rs:269-261, 293, 571`).
- **No existe** invocación a `subprocess`/`python` desde `src/`. Cero dependencia
  en runtime.
- `pyo3` es dependency **opcional** (`Cargo.toml:22, 44-45`); el feature `python`
  solo construye la extensión cdylib `_impl`.
- Todos los bindings `#[pymethods]`/`#[pyfunction]` están protegidos por
  `#[cfg(feature = "python")]`, así que el módulo `python.rs` no afecta al binario.

### 1.2 Lo que sigue siendo Python
Inventario real del repo (excluye `.venv`, `node_modules`):
- **249** archivos `.py`.
- **55** importan la extensión PyO3 (`import _impl` / `from _impl`).
- **14** invocan subprocesos o el binario.
- `python/` (23 scripts), `scripts/` (~130 scripts) y `tests/` (mix `.py` + `.rs`)
  siguen siendo el flujo de trabajo de build/validación/benchmark.
- `benchmarks/` contiene decenas de `.py` de diagnóstico y certificación que usan `_impl`.

## 2. Objetivo y alcance

### 2.1 Alcance objetivo
Que el **flujo de producción** (inferencia, generación, entrenamiento, evolución,
DNI, import GGUF→GAJE, inspect, eval) dependa únicamente de `gaje-cli`. Python queda
relegado a herramientas de investigación/benchmark opcionales, nunca como requisito
para correr el motor.

### 2.2 Alcance fuera (explícitamente NO incluido)
- No se elimina `_impl` ni el feature `python`: sigue siendo útil para investigación
  y compatibilidad con los `.py` de benchmarking.
- No se migran los 249 `.py` uno a uno: se priorizan los de **flujo de trabajo**
  (build/export/validación), dejando los experimentos de investigación en Python.

## 3. Estrategia: subcomandos de `gaje-cli`

En lugar de un `.py` por tarea, el binario expone subcomandos que cubren los flujos
que hoy orquestan los `.py` más importantes. Estructura CLI objetivo:

```text
gaje-cli
├── chat                                     # REPL interactivo en terminal (ya existe)
├── serve [--port 8080]                      # Servidor HTTP nativo con Web UI Chat embebida (Zero-Disk)
├── doctor                                   # Diagnóstico de hardware y SIMD (ya existe)
├── models [--inspect <m>]                   # Catálogo e inspección estructural (ya existe)
├── pull <model>                             # Descarga nativa multi-stream (ya existe)
├── bench                                    # Benchmark TTFT y velocidad (ya existe)
├── epoch                                    # Gestión de épocas .gmem v2 (ya existe)
│
└── (NUEVO) subcomandos de utilidad para reemplazar scripts clave:
    ├── export-flat <model> --output <out.flat>   # reemplaza export_*.py
    ├── benchmark <corpus> [--ppl] [--latency]    # reemplaza engine_benchmark.py / ppl_suite.py
    ├── dataset-build <inputs...> --output <txt>  # reemplaza generate_synthetic_*.py / create_*dataset.py
    └── audit <model> [--coherence] [--entropy]   # reemplaza los *audit*.py / check_*.py
```

## 4. Arquitectura de Web UI Embebida (Chat Exclusivo)

Para garantizar un binario completamente autocontenido y liviano que pueda distribuirse y ejecutarse en cualquier entorno sin requerir carpetas externas de assets:

### 4.1 Alcance del Asset Embedding:
* **Incluido (Core Chat):**
  * `index.html` (interfaz principal de chat, HUD de telemetría y configuración).
  * `manifest.json` y `sw.js` (PWA / Service Worker).
  * `static/css/` (estilos base, chat, temas Y2K / Scandinavian).
  * `static/js/` (lógica de streaming, chat y renderizado de telemetría).
  * `static/icons/` (sprites SVG y favicon).
  * `static/wasm/` (runtimes WebAssembly livianos si aplica).
* **Excluido explícitamente (Reducción de huella y separación de responsabilidades):**
  * `docs.html` (centro de documentación interactiva).
  * `architecture.html` y `architecture_graph.json` (visualizador de grafos).
  * Scripts Python del servidor backend heredado (`server.py`, `model_manager.py`, etc.).
  * Próximas páginas auxiliares o herramientas de análisis exploratorio.

### 4.2 Mecanismo de Despacho Híbrido (`src/server/static_files.rs`):
1. **Modo Producción / Autónomo (Memoria Directa):**
   * Se compila con `rust-embed` filtrando los archivos de `examples/ui/web_ui/` permitidos.
   * Si no se especifica `--static-dir` o la ruta física no existe, el servidor despacha los recursos directamente desde la memoria (`.rodata`), con latencia cero y sin dependencias de disco.
2. **Modo Desarrollo (Disco Local):**
   * Si se proporciona una ruta válida en `--static-dir` (o se detecta el entorno local de desarrollo), el servidor prioriza la lectura desde el sistema de archivos para permitir recarga en vivo de estilos y scripts.
3. **Restricción de Rutas:**
   * Cualquier petición a `/docs`, `/architecture` o recursos excluidos devolverá `404 Not Found` en el servidor embebido estándar de producción.

---

## 5. Fases y esfuerzo estimado

### Fase 1 — Servidor Autónomo y Web UI Embebida de Chat (✅ COMPLETADA)
- [x] Integrar `rust-embed` en `src/server/` con inclusión del Chat (`index.html`, `static/`, PWA).
- [x] Adaptar `src/server/static_files.rs` para servir desde memoria con fallback transparente a disco.
- [x] Probar ejecución de `gaje-cli serve` en un directorio aislado sin carpetas de assets (`HTTP 200 OK` en memoria, `HTTP 404` en `/docs`).

### Fase 2 — Subcomandos de utilidad CLI (✅ COMPLETADA)
- [x] Añadir al CLI: `export-flat`, `benchmark`, `dataset-build`, `audit`.
- [x] Reutilizar lógica nativa en Rust:
  - `export-flat`: serializador zero-copy mmap SIMD 64B (`src/io/flat_writer.rs`).
  - `benchmark`: cálculo de TTFT, TPS y Perplejidad PPL sobre corpus de texto/jsonl (`src/io/cli_tools.rs`).
  - `dataset-build`: normalizador de pares conversacionales/instrucciones con tokenización GTOK (`src/io/cli_tools.rs`).
  - `audit`: verificación exhaustiva de 0 NaNs/Infs, consistencia estructural y entropía (`src/io/cli_tools.rs`).
- [x] Implementar dispatcher y argumentos clap tipados en `src/bin/gaje-cli.rs`.

### Fase 3 — Sustituir los `.py` de flujo de trabajo (✅ COMPLETADA)
- [x] Mapear los `.py` de `scripts/` a subcomandos CLI (`export-flat`, `pull`, `models`, `benchmark`, `dataset-build`, `audit`, `doctor`).
- [x] Crear `scripts/README.md` con la matriz integral de equivalencias y marcar scripts obsoletos con avisos de deprecación.
- [x] `scripts/*.sh` (ej. `download_hf_model.sh`) actualizados para priorizar el motor nativo `gaje-cli`.

### Fase 4 — Migrar la suite de validación (✅ COMPLETADA)
- [x] Implementar tests de integración nativos en Rust (`tests/cli_standalone_test.rs`) para validar CLI, dataset-build, doctor y models.
- [x] Reemplazar validaciones de producción dependientes de Python por `gaje-cli benchmark` y `gaje-cli audit`.
- [x] Documentar `gaje-cli` en `README.md` y `scripts/README.md` como el artefacto y punto de entrada soberano y único de producción.

---

## 6. Criterio de "hecho" (Definition of Done) — 100% Cumplido
- [x] `gaje-cli serve` levanta la Web UI de Chat en cualquier máquina/directorio sin requerir archivos externos en disco.
- [x] `docs.html`, `architecture.html` y páginas pesadas quedan excluidas del empaquetado del binario.
- [x] `gaje-cli` (release) ejecuta los flujos de producción e inferencia sin Python presente.
- [x] `python/` y `scripts/` ya no son requisito para build/export/validación core (reemplazados por `gaje-cli`).
- [x] README/INDEX documentan `gaje-cli` como la pieza única de producción.

---

## 7. Riesgos y mitigaciones
- **Tamaño del binario:** Al excluir `docs.html`, `architecture.html` y datasets JSON de grafos, el incremento en el tamaño del ejecutable por embeber el chat es mínimo (< 500 KB).
- **Regresión de paridad:** Los `.py` de benchmark comparan contra HF/torch. Mitigación: mantener esos como referencia mientras se valida el subcomando `benchmark`.
- **Boundary de I/O:** Todos los formatos binarios (`.flat`, `.gaje`, gguf, `.gmem`) ya tienen lectores/escritores Rust en `src/io/`.

---

## 8. Anexo: inventario rápido
| Área | # scripts | Depende de `_impl` | Acción propuesta |
|---|---|---|---|
| `src/` (motor) | — | no | Ya Rust, sin cambios |
| `src/server/` (Web UI Chat) | — | no | Embebido en binario (`rust-embed`) |
| `python/` | 23 | mayormente | Migrar flujo; investigar en Python |
| `scripts/` | ~130 | muchos | Subcomandos CLI + obsoletos |
| `tests/` | mix | ~55 | Migrar a `cargo test` / subcomandos |
| `benchmarks/` | decenas | muchos | Subcomando `benchmark`; investigación en Python |

