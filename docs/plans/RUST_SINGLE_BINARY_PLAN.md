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
├── <model.gaje|.gguf> --prompt "..."        # inferencia/generación (ya existe)
├── --import <gguf> --output <out.gaje>      # import GGUF (ya existe)
├── --inspect <model>                        # metadatos (ya existe)
├── --eval <corpus>                          # perplejidad (ya existe)
├── --tokenize <text>                        # tokenizar (ya existe)
├── --evolve "<target>" --gens N             # evolución (ya existe)
├── --train <dataset> --epochs N             # entrenamiento (ya existe)
├── ingest --model <m> --file <doc>          # DNI ingest (ya existe)
├── --iqat --teacher <gguf> --teacher-tok <tok>  # IQAT (ya existe)
│
└── (NUEVO) subcomandos de utilidad para reemplazar scripts clave:
    ├── export-flat <model> --output <out.flat>   # reemplaza export_*.py
    ├── benchmark <corpus> [--ppl] [--latency]    # reemplaza engine_benchmark.py / ppl_suite.py
    ├── dataset-build <inputs...> --output <txt>  # reemplaza generate_synthetic_*.py / create_*dataset.py
    └── audit <model> [--coherence] [--entropy]   # reemplaza los *audit*.py / check_*.py
```

## 4. Fases y esfuerzo estimado

### Fase 1 — Verificar/consolidar el single-binary (esfuerzo BAJO, ~0.5-1 día)
- Confirmar `cargo build --release --bin gaje-cli` y ejecutar el flujo end-to-end
  (import GGUF → generar → evaluar) sin Python.
- `cargo test --lib` (26 tests) + `cargo check --features python` como regresión.
- Documentar el binario como única pieza de producción.

### Fase 2 — Subcomandos de utilidad (esfuerzo MEDIO, ~2-4 días)
- Añadir al CLI: `export-flat`, `benchmark`, `dataset-build`, `audit`.
- Reutilizar lógica ya existente en Rust:
  - Export flat: `io/flat_writer.rs` (`save_genomic_model` / `GajeFlatFileWriter`).
  - Benchmark/PPL: lógica ya en `gaje-cli.rs` (eval) y `compute/metrics`.
  - Dataset: tokenizar + normalizar texto (crédito a `GajeTokenizer`).
  - Audit: `compute/math` (entropías, MSE, similitud) y `core/index`.
- Implementar un dispatcher de subcomandos simple (sin crate nuevo, `match` sobre args).

### Fase 3 — Sustituir los `.py` de flujo de trabajo (esfuerzo MEDIO-ALTO, ~1 semana)
- Mapear los `.py` de `scripts/` a subcomandos CLI; marcar como obsoletos los
  cubiertos.
- `scripts/*.sh` existentes se reescriben para llamar solo a `gaje-cli` en vez de
  `python train_*.py`.

### Fase 4 — Migrar la suite de validación (esfuerzo ALTO, continuo)
- `tests/*.py` y `benchmarks/*.py` (55 que importan `_impl`) se migran a:
  - `cargo test` (tests Rust), o
  - subcomandos `benchmark`/`audit` del binario.
- Los experimentos de investigación que usan PyO3 (MCTS, Monte Carlo, topología)
  **se conservan en Python** como herramientas opcionales, documentadas como tal.

## 5. Criterio de "hecho" (Definition of Done)
- [ ] `gaje-cli` (release) ejecuta los flujos de producción sin Python presente.
- [ ] `python/` y `scripts/` ya no son requisito para build/export/validación core.
- [ ] La suite de validación principal es `cargo test` + subcomandos `benchmark`/`audit`.
- [ ] README/INDEX documentan `gaje-cli` como la pieza única de producción y a los
      `.py` restantes como herramientas opcionales de investigación.

## 6. Riesgos y mitigaciones
- **Regresión de paridad:** los `.py` de benchmark comparan contra HF/torch.
  Mitigación: mantener esos como referencia mientras se valida el subcomando
  `benchmark` contra los mismos umbrales.
- **Funcionalidad no cubierta por el CLI:** algunos scripts hacen tareas muy
  específicas (download HF, topología). Mitigación: conservarlos en Python y
  documentarlos, no bloquear la migración core.
- **Boundary de I/O:** los scripts leen/escriben `.flat`, `.gaje`, gguf, bases redb.
  Mitigación: todos esos formatos ya tienen lectores/escritores Rust
  (`io/`), solo falta exponerlos vía subcomandos.

## 7. Anexo: inventario rápido
| Área | # scripts | Depende de `_impl` | Acción propuesta |
|---|---|---|---|
| `src/` (motor) | — | no | Ya Rust, sin cambios |
| `python/` | 23 | mayormente | Migrar flujo; investigar en Python |
| `scripts/` | ~130 | muchos | Subcomandos CLI + obsoletos |
| `tests/` | mix | ~55 | Migrar a `cargo test` / subcomandos |
| `benchmarks/` | decenas | muchos | Subcomando `benchmark`; investigación en Python |
