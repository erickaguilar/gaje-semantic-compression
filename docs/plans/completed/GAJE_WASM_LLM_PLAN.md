# Plan: WebAssembly + LLM en el navegador

## Objetivo
Portar el motor de inferencia Rust de GAJE (`gaje-core`) a `wasm32` para ejecutar un LLM localmente en el navegador, reemplazando la dependencia del servidor Python (`server.py` + `model_manager.py`) por un cliente autónomo.

## Evaluación de arquitectura (resumen)

**Viable, con reescrituras focalizadas en la capa I/O, no en la matemática.**

Ventajas del diseño actual:
- `pyo3` está aislado tras `#[cfg(feature = "python")]` (`src/lib.rs`) → el core es `rlib` puro.
- Ya existe fallback escalar SIMD en `kernels/genomic.rs` (`#[cfg(not(any(x86_64, aarch64)))]`) → cubre `wasm32`.
- El formato `.flat` zero-copy se traduce bien a `ArrayBuffer`.

Obstáculos para `wasm32`:

| Dependencia | Problema | Solución |
|---|---|---|
| `memmap2` (`flat_reader.rs`) | No-op en wasm32 | `fetch` → `ArrayBuffer`, usar como slice; quitar `unsafe mmap` |
| `redb` (`core/db.rs`) | Necesita filesystem | En wasm solo cargar `.flat`, no `.gaje`/redb |
| `rayon` | Necesita threads | `wasm-bindgen-rayon` (SharedArrayBuffer + COOP/COEP) o single-thread |
| `libc` / `ctrlc` / `indicatif` | No compatible | Excluir con `#[cfg(not(target_arch = "wasm32"))]` |
| `tokenizers` (HF) | Compila pero pesado | `tokenizer.json` embebido |

**Cuello de botella real: tamaño del modelo.**
- Límite wasm32 = 4GB; el navegador maneja mal >1-1.5GB.
- Candidatos viables: `smollm2_4bit.gaje.flat` (472MB) y `qwen2_0_5b_q4_0_q8_0_embd.gaje.flat` (499MB).
- Modelos 1.5B+ (2.6GB) quedan fuera del alcance del navegador con descarga completa.

## Arquitectura objetivo

```
NAVEGADOR
├── Web UI (index.html / chat.js)
├── Web Worker
│   └── WASM gaje-core (wasm32)
│       ├── kernels (fallback escalar)
│       ├── flat_reader → ArrayBuffer
│       ├── forward_core / sampler
│       └── wasm-bindgen
├── Model .flat (servido estático / CDN)
└── Memory (ArrayBuffer compartido)
```

Clave: `load_genomic_auto → GajeFlatFileReader`. En WASM solo se sustituye `mmap` por slice de `ArrayBuffer`; el resto de la cadena (`forward_core` → bloques → atención → sampler) funciona igual.

## Roadmap

### Fase 0 — Compilación WASM
- Agregar target `wasm32-unknown-unknown`.
- Aislar con `#[cfg(not(target_arch = "wasm32"))]`: `libc`, `ctrlc`, `indicatif`, `redb`.
- Desactivar `rayon` (single-thread) o feature `rayon-wasm`.
- Objetivo: `cargo build --target wasm32-unknown-unknown` compila sin errores.

### Fase 1 — Carga
- `wasm-bindgen` exponiendo `load_from_bytes(&[u8])` y `generate(prompt)`.
- Servir el `.flat` por endpoint estático (`models/smollm2_4bit.gaje.flat`).

### Fase 2 — Web Worker
- Mover el WASM a un Web Worker (no bloquear el hilo principal).
- Opcional: `wasm-bindgen-rayon` con `SharedArrayBuffer` para multihilo.

### Fase 3 — Optimización
- Multihilo masivo con `wasm-bindgen-rayon`.
- Considerar `simd128` / `relaxed-simd` (wasm32) para recuperar rendimiento tipo AVX.

## Recomendación de modelo inicial
Apuntar a `smollm2_4bit.gaje.flat` (472MB). Los modelos 1.5B+ requieren otro enfoque (GGUF + mmap vía OPFS, estilo transformers.js).