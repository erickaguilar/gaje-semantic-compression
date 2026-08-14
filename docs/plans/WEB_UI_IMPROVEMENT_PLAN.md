# Plan de Mejora — `examples/ui/web_ui`

**Fecha:** 2026-08-13
**Rama:** `test/experimental`
**Estado:** Propuesto (pendiente de aprobación e implementación)
**Ámbitos:** Integración del diagrama de arquitectura · Calidad y robustez (backend + frontend) · Streaming real · Tests · Migración futura a Rust · Descarga de modelos desde la web · Uso local/WebAssembly (estilo Cactus Needle)

---

## 1. Contexto y objetivos

La carpeta `examples/ui/web_ui` contiene el portal web de GAJE: un chat genómico que se apoya en el motor nativo Rust (PyO3) mediante `server.py`, `model_manager.py` y `prompt_templates.py`. A día de hoy funciona, pero tiene varias limitaciones de experiencia de usuario, robustez y consolidación:

- El chat espera la respuesta completa de `/api/chat` de golpe (sin streaming).
- No hay historial de conversación (ni en cliente ni en servidor).
- Los parámetros de configuración están hardcodeados (puerto, rutas, límites).
- La salida del LLM se inserta con `innerHTML` sin escapar (riesgo XSS).
- El diagrama de arquitectura (ya creado en `architecture.html` + `architecture_graph.json`) vive como página separada, no integrado en el portal.
- No hay tests automatizados para el servidor ni el flujo de chat.
- El portal es una app Python local; no hay forma de **descargar modelos** desde la propia web ni de usarla de forma autocontenida/portable como un single-binary.

**Objetivos:**

1. Integrar el diagrama de arquitectura como vista dentro del portal, con navegación unificada y fuente de verdad única (`architecture_graph.json`).
2. Hacer el backend configurable, logueado y robusto.
3. Añadir streaming real de tokens por SSE para una UX fluida.
4. Blindar el frontend contra XSS y mejorar su calidad/estado de generación.
5. Añadir tests (unitarios del backend + E2E Playwright).
6. **Documentar la migración futura a Rust** del servidor (single-binary, sin runtime Python).
7. **Permitir descargar modelos desde la propia web** y **usar el portal localmente / vía WebAssembly** al estilo Cactus Needle.

**Resultado esperado:** un portal que no solo chatea, sino que muestra el sistema; un backend robusto y configurable; una conversación fluida con streaming; y un camino claro hacia una distribución autocontenida con descarga de modelos desde la web (descarga + ejecución local/WASM).

---

## 2. Estado actual (auditoría)

### 2.1 Arquitectura del portal

```
Navegador
   │  fetch (JSON / SSE)
   ▼
server.py  (http.server.SimpleHTTPRequestHandler, ThreadingTCPServer, puerto 8080)
   │
   ├── /api/models        GET   → model_manager.list_available_models()
   ├── /api/info          GET   → get_runtime_info()
   ├── /api/load_model    POST  → model_manager.get_model()
   ├── /api/chat          POST  → prompt_templates + rust_llm.generate_native_py()
   └── (estáticos) index.html, script.js, style.css, docs.*, architecture.*
```

- **Frontend:** `index.html` (SPA chat), `script.js` (fetch, tema, métricas, DNA), `style.css` (temas claro/oscuro, responsive).
- **Modelos:** `model_manager.py` — caché thread-safe, un solo modelo activo en RAM, carga lazy.
- **Prompts:** `prompt_templates.py` — ChatML/Gemma/fallback + stop tokens.

### 2.2 Diagrama de arquitectura (ya existente)

- `architecture_graph.json` — grafo `{meta, nodes, edges, flows:[{steps}]}` con 42 nodos, 59 aristas y 10 flujos. Fuente única de verdad.
- `architecture.html` — página SVG autocontenida, interactiva (resaltado de rutas, tooltips, zoom, responsive). Validada con Playwright.

### 2.3 Descarga de modelos y distribución (estado actual)

- Los modelos `.gaje`/`.flat` viven en `models/production/` y el server los **descubre localmente** (`list_available_models`). No hay catálogo remoto ni endpoint de descarga.
- El server actual **requiere runtime Python** (módulos `gaje`, `transformers`/`tokenizers`, numpy) — no es un single-binary portable.
- No hay target **WebAssembly** del motor; la UI es una app cliente/servidor, no un sandbox en el navegador.

---

## 3. Plan de trabajo por fases

### Fase 1 — Integración del diagrama de arquitectura

**Objetivo:** el diagrama pasa a ser una vista del portal (pestaña), no una página suelta.

| # | Tarea | Archivos |
|---|-------|----------|
| 1.1 | Añadir navegación por pestañas **Chat** / **Arquitectura** en `index.html`. | `index.html` |
| 1.2 | Convertir el diagrama en una vista embebible (`architecture_view.js` + `<section id="arch-view">`) montada bajo demanda. Mantener `architecture_graph.json` como fuente de datos. | `architecture_view.js`, `architecture.html` → adaptar, `index.html` |
| 1.3 | Compartir estado del flujo activo entre vistas (sin duplicar lógica). | `script.js`, `architecture_view.js` |
| 1.4 | Vínculo contextual: al hacer chat, resaltar la ruta del flujo `inference` en el diagrama. | `script.js`, `architecture_view.js` |
| 1.5 | Aplicar el tema claro/oscuro también a la vista de arquitectura. | `style.css`, `architecture_view.js` |

**Entregable:** portal unificado con navegación, diagrama integrado y contexto del modelo cargado.

---

### Fase 2 — Robustez del backend

#### 2.1 Configuración por variables de entorno

| Variable | Default | Uso |
|----------|---------|-----|
| `GAJE_PORT` | `8080` | Puerto HTTP |
| `GAJE_MODELS_ROOT` | `<proyecto>/models` | Directorio raíz de modelos |
| `GAJE_MAX_TOKENS` | `512` | Límite de tokens por respuesta |
| `GAJE_TEMPERATURE` | `0.2` | Sampling estable (evita loops/hallucination) |
| `GAJE_TOP_P` | `0.9` | Nucleus sampling |
| `GAJE_REP_PENALTY` | `1.1` | Repetition penalty |
| `GAJE_LOG_LEVEL` | `INFO` | Nivel de logging |

#### 2.2 Streaming real de tokens (SSE)

Viable sin tocar Rust — ver Sección 4. Nuevo endpoint `POST /api/chat/stream` que reutiliza `llm.generate()` (generador de `stabilized.py`) y hace `yield` por token. `/api/chat` queda como fallback no-streaming.

#### 2.3 Historial de conversación multi-turno

- Mantener cola de mensajes por sesión en el servidor (respetando `context_budget`).
- Incluir el historial en el prompt para dar continuidad a la conversación.

#### 2.4 Logging estructurado

- Sustituir `print` por `logging` con niveles, timestamps y formato uniforme.

#### 2.5 Validación y seguridad

- Validar `model_name` (evitar path traversal en `find_model_path`).
- Manejar `Content-Length` inválido y cuerpos vacíos en `_read_json_body`.
- Timeouts en generación.
- Sanitización de salida del LLM en el frontend (Sección 3.1).

**Archivos:** `server.py`, `model_manager.py`, `prompt_templates.py`.

---

### Fase 3 — Calidad del frontend

| # | Tarea |
|---|-------|
| 3.1 | **XSS:** sustituir `innerHTML` de `addMessage` por `textContent`/`escapeHtml()` para contenido del LLM y del usuario. |
| 3.2 | **Streaming en UI:** `sendMessage()` consume SSE y renderiza el mensaje bot incrementando en vivo; botón **Detener** que cierra la conexión. |
| 3.3 | **Historial local:** persistir conversación en `localStorage` y recuperarla al recargar. |
| 3.4 | **Estados de generación:** indicador "generando…" animado + feedback de errores más visible. |
| 3.5 | **Accesibilidad:** `aria-live` en el chat, roles y atributos `aria-label` en controles. |
| 3.6 | **Carga de modelo:** barra de progreso real (no solo mensaje) y manejo de estados intermedios. |

**Archivos:** `script.js`, `index.html`, `style.css`.

---

### Fase 4 — Tests

| # | Tipo | Alcance | Ubicación sugerida |
|---|------|---------|--------------------|
| 4.1 | Unit | `format_prompt`, `get_stop_tokens`, `list_available_models`, validación de `model_name`, lógica de métricas | `tests/` (pytest) |
| 4.2 | E2E | Flujo de chat con streaming (mock del LLM) y selección de flujo en el diagrama | `tests/ui_e2e/web_ui.test.js` (Playwright) |
| 4.3 | CI | Job que ejecute lint (ruff/pre-commit) + tests del server | workflow de CI |

---

### Fase 5 — Migración futura del servidor a Rust (single-binary)

**Objetivo:** reemplazar `server.py` (HTTP + orquestación) por un binario Rust autocontenido, sin runtime Python, con streaming nativo sin overhead FFI.

**Por qué es viable (verificado en el código fuente):** el motor, el tokenizer y la carga de modelos ya son 100% nativos.

| Pieza | Estado en Rust | Ubicación |
|-------|----------------|-----------|
| Motor de inferencia | ✅ Nativo (`GenomicLLM`, `generate_native_py`, `forward`) | `src/nn/llm.rs` |
| Tokenizer | ✅ Nativo (`GajeTokenizer`, envuelve crate `tokenizers 0.21`) | `src/core/tokenizer.rs` |
| Carga de modelos `.flat` | ✅ Nativo (`load_genomic_auto`, `NativeLoader`) | `src/io/loader.rs` |
| Sampling | ✅ Nativo (`sample_top_p`, `apply_repetition_penalty`) | `src/compute/math.rs` |
| RAG semántico | ✅ Nativo (`NativeSemanticRAG`) | `src/compute/rag.rs` |
| Lectura de metadatos `.flat` | ✅ Nativo (`ModelConfig`, `FlatArchive::load_config`) | `src/io/loader.rs` |
| Servidor HTTP/S + SSE | ❌ Pendiente de añadir | — |

**Tareas:**

| # | Tarea |
|---|-------|
| 5.1 | Añadir binario `gaje-serve` (nuevo `[[bin]]` en `Cargo.toml`) con servidor HTTP — recomendado `axum` + `tokio` (SSE nativo), o `std::net::TcpListener` (cero deps). |
| 5.2 | Traducir `server.py` + `model_manager.py` + `prompt_templates.py` (~250-300 líneas) a Rust: `fn format_prompt`, descubrimiento de modelos con `walk`, caché de un modelo en RAM con `OnceCell`/`Mutex<HashMap>`, `get_runtime_info`. |
| 5.3 | **Streaming nativo:** expone `GenomicLLM::forward_core` y `GajeTokenizer` al binario; implementa el bucle de generación token a token usando `sample_top_p` + `apply_repetition_penalty` (ambos ya son funciones Rust puras, **sin overhead FFI**). |
| 5.4 | Mantener `/api/models`, `/api/info`, `/api/load_model`, `/api/chat` y añadir `/api/chat/stream` (SSE). Servir los estáticos (`index.html`, `script.js`, `style.css`, `architecture.*`). |
| 5.5 | Build `cargo build --release` → `target/release/gaje-serve.exe` (o binario Linux/Mac). Documentar ejecución. |

**Beneficios vs. server Python:**

| Aspecto | Python actual | Rust (`gaje-serve`) |
|---------|---------------|---------------------|
| Arranque | ~113s import `stabilized` + ~77s `transformers` | Instantáneo |
| Streaming | FFI por token | Nativo, sin overhead |
| Dependencias | `gaje`, `transformers`, `tokenizers`, numpy | Ninguna externa (single binary) |
| Deploy | Requiere runtime Python | `.exe` autocontenido |

**Riesgo / consideración:** reescritura de ~300 líneas y nueva dependencia de build; se recomienda hacerla **después** de estabilizar Fases 1-4 en Python para no bloquear mejoras de bajo riesgo. La Fase 5 es independiente del plan de UI (el mismo frontend sirve para ambos backends).

---

### Fase 6 — Descarga de modelos desde la web y uso local / WebAssembly (estilo Cactus Needle)

**Objetivo:** replicar el flujo de distribución de Cactus Needle: un modelo pequeño descargable desde la web + un artefacto autocontenido que se ejecuta localmente (o en WASM en el navegador), con la página como punto de entrada para descarga y prueba.

**Modelo de referencia (cactuscompute.com/needle):**
- Modelo abierto pequeño (14 MB, 45M params) con sandbox interactivo que corre **en WebAssembly dentro del navegador**.
- Enlace de **descarga** (HuggingFace) para usarlo localmente.
- Un **artefacto único, sin dependencias** (C++ binario que incluye modelo, tokenizer y grammar; corre de Cortex-M a x86 a WASM).

#### 6.1 Catálogo y descarga de modelos desde la web

| # | Tarea | Archivos |
|---|-------|----------|
| 6.1 | Ampliar `/api/models` para incluir metadatos (tamaño, params, descripción) y un endpoint `/api/models/{name}/download` que sirva el `.gaje`/`.flat` (o redirija a una URL de HuggingFace/CDN). | `server.py`, `model_manager.py` |
| 6.2 | Añadir una pestaña/panel **"Modelos"** en el portal con: lista del catálogo, tamaño, botón **Descargar** y **Usar localmente** (instrucciones de despliegue). | `index.html`, `script.js`, `style.css` |
| 6.3 | Validar checksums (SHA-256) de los modelos descargados y exponer versión/licencia. | `server.py`, `model_manager.py` |

#### 6.2 Artefacto autocontenido para uso local

| # | Tarea |
|---|-------|
| 6.4 | Empaquetar modelo + tokenizer + config en **un único archivo** de distribución. El formato `.gaje.flat` ya es un binario autocontenido (pesos + metadatos + tokenizer embebido) — aprovecharlo como "single artifact". |
| 6.5 | Con la Fase 5, publicar el binario `gaje-serve` autocontenido: el usuario descarga modelo + ejecutable y corre el portal sin instalar Python. |
| 6.6 | Documentar el flujo "descargar → ejecutar localmente" (paso a paso por SO). |

#### 6.3 WebAssembly en el navegador (sandbox, a largo plazo)

| # | Tarea |
|---|-------|
| 6.7 | Evaluar compilar el motor a `wasm32-unknown-unknown` (target WASM) para un sandbox que corra la inferencia **en el navegador**, como el sandbox de Needle. |
| 6.8 | Servir el `.wasm` desde el portal y añadir un modo "demo en navegador" para modelos muy pequeños. |
| 6.9 | Nota: requiere revisar que el kernel (kernels.rs) sea compatible con WASM (sin `rayon`/threads, o con fallback single-thread), y validar el footprint de RAM. |

**Dependencias:** 6.4-6.6 dependen de la Fase 5 (single-binary). 6.7-6.9 son exploración a largo plazo, desacoplada del resto.

---

## 4. Viabilidad del streaming real (verificado)

### 4.1 Por qué `generate_native_py` no sirve para streaming

`src/nn/llm.rs:588` genera todo el bloque de una vez y devuelve `Vec<usize>` completo. No hay callback de token ni yield intermedio. **Pero no es necesario usarla.**

### 4.2 Piezas nativas ya expuestas a Python (habilitan el streaming)

| Función | Firma Python | Ubicación Rust |
|---------|--------------|----------------|
| `rust_llm.forward(token_id, clear_cache)` → logits `Vec<f32>` | `forward` | `src/nn/llm.rs:553` |
| `rust_llm.forward_with_hidden(token_id, clear_cache)` → `(hidden, logits)` | `forward_with_hidden` | `src/nn/llm.rs:558` |
| `rust_llm.clear_cache_py()` | `clear_cache_py` | `src/nn/llm.rs:582` |
| `dna_semantic_compression.sample_top_p(logits, temperature, top_p)` → token | `sample_top_p` | `src/compute/math.rs:866` (registrada en `lib.rs`) |
| `dna_semantic_compression.apply_repetition_penalty(logits, penalty, recent)` | `apply_repetition_penalty` | `src/compute/math.rs:1146` |

### 4.3 El bucle de streaming ya existe en Python

`GenomicLLM.generate()` en `python/gaje/nn/stabilized.py:1043-1080` ya genera token a token:

```python
for _ in range(max_new_tokens):
    next_id = dna_semantic_compression.sample_top_p(logits, top_p, temperature)
    yield tokenizer.decode([next_id])       # yield por token
    logits = self.rust_llm.forward(next_id, False)  # incremento del KV-cache
```

### 4.4 Detalle de muestreo (dos caminos)

| Camino | Muestreo | Uso |
|--------|----------|-----|
| `generate_native_py` (actual en `server.py`) | multinomial puro sobre softmax | bloque completo, sin streaming |
| Loop incremental (`stabilized.py:generate`) | top-p (nucleus) + rep-penalty | token a token, streaming |

**Consecuencia:** el streaming incremental no reproducirá bit a bit los tokens de `generate_native_py`, pero es **equivalente en calidad** e incluso más robusto (top-p es el muestreo recomendado). Si se quisiera salida idéntica, habría que añadir en Rust un modo "multinomial puro" en `sample_top_p_core` — no necesario para streaming.

### 4.5 Implementación del endpoint SSE

```python
class GajeHandler(...):
    def _handle_chat_stream(self):
        llm = get_model(MODELS_ROOT, model_name, GenomicLLM)
        gen = llm.generate(formatted_prompt, max_new_tokens=MAX_TOKENS,
                           temperature=0.2, top_p=0.9, repetition_penalty=1.1)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for token in gen:                       # yield real por token
            self.wfile.write(f"data: {token}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
```

**Ventaja clave:** reutiliza `llm.generate()` (generador de `stabilized.py`), que ya gestiona KV-cache incremental y top-p/rep-penalty de forma nativa. No es necesario escribir el bucle a mano en `server.py`.

**Frontend:** consumir con `fetch` + `ReadableStream` (o `EventSource`); `addMessage` renderiza un mensaje "bot" parcial que se actualiza con cada `data:` recibido, con botón **Detener**.

---

## 5. Orden de ejecución recomendado

1. **Fase 2.1–2.5** (config, logging, validaciones) + **Fase 3.1** (escape XSS) — mejoras de robustez rápidas y de bajo riesgo.
2. **Fase 1** (integración del diagrama) — mayor valor visible.
3. **Fase 2.2–2.3** (streaming + historial) — mayor esfuerzo, alto impacto en UX.
4. **Fase 4** (tests).
5. **Fase 6.1** (catálogo y descarga de modelos desde la web) — se puede hacer sobre el backend Python actual.
6. **Fase 5** (migración del servidor a Rust) — habilita el single-binary; hacerla tras estabilizar 1-4.
7. **Fase 6.2** (artefacto autocontenido + documentación de uso local) — depende de la Fase 5.
8. **Fase 6.3** (WebAssembly en el navegador) — exploración a largo plazo, desacoplada.

**Nota de desacoplamiento:** el frontend es agnóstico al backend (habla por HTTP/SSE). Por eso Fases 1-4 y 6.1 funcionan igual sobre `server.py` (Python) o sobre `gaje-serve` (Rust). Migrar a Rust no rompe el frontend; solo cambia quién sirve las rutas.

---

## 6. Riesgos y consideraciones

| Riesgo | Impacto | Mitigación |
|--------|---------|------------|
| `llm.generate()` es un generador de Python; por tokens puede ser ligeramente más lento que `generate_native_py` (overhead FFI por token) | Medio | Es la única vía de streaming sin tocar Rust; el overhead es aceptable para UX. Alternativa futura: añadir callback de token en Rust. |
| Tokens de streaming ≠ tokens de `generate_native_py` (top-p vs multinomial) | Bajo | Equivalente en calidad; documentar que el stream usa top-p. |
| Modelos muy grandes tardan en cargarse antes de poder generar | Medio | `preloadModel` mantiene feedback; el streaming solo aplica tras la carga. |
| Regresión visual al integrar el diagrama | Bajo | Validación con Playwright (layout, responsive). |
| Migración a Rust (Fase 5) añade dependencia de build y ~300 líneas de reescritura | Medio | Hacerla tras estabilizar 1-4; el frontend es agnóstico al backend. |
| Servir modelos para descarga puede exponer archivos sensibles (Fase 6.1) | Alto | Validar rutas (bloquear path traversal), solo servir dentro de `MODELS_ROOT`, checksums SHA-256. |
| Target WASM del motor puede ser incompatible con kernels actuales (`rayon`, threads) | Alto | Validar `kernels.rs` para `wasm32`; usar fallback single-thread; dejar como exploración a largo plazo. |
| Descargas grandes desde el portal saturan ancho de banda | Bajo | Apuntar a CDN/HuggingFace en vez de servir localmente; mostrar progreso de descarga. |

---

## 7. Anexo — Archivos afectados

| Archivo | Cambios |
|---------|---------|
| `examples/ui/web_ui/server.py` | Env config, logging, endpoint SSE `/api/chat/stream`, historial, validaciones, timeouts, endpoints de catálogo/descarga de modelos |
| `examples/ui/web_ui/model_manager.py` | Validación de `model_name` (path traversal), logging, metadatos de catálogo, checksums |
| `examples/ui/web_ui/prompt_templates.py` | Sin cambios funcionales (o ampliar plantillas) |
| `examples/ui/web_ui/index.html` | Pestañas Chat/Arquitectura/Modelos, sección `#arch-view`, accesibilidad |
| `examples/ui/web_ui/script.js` | Streaming SSE, escape HTML, historial localStorage, estados de generación, lógica de descarga |
| `examples/ui/web_ui/style.css` | Estilos de pestañas, streaming, vista de arquitectura, tema en diagrama, panel de modelos |
| `examples/ui/web_ui/architecture_view.js` | **Nuevo** — render embebible del diagrama (basado en `architecture.html`) |
| `examples/ui/web_ui/architecture.html` | Adaptar a vista embebible (o mantener como standalone) |
| `examples/ui/web_ui/architecture_graph.json` | Fuente única de verdad (sin cambios) |
| `tests/` (nuevos test files) | Unit tests del backend |
| `tests/ui_e2e/web_ui.test.js` | Ampliar con streaming + diagrama |
| `Cargo.toml` | **Fase 5** — añadir binario `gaje-serve` y deps `axum`/`tokio` (o usar TCP manual) |
| `src/bin/gaje-serve.rs` | **Nuevo (Fase 5)** — servidor HTTP Rust (rutas, SSE, estáticos, caché de modelo) |
| `src/nn/llm.rs` | **Fase 5** — exponer `forward_core`/sampling al binario si es necesario |
| `src/core/tokenizer.rs` | **Fase 5** — asegurar exposición de `GajeTokenizer` a código no-PyO3 |
| `target/wasm32-...` + `*.wasm` | **Fase 6.3** — build WASM del motor para sandbox en navegador |

---

## 8. Progreso de ejecución

| Fecha | Hito | Estado |
|-------|------|--------|
| 2026-08-13 | Fase 2.1–2.5 + 3.1 (robustez backend, XSS, SSE) | ✅ Commit `8bda756` |
| 2026-08-13 | Fase 1 (diagrama de arquitectura embebible) | ✅ Commit `8bda756` |
| 2026-08-13 | Fase 2.2–2.3 (streaming SSE, botón Detener, historial localStorage) | ✅ Commit `d29e1c8` |
| 2026-08-13 | Fase 4 parcial — test E2E de streaming + historial (mock) | ✅ Commit `f3b42e9` |
| 2026-08-13 | Fase 3.4 (estados de generación), 3.5 (accesibilidad), 3.6 (barra de progreso de carga) | ✅ Pendiente de commit |
| — | Fase 4 completo (unit tests backend, E2E contra servidor real, CI) | ⏳ Pendiente |
| — | Fase 5 (migración servidor a Rust single-binary) | ⏳ Pendiente |
| — | Fase 6 (catálogo/descarga de modelos, uso local, WASM) | ⏳ Pendiente |

**Nota rendimiento (verificado 2026-08-13):** la inferencia nativa genera ~0.1 tok/s en este hardware; el streaming por SSE y `generate_native_py` en bloque son igual de lentos (ambos ~103 s para 10 tokens). El cuello de botella es la velocidad del modelo, no el overhead FFI del streaming. Acelerar requiere optimizar la inferencia (Fase 5 / mejores kernels).
