# 🧬 Plan de Implementación: `gaje-cli serve` (Servidor HTTP Nativo en Rust)

**Fecha de creación:** 2026-08-27  
**Estado:** Aprobado / En Planificación  
**Versión de consolidación:** `1.7.0-alpha`  
**Enfoque:** Opción B — Mínimas Dependencias (`tiny_http` + multihilo síncrono nativo)  
**Objetivo principal:** Eliminar la dependencia del runtime de Python (`server.py`) y proporcionar un binario único ejecutable autónomo y portable (`gaje-cli serve`) con inferencia streaming SSE nativa y servicio de la Web UI.

---

## 1. Contexto y Justificación Técnica

Actualmente, el portal interactivo y la API de inferencia local de GAJE dependen de `examples/ui/web_ui/server.py`. Aunque funcional, este enfoque presenta importantes limitaciones operativas:

1. **Dependencia de Runtime Python:** Requiere Python 3.12, PyTorch/Transformers, numpy y compilación de extensiones C/FFI (`_impl.pyd`/`.so`), lo que genera un entorno de más de 500 MB y problemas de permisos en librerías dinámicas del sistema (ej. `shm.dll` en Windows).
2. **Latencia de Arranque (Cold Start):** El servidor en Python tarda entre 5 y 15 segundos en importar librerías antes de abrir el puerto de red.
3. **Overhead FFI en Streaming:** El streaming de tokens cruza repetidamente la barrera FFI (Rust ↔ PyO3 ↔ Python) por cada token generado.
4. **Soberanía Nativa (Regla de Oro en `AGENTS.md`):** Establece que todas las herramientas administrativas y de alto rendimiento deben residir en el núcleo nativo de Rust (`gaje-cli`).

### Objetivos del Plan:
* Crear el subcomando `gaje-cli serve` en Rust.
* Emplear una arquitectura de **mínimas dependencias** sin runtimes asíncronos pesados (sin Tokio/Hyper).
* Ofrecer streaming SSE nativo token-a-token a velocidad máxima de CPU/SIMD.
* Servir todos los recursos estáticos de la Web UI (`index.html`, CSS, JS, SVG, WASM) directamente desde el disco o embebidos.
* Distribuir la plataforma como **un único binario ejecutable (`gaje-cli.exe` / `gaje-cli`) de ~18 MB**.

---

## 2. Arquitectura de Mínimas Dependencias (Opción B)

```
                       ┌────────────────────────────────────────┐
                       │               Navegador Web            │
                       │    (Chat Sandbox, Docs, Architecture)   │
                       └───────────────────┬────────────────────┘
                                           │ HTTP / SSE
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                `gaje-cli serve` (Rust)                                │
│                                                                                        │
│  ┌─────────────────────────┐  ┌────────────────────────┐  ┌─────────────────────────┐  │
│  │   Router HTTP Nativo    │  │  Servidor de Estáticos │  │   Endpoints API REST    │  │
│  │       (tiny_http)       │  │  (HTML, CSS, JS, SVG)  │  │ (/api/models, /api/info)│  │
│  └────────────┬────────────┘  └────────────────────────┘  └────────────┬────────────┘  │
│               │                                                        │               │
│               └──────────────────────────┬─────────────────────────────┘               │
│                                          ▼                                             │
│                       ┌──────────────────────────────────────┐                         │
│                       │   Loop de Inferencia SSE Streaming   │                         │
│                       │     (/api/chat/stream - Token/s)     │                         │
│                       └──────────────────┬───────────────────┘                         │
│                                          ▼                                             │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │                              Núcleo Matemático Nativo                            │  │
│  │  • GenomicLLM (src/nn/llm.rs)              • GtokTokenizer (src/core/gtok.rs)    │  │
│  │  • Loader Flat Zero-Copy (src/io/loader.rs)• Math Sampling (src/compute/math.rs) │  │
│  │  • Memoria Islas .gmem (src/compute/island)• AVX2/NEON SIMD (src/compute/simd)   │  │
│  └──────────────────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### ¿Por qué `tiny_http`?
* **Cero Asincronía Compleja:** Funciona con hilos estándar de Rust (`std::thread::spawn` o `rayon`), sin requerir el runtime `tokio` ni dependencias asociadas.
* **Tamaño Mínimo:** Añade menos de 150 KB al binario final.
* **Compatibilidad Total:** Multiplataforma inmediata (Windows, Linux, macOS) con sockets `std::net`.

---

## 3. Especificación de Endpoints y Comportamiento

El servidor nativo implementará exactamente las mismas rutas que la Web UI ya consume:

| Ruta | Método | Descripción | Formato de Respuesta |
| :--- | :---: | :--- | :--- |
| `/api/models` | `GET` | Lista los modelos planos (`.flat`/`.gaje`) encontrados en el directorio configurado, con metadatos de arquitectura (`n_embd`, `n_layer`, `vocab_size`, tamaño en disco). | `application/json` |
| `/api/info` | `GET` | Retorna versión del motor (`1.7.0-alpha`), arquitectura del hardware (AVX2/NEON/GPU), memoria RAM usada y estado del modelo activo. | `application/json` |
| `/api/load_model` | `POST` | Carga o intercambia el modelo activo en RAM con mapeo zero-copy mmap. | `application/json` |
| `/api/chat` | `POST` | Inferencia completa síncrona (fallback sin streaming). | `application/json` |
| `/api/chat/stream` | `POST` / `GET` | **Streaming SSE nativo token-a-token** (`text/event-stream`). Emite cada token inmediatamente al generarse (`data: {"token": "..."}\n\n`) y finaliza con `data: [DONE]\n\n`. | `text/event-stream` |
| `/*` | `GET` | Servidor de archivos estáticos: entrega `index.html`, `architecture.html`, `docs.html`, hojas de estilo CSS, scripts JS, sprite SVG y binarios WASM. | Según extensión MIME |

---

## 4. Estructura de Archivos del Módulo Servidor

Se creará el módulo `src/server/` dentro del crate `gaje-core`:

```
src/
├── bin/
│   └── gaje-cli.rs               <-- Integración del subcomando `serve`
└── server/
    ├── mod.rs                    <-- Orquestador del servidor HTTP y loop principal
    ├── routes.rs                 <-- Enrutador de peticiones (REST vs Estáticos)
    ├── static_files.rs           <-- Lector y despachador de archivos Web UI
    ├── api.rs                    <-- Handlers de /api/models, /api/info, /api/load_model
    └── streaming.rs              <-- Bucle generativo SSE token-a-token directo al socket
```

---

## 5. Fases de Ejecución Paso a Paso

### **Fase 1: Configuración de Dependencias**
* Actualizar `Cargo.toml` para incluir `tiny_http = "0.12"` y `serde_json = "1.0"`.

### **Fase 2: Interfaz de Línea de Comandos (`gaje-cli`)**
* Añadir el comando `Serve` a la estructura `Commands` en [`src/bin/gaje-cli.rs`](file:///E:/Desarrollos/develop/gaje-semantic-compression/src/bin/gaje-cli.rs):
  * `--port <PORT>`: Puerto de escucha (por defecto `8080`).
  * `--host <HOST>`: Interfaz de red (por defecto `127.0.0.1`).
  * `--models-dir <PATH>`: Ruta hacia la carpeta de modelos `.flat` (por defecto `./models`).
  * `--static-dir <PATH>`: Ruta hacia la carpeta de la Web UI (por defecto `./examples/ui/web_ui`).
  * `--model <NAME>`: Modelo inicial a precargar en RAM.

### **Fase 3: Servicio de Archivos Estáticos y Seguridad**
* Implementar `static_files.rs` con:
  * Resolución segura de rutas (bloqueo estricto de *path traversal* contra `..` o accesos fuera de `static_dir`).
  * Mapa determinista de tipos MIME sin dependencias externas (`.html` ➔ `text/html`, `.css` ➔ `text/css`, `.js` ➔ `application/javascript`, `.svg` ➔ `image/svg+xml`, `.wasm` ➔ `application/wasm`, `.json` ➔ `application/json`).
  * Entrega de `index.html` en la raíz `/`.

### **Fase 4: Endpoints REST de Administración**
* Implementar `api.rs`:
  * Descubrimiento dinámico de modelos planos con lectura de cabecera `ModelConfig` en `models_dir`.
  * Endpoint `/api/info` con telemetría de hardware nativa.
  * Estructura thread-safe `Arc<RwLock<Option<LoadedModel>>>` para compartir el modelo en memoria entre hilos.

### **Fase 5: Motor de Generación y Streaming SSE Nativo**
* Implementar `streaming.rs`:
  * Parseo de JSON de entrada (`prompt`, `temperature`, `top_p`, `repetition_penalty`, `max_tokens`).
  * Formateo de plantillas (ChatML / Gemma / Alpaca).
  * Prefill de KV-cache y bucle token a token:
    * Llamada a `GenomicLLM::forward_step`.
    * Aplicación de `apply_repetition_penalty` y `sample_top_p`.
    * Decodificación con `GtokNativeTokenizer`.
    * Escritura inmediata (`write_all` + `flush`) del fragmento SSE `data: {"token": "..."}\n\n` en el socket TCP.

### **Fase 6: Pruebas y Certificación**
* Ejecución y verificación con la suite E2E de Playwright (`npm run test:ui`) apuntando a `gaje-cli serve`.
* Medición de latencia de primer token (TTFT) y throughput sostenido (tokens/segundo) comparado contra Python.

---

## 6. Criterios de Aceptación y Certificación

1. **Binario Único:** `cargo build --release` genera un ejecutable autocontenido `target/release/gaje-cli.exe` (o binario Unix).
2. **Arranque Instantáneo:** El servidor responde a peticiones en menos de 50 milisegundos tras la ejecución.
3. **Cero Python Runtime:** La Web UI y la inferencia funcionan de forma completa en un sistema sin Python instalado.
4. **Paridad Total de la UI:** La Web UI modularizada (3 temas, telemetría HUD, selector de modelos, bitácora `.md`, pantalla completa) opera con 100% de compatibilidad.
5. **Streaming Fluido:** Los tokens fluyen de forma continua y suave en el navegador a través de SSE sin bloqueos de búfer.
