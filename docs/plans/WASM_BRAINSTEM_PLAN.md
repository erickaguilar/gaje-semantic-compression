# Plan: GAJE-WASM — El Motor como Tronco Encefálico (Build WASM + API Sensorio-Motora)

> Rama: `test/experimental` · Estado: **PROPUESTO** · Fecha: 2026-08-20
> Complementa a `docs/plans/MASTER_ROADMAP_2026.md`, `docs/plans/NATIVE_SEMANTIC_RAG_PLAN.md`
> y al mandato de `docs/meta/EMPIRICAL_TRUTH_STATE.md`.
> **Tesis**: el motor GAJE puede compilarse a WebAssembly sin alterar su semántica y,
> expuesto tras una API sensorio-motora estable, actuar como *tronco encefálico*:
> la capa involuntaria que conecta al modelo (córtex) con el exterior (periferia).

---

## 1. Contexto y motivación

### 1.1 Origen

Dos preguntas impulsan este plan:

1. **¿Es posible un WASM con el modelo GAJE?** — Análisis de compilabilidad del motor
   Rust a `wasm32-unknown-unknown` cargando modelos `.gaje.flat`.
2. **¿Puede usarse como tronco encefálico para conectar el modelo con el exterior?** —
   Arquitectura donde GAJE gestiona las funciones vitales involuntarias de inferencia
   (aferencia, eferencia, autonomía, memoria) y es el único conducto entre el LLM y el mundo.

### 1.2 Mapeo anatómico: metáfora → componentes existentes

La metáfora no es decorativa: cada función del tronco encefálico ya tiene correspondencia
estructural en el código actual.

| Función del tronco | Componente GAJE | Estado |
|:---|:---|:---:|
| Vías aferentes (sentidos → cerebro) | `DNIEngine` (ingestión neuronal directa), RAG (`src/compute/island.rs`, `rag.rs`) | ✅ Operativo |
| Vías eferentes (decisión → acto) | Muestreo Lagrangiano + generación autoregresiva (`src/compute/lagrangian.rs`) | ✅ Operativo |
| Funciones autonómicas (ritmo, respiración) | KV-cache comprimido 2-bit, presupuesto de contexto 128 tokens | ✅ Operativo |
| Memoria de hábito (procedimental) | Island Model `.gmem` persistente (cold start 0.12 ms) | ✅ Operativo |
| Nervios craneales (interfaz periférica) | FFI C (`src/ffi.rs`), PyO3 (`python/gaje/`) | ✅ Operativo |
| Reflejos condicionados (API sensorio-motora) | **No existe** | ❌ Este plan |

El modelo cuantizado es córtex puro: cómputo pasivo. GAJE respira por él — tokeniza,
recupera memoria, decide cuándo inyectar contexto y emite tokens sin que el modelo sepa
nada del exterior. Lo que falta es el equivalente de los reflejos: una **API sensorio-motora
estable** hacia afuera (§4.4).

### 1.3 Valor estratégico del WASM

Un tronco encefálico compilado a wasm-bindgen corre idéntico en navegador, móvil y edge,
con los mismos "nervios". Convierte al embudo obligatorio de señales del modelo en un
artefacto independiente del hardware y distribuible sin instalación (un URL).

---

## 2. Objetivo e hipótesis

**Objetivo**: compilar el núcleo GAJE a WebAssembly, cargar modelos `.gaje.flat` desde el
navegador y exponer una API sensorio-motora (aferencia/eferencia/autonomía) sobre ese build.

**Hipótesis**:

> **H1 (compilabilidad)** — El núcleo (kernels + nn + io `.flat`) compila a
> `wasm32-unknown-unknown` usando los fallbacks escalares ya existentes, sin cambios
> semánticos. Los kernels SIMD están correctamente aislados tras
> `#[cfg(target_arch = "...")]` con ruta escalar para cualquier otro target
> (verificado en `src/compute/kernels/dot.rs:90`, `norm.rs`, `genomic.rs`, `lut.rs`).
>
> **H2 (usabilidad)** — SmolLM2 135M `.flat` (~200–250 MB en memoria WASM) genera
> en navegador a velocidad útil (≥ 3 tok/s escalar single-thread; objetivo ≥ 2× con SIMD128).
>
> **H3 (paridad)** — Con decodificación greedy y misma entrada, el build WASM produce
> exactamente los mismos token IDs que el build nativo (determinismo bit a bit).

**Hipótesis nula (lo que descartaría el plan)**:

> Si el fallback escalar rinde < 1 tok/s en SmolLM2 135M incluso con SIMD128, o el
> consumo de memoria excede los límites prácticos del navegador (~2 GB), el frente WASM
> se congela y se documenta en `EMPIRICAL_TRUTH_STATE.md` (patrón Q2_0).

---

## 3. Auditoría técnica de compilación (estado actual verificado)

### 3.1 Lo que ya está listo

| Elemento | Evidencia | Impacto WASM |
|:---|:---|:---|
| Kernels SIMD aislados por arquitectura | `#[cfg(target_arch = "x86_64"/"aarch64")]` + fallback escalar (`not(any(...))`) en `dot.rs`, `norm.rs`, `genomic.rs`, `lut.rs` | Compila hoy en wasm32 (escalar) |
| PyO3 opcional | `Cargo.toml`: `python = ["pyo3", ...]` (feature, no default) | Sin arrastre de Python |
| Formato `.flat` plano y autodescriptivo | `FlatHeaderV2` de 4096 B + directorio JSON + pesos alineados | Parseable desde un `ArrayBuffer`; mmap es solo la vía rápida, no un requisito |
| `.gmem` binario plano alineado a 64 B | Island Model | Persistible tal cual en OPFS/IndexedDB |
| Superficie FFI C de referencia | `src/ffi.rs`: `gaje_session_load/chat/free` | Patrón a replicar en wasm-bindgen |
| `half`, `lz4_flex`, `serde_json`, `bincode` | Rust puro | Compatibles con wasm |

### 3.2 Bloqueos y solución propuesta

| Dependencia | Uso actual | Problema en wasm32-unknown-unknown | Solución |
|:---|:---|:---|:---|
| `memmap2` | `src/io/flat_reader.rs:25`, `src/io/gguf/reader.rs` | No existe mmap en navegador | Trait `TensorSource` con dos impls: `MmapSource` (nativo) / `MemSource` (buffer en memoria). Ver §4.2 |
| `redb` | `src/core/db.rs`, `src/io/db_loader/*`, `src/io/smg1.rs` | File locks + fs; no compila/opera en browser | Feature-gate `native-db`; en WASM solo ruta `.flat` |
| `libc` | `src/compute/power.rs` (afinidad CPU) | No disponible | Gate por `cfg`; no-op en wasm |
| `ctrlc` | `src/bin/gaje-cli.rs` | Señales POSIX inexistentes | Mover al feature `cli` |
| `indicatif` | Progreso en CLI | Terminal inexistente en browser | Mover al feature `cli` |
| `rayon` | Paralelismo masivo | `std::thread::spawn` falla en wasm | Fase inicial: `cfg` → iteración secuencial. Fase posterior: `wasm-bindgen-rayon` (exige COOP/COEP) |
| `tokenizers` (HF) | Tokenización | Backend regex dudoso en wasm | Evaluarse en Fase 0; opciones: embeber en Rust (§4.6, soberanía) o tokenizar en JS |
| `rand` | Muestreo | Requiere `getrandom` con feature `js` en wasm | Activar feature en target wasm |

---

## 4. Diseño

### 4.1 Features en `Cargo.toml`

```toml
[features]
default = ["cli", "native-db"]        # comportamiento actual intacto
cli = ["ctrlc", "indicatif"]
native-db = ["redb"]
wasm = []                              # excluye cli/native-db; fuerza rutas en memoria
```

Principio rector: **el árbol nativo por defecto no cambia ni un bit**; el build WASM es
una proyección restringida del mismo crate.

### 4.2 Abstracción `TensorSource`

Hoy `GajeFlatFileReader` acopla lectura y mmap (`flat_reader.rs:24`). Se extrae:

```rust
pub trait TensorSource: Send + Sync {
    fn header_bytes(&self) -> &[u8];       // primeros 4096 B
    fn slice(&self, off: usize, len: usize) -> &[u8];  // relativo a weights_offset
    fn len(&self) -> usize;
}
// impls: MmapSource (nativo, cero costo adicional) | MemSource (Vec<u8>/Arc<[u8]>, WASM)
```

`get_slice`/`get_f32_slice` pasan a delegar en el trait. El formato no cambia: el
`.gaje.flat` v2 ya es plano y autodescriptivo, ideal para parseo zero-copy desde
memoria lineal WASM (respetando alineación de 64 B).

### 4.3 Crate `gaje-wasm` (wasm-bindgen)

Nuevo crate miembro que replica el patrón de `ffi.rs` pero para JS:

```ts
// Superficie mínima viable (Fase 2)
const engine = await GajeWasm.load(flatBytes: Uint8Array, opts?)   // header + dir + pesos
engine.generate(tokenIds: number[], maxTokens, temperature, topP): number[]
engine.chat(text: string, maxTokens, temperature, topP): string    // si hay tokenizador embebido
engine.free()
```

Convenciones: ownership explícito (`free`), errores como excepciones JS con mensaje del
`io::Error`, sin promesas dentro del cálculo (bloqueo controlado en un Worker).

### 4.4 API Sensorio-Motora — el tronco completo (Fase 5)

Sobre el build WASM (y exportable también a nativo vía FFI):

1. **Aferencia estandarizada**: adaptadores de entrada (HTTP, WebSocket, eventos DOM,
   sensores móviles) que desembocan todos en `ingest()` del DNI/RAG. El origen da igual;
   el conducto es uno.
2. **Eferencia con actuadores**: bus de acciones (tool calls) que el muestreo puede emitir
   además de texto. Gramática de salida restringida post-sampling.
3. **Ciclo autonómico**: event loop propio sobre el scheduler timing-wheel O(1) existente
   (`src/compute/`), encargado de la consolidación de memoria a `.gmem` en background
   (análogo del sueño), con presupuesto de cómputo acotado para no robar latencia al decode.

### 4.5 Entrega del modelo en navegador

- `fetch` con streaming + caché en **OPFS** o Cache API (re-descarga solo ante cambio de versión).
- Verificación de integridad (hash del header o checksum externo) antes de instanciar.
- Presupuesto de memoria: asignar `WebAssembly.Memory` única y copiar el buffer una vez;
  `MemSource` opera sobre esa región sin duplicados.

### 4.6 Camino de soberanía máxima: WASI + tokenizador embebido

El desglose de soberanía tras este plan deja el runtime en Rust puro salvo dos fronteras:
el shim JS que wasm-bindgen genera (irreductible *en navegador*: es la membrana del nervio,
solo puede hacerse delgada) y la tokenización si se delega a JS. Este camino elimina la segunda
y ofrece una variante sin JS alguno:

1. **Tokenizador embebido (Rust puro, browser)** — compilar el tokenizador dentro del `.wasm`
   (crate `tokenizers` con `default-features = false`, o BPE mínimo propio en Rust) y exponer
   `encode/decode` vía wasm-bindgen. JS queda reducido a cableado del Worker y del canvas/DOM.
   Coste: +MBs de binario y evaluación del backend regex en wasm (Fase 0 decide).
2. **Target WASI (soberanía total, server/edge)** — compilar a `wasm32-wasi` y ejecutar sobre
   WasmTime/WasmEdge o runtimes edge (Cloudflare Workers, Fastly): **cero líneas de JS**,
   runtime 100% Rust end-to-end. El tronco corre donde el cómputo es barato; el navegador
   queda como una periferia más consumiendo la misma API.

**Gate de adopción** (Fase 2): si el tokenizador embebido añade < 30% al tamaño del binario
y no introduce regresión de carga, se adopta por defecto; si no, JS-tokenizer queda como
fallback documentado. La variante WASI se valida tras Fase 3 con el mismo harness de paridad.

---

## 5. Fases con umbrales de decisión

Cada fase termina en veredicto empírico registrado (éxito o fracaso documentado, patrón
del Mandato de Verdad Empírica).

### Fase 0 — Spike de compilación (horas)
`cargo check --target wasm32-unknown-unknown` sobre el árbol actual. Inventario exacto de
errores por dependencia; validar/refutar la tabla §3.2. Decisión: proceder si los bloqueos
coinciden con lo previsto (ningún bloqueo dentro de `compute`/`nn`).

### Fase 1 — Feature-gating + `TensorSource` (1–2 días)
Refactor §4.1–4.2. **Gate**: suite nativa verde (`cargo test`, pytest unit/integration/metrics)
y benchmarks nativos sin regresión (±2% tok/s). Si el refactor toca semántica, se revierte.

### Fase 2 — `gaje-wasm` + demo SmolLM2 135M (2–4 días)
Demo JS en Worker: carga `.flat` desde OPFS, generación greedy.
**Gates**: carga completa < 10 s en desktop; ≥ 3 tok/s escalar; RSS WASM < 400 MB;
**paridad H3**: mismos token IDs que nativo en 20 prompts fijos greedy. Fracaso en paridad
= bug de layout/alineación: bloqueante, no negociable.

### Fase 3 — Kernel SIMD128 (2–3 días)
Puerto AVX2→wasm32 de `dot_product` y `genomic_dot_product_q4_0` (patrón ya existente:
función `unsafe` + `#[target_feature]` + detección en runtime).
**Gate**: ≥ 2× vs escalar en micro-benchmark y ≥ 1.5× E2E; outputs idénticos al escalar.

### Fase 4 — Hilos (opcional)
`wasm-bindgen-rayon` + COOP/COEP en el servidor demo.
**Gate**: escalado ≥ 1.6× con 4 hilos en decode. Si el hosting no puede servir COOP/COEP,
se documenta y se queda en single-thread.

### Fase 5 — API Sensorio-Motora (1–2 semanas)
Aferencia unificada (`ingest`), bus de acciones, ciclo autonómico con consolidación `.gmem`
en OPFS. **Gate**: RAG multinicho < 5 ms en navegador (vs 0.75 ms nativo, tolerancia 7×);
consolidación background sin impacto > 5% en throughput de decode.

---

## 6. Riesgos y mitigaciones

| Riesgo | Prob. | Mitigación |
|:---|:---:|:---|
| Límite de memoria lineal 32-bit (~2 GB prácticos; `memory64` aún experimental) | Alta | Foco en 135M (holgado) y 0.5B (desktop-only); medir en Fase 2 antes de prometer soporte |
| Escalar demasiado lento (< 1 tok/s) | Media | Es la hipótesis nula: SIMD128 es la respuesta prevista (Fase 3); si tampoco alcanza, congelar frente |
| `tokenizers` no compila en wasm | Media | Plan B definido: tokenización en JS (el tokenizador HF tiene builds JS oficiales) |
| Degradación por falta de mmap (copia completa a RAM lineal) | Baja | El formato ya es plano; el costo es una sola copia en carga, amortizada |
| Rayon sin hilos degrada batch/parallelismo | Baja | Decode autoregresivo es secuencial por naturaleza; rayon aporta poco al camino crítico |
| Deriva del árbol nativo durante el refactor | Media | Fase 1 con gates de no-regresión; features default intactas |

---

## 7. Métricas de éxito (resumen ejecutable)

| Métrica | Umbral mínimo | Objetivo |
|:---|:---:|:---:|
| Paridad de tokens WASM vs nativo (greedy) | 100% en 20 prompts | 100% |
| Throughput SmolLM2 135M (escalar) | ≥ 3 tok/s | ≥ 5 tok/s |
| Throughput con SIMD128 | ≥ 6 tok/s | ≥ 10 tok/s |
| Carga completa desde OPFS | < 10 s | < 4 s |
| Memoria WASM pico (135M) | < 400 MB | < 300 MB |
| RAG multinicho en navegador | < 5 ms | < 2 ms |
| Regresión nativa (tok/s, suite) | 0 fallos, ±2% | 0 fallos, 0% |

---

## 8. Referencias

- Kernels con fallback escalar: `src/compute/kernels/{dot,norm,genomic,lut}.rs`
- Lector `.flat` a abstraer: `src/io/flat_reader.rs`
- FFI de referencia: `src/ffi.rs` · Puente PyO3: `python/gaje/`
- Formato y cabecera: `src/io/header/{flat,blocks,types}.rs`
- Precendente de veredicto negativo documentado: `docs/research/Q2_0_2BIT_SPATIAL_EXPERIMENT.md`
- Gobernanza: `docs/meta/EMPIRICAL_TRUTH_STATE.md`

---
*Plan GAJE-WASM v1 (Agosto 2026) — El tronco antes que el cerebro: primero el conducto, después la corteza.*
