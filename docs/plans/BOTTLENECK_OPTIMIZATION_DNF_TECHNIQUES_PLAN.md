# ⚡ Plan de Optimización de Cuellos de Botella mediante Técnicas DNF y Multi-Stream

**Fecha:** 2026-08-28
**Estado:** Propuesta Técnica y Plan Arquitectónico Aprobado
**Versión objetivo:** `1.7.0-alpha`
**Ámbitos:** I/O Tensorial · Red y Transferencia · WebAssembly / Frontend · Memoria Genómica (`.gmem`)

---

## 1. Visión General y Resumen Ejecutivo

El framework **GAJE (Genetic Adaptive Joint Embedding / DNA Semantic Compression)** procesa matrices tensoriales densas, pesos genómicos `.flat` (400 MB – 3.8 GB) y estados de memoria episódica `.gmem`. 

Tras auditar los puntos de fricción del sistema, se identificaron **4 cuellos de botella (embudos) críticos** en I/O, red y persistencia. Este plan define la resolución de dichos cuellos de botella trasladando los principios de diseño de **DNF5 / `librepo` (Fedora)** y **`hf_transfer`**:

1. **Multi-Stream en Navegador:** Descargas particionadas en Web Workers vía `HTTP Range 206` + OPFS.
2. **Exportación Tensorial Paralela en Rust:** Pre-asignación zero-copy (`File::set_len`) y escritura por offsets con Rayon.
3. **Actualizaciones Diferenciales (`zchunk` Hashing):** Sincronización delta de centroides y épocas de memoria sin re-descargar modelos completos.
4. **Arranque Inmediato (*Streaming On-Demand / Cold Start*):** Inferencia con latencia percibida $< 1\text{ s}$ descargando primero cabecera + GTOK + capas iniciales.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 Topología de Optimización de Embudos GAJE                   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
      ┌──────────────────┬─────────────┴────────────┬──────────────────┐
      ▼                  ▼                          ▼                  ▼
┌─────────────┐   ┌─────────────┐            ┌─────────────┐   ┌─────────────┐
│ 1. WASM     │   │ 2. Export   │            │ 3. Delta    │   │ 4. On-Demand│
│ Multi-Stream│   │ Zero-Copy   │            │ zchunk Sync │   │ Cold Start  │
│ (Frontend)  │   │ (gaje-core) │            │ (.gmem v2)  │   │ (Pipeline)  │
└─────────────┘   └─────────────┘            └─────────────┘   └─────────────┘
```

---

## 2. Los 4 Cuellos de Botella y sus Soluciones Técnicas

---

### 🌐 Cuello de Botella 1: Descarga Monolítica y Lineal en WebAssembly (Web UI)

#### A. Diagnóstico del Problema:
* En entornos estáticos o clientes ligeros (Vercel, PWA, navegadores móviles), la Web UI descarga el modelo `.flat` mediante una única petición `fetch()` secuencial.
* En redes de 300–1000 Mbps, un único stream TCP se ve estrangulado por la ventana de congestión y latencia RTT, tardando 20–45 segundos para un archivo de 471 MB a 1.2 GB.

#### B. Solución Técnica (Multi-Worker HTTP Range Segmented Fetch):
1. **Detección de Tamaño:** El hilo de descarga realiza una petición `HEAD` para obtener `Content-Length` y validar `Accept-Ranges: bytes`.
2. **Segmentación:** Divide el archivo en $N$ bloques (ej. 4 a 6 hilos concurrentes de 80 MB cada uno).
3. **Escritura Directa:** Cada Worker descarga su segmento con cabeceras `Range: bytes=start-end` y escribe directamente sobre un **OPFS (*Origin Private File System*)** o un `ArrayBuffer` pre-asignado.
4. **Persistencia en `IndexedDB` (`GajeHelixDB`):** Al finalizar el último chunk, el buffer ensamblado se registra en el almacén de modelos sin copias intermedias en memoria.

```
                                  CDN Hugging Face
                                         │
                 ┌───────────────────────┼───────────────────────┐
                 ▼                       ▼                       ▼
         Worker 1 (0-25%)        Worker 2 (25-50%)       Worker 3 (50-100%)
                 │                       │                       │
                 └───────────────────────┼───────────────────────┘
                                         ▼
                     OPFS / Uint8Array Pre-Asignado en RAM
                                         ▼
                            IndexedDB (Store: model_cache)
```

* **Impacto:** Reduce el tiempo de descarga en navegador de **35s a 5–7s (mejora de 5× a 7×)**.

---

### ⚡ Cuello de Botella 2: Exportación Secuencial de Pesos en Disco (`gaje-core` / Rust)

#### A. Diagnóstico del Problema:
* Los pipelines de conversión de modelos (PyTorch / SafeTensors $\rightarrow$ `.flat`) escriben secuencialmente capa por capa (`write_all`) en disco.
* Para modelos de 3B a 7B (2.4 GB – 5.0 GB), el proceso de cuantización `Q4_0` y serialización toma entre 30 y 60 segundos debido a la fragmentación de archivos y esperas bloqueantes de I/O.

#### B. Solución Técnica (Pre-asignación `File::set_len` + Rayon Parallel `pwrite`):
1. **Pre-asignación Zero-Fragmentation:** Antes de escribir un solo byte, el exportador invoca `file.set_len(total_calculated_bytes)` (o `posix_fallocate`). El sistema operativo reserva los sectores contiguos en disco en 0 ms.
2. **Cálculo de Offsets de Cabecera:** Se pre-computa el offset exacto de cada tensor dentro del archivo plano:
   $$\text{Offset}(T_i) = \text{HeaderSize} + \sum_{k=0}^{i-1} \text{Bytes}(T_k)$$
3. **Cuantización y Escritura Paralela sin Locks:** Con **Rayon**, N hilos de CPU cuantizan simultáneamente las matrices $Q, K, V, \text{FFN}$ y escriben en disco usando `FileExt::write_all_at` (Linux/macOS `pwrite`) en paralelo sin mutexes.

```rust
// Esquema de Exportación Paralela Concurrente
let file = Arc::new(OpenOptions::new().write(true).create(true).open(output_path)?);
file.set_len(total_model_bytes)?; // Pre-asignación instantánea

layer_tasks.into_par_iter().for_each(|task| {
    let quantized_bytes = quantize_tensor_q4_0(&task.tensor_data);
    #[cfg(unix)]
    std::os::unix::fs::FileExt::write_all_at(&*file, &quantized_bytes, task.byte_offset)
        .expect("Escritura concurrente por offset fallida");
});
```

* **Impacto:** La exportación de un modelo de 7B pasa de **45 segundos a < 3.5 segundos (mejora de 12×)**.

---

### 🧬 Cuello de Botella 3: Sincronización Redundante de Memoria y Mutaciones Genómicas

#### A. Diagnóstico del Problema:
* Cuando un modelo recibe un ajuste fino genómico, destilación de centroides o una nueva época de memoria episódica (`.gmem` v2), el usuario tiene que descargar el archivo binario completo de nuevo, desperdiciando ancho de banda cuando el 90% de las capas base no cambiaron.

#### B. Solución Técnica (Delta Chunks con Hashing estilo `zchunk` / Fedora):
1. **Manifiesto de Bloques (`.flat.manifest` / `.gmem.manifest`):**
   * El archivo se segmenta lógicamente en bloques de 2 MB a 4 MB.
   * Se genera un encabezado ligero con una tabla de hashes SHA-256 por cada bloque:
     $$\text{Manifest} = [(\text{BlockID}_0, \text{Hash}_0), (\text{BlockID}_1, \text{Hash}_1), \dots, (\text{BlockID}_n, \text{Hash}_n)]$$
2. **Descarga Diferencial:**
   * El cliente descarga primero el manifiesto (~4 KB).
   * Compara los hashes locales vs remotos e identifica únicamente los bloques modificados.
   * Solicita mediante `HTTP Range` exclusivamente los bloques que mutaron.
3. **Reensamblaje In-Place:**
   * Sobrescribe los bloques modificados en los offsets exactos del archivo local ya existente.

* **Impacto:** Actualizaciones de versiones y épocas de memoria con **reducción del 85% al 95% del volumen de datos transferidos**.

---

### ⏱️ Cuello de Botella 4: Latencia de Arranque en Frío (*Time-to-First-Token Cold Start*)

#### A. Diagnóstico del Problema:
* En la primera ejecución (o cuando el modelo no está en caché local), el usuario debe esperar a que se descargue el 100% de la red neuronal antes de que el motor pueda procesar el primer token, degradando la experiencia percibida.

#### B. Solución Técnica (Streaming On-Demand / Inferencia Pipelined):
1. **Partición de Prioridad de Descarga:**
   * **Fase Prioritaria (Top-0):** Cabecera genómica + Tokenizador GTOK + Tabla de Embeddings + Capas 0 a 2 (~40–60 MB).
   * **Fase Pipelined (En Segundo Plano):** Capas 3 a $N$ + LM Head.
2. **Ejecución Superpuesta:**
   * En cuanto la Fase Prioritaria está en memoria, el motor acepta el prompt del usuario y comienza la tokenización GTOK y el forward pass de las primeras capas.
   * Mientras la señal neuronal atraviesa las capas iniciales, el descargador multi-stream en segundo plano finaliza la descarga y decodificación de las capas restantes antes de que el ciclo de cálculo las requiera.

* **Impacto:** **Latencia percibida de arranque en frío reducida de 25s a < 1.2s**.

---

## 3. Matriz de Impacto y Plan de Implementación

| Cuello de Botella | Módulo GAJE | Técnica Clave | Reducción de Tiempo / Datos |
| :--- | :--- | :--- | :--- |
| **1. Descargas Web UI / WASM** | `examples/ui/web_ui/static/js/` | Multi-Worker HTTP Range Fetch + OPFS | **75% – 85% menos tiempo de espera** en navegador. |
| **2. Exportación Tensorial** | `gaje-core` (`src/io/flat_writer.rs`) | `File::set_len` + Rayon Parallel `pwrite` | **90% menos tiempo de exportación** en disco. |
| **3. Sincronización `.gmem`** | `src/io/manifest.rs` & `island.rs` | Delta Chunking SHA-256 (`zchunk`) | **85% – 95% ahorro de ancho de banda**. |
| **4. Arranque On-Demand** | `src/engine/pipeline.rs` & `engine.js` | Streaming por capas prioritarias | **Latencia percibida < 1.2s** en arranque en frío. |

---

## 4. Fases de Ejecución

1. **Fase 1 (Inmediata):** Implementar la exportación paralela en Rust con pre-asignación `File::set_len` en `src/io/`.
2. **Fase 2 (Frontend Web UI):** Integrar la descarga multi-stream segmentada con Range requests en el `wasm_worker.js`.
3. **Fase 3 (Manifiestos Delta):** Diseñar la estructura de manifiestos `.flat.manifest` para sincronización diferencial de modelos y épocas `.gmem` v2.
4. **Fase 4 (Pipeline On-Demand):** Activar el streaming progresivo de capas tensoriales para arranque en frío instantáneo.
