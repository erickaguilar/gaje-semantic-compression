# 🛠️ GAJE-CLI: Estatus Global, Capacidades Operativas y Límites del Motor

> **Versión:** v1.6.0-alpha (Silver Adult)
> **Fecha:** 20 de agosto de 2026
> **Ubicación:** `docs/guides/GAJE_CLI_CAPABILITIES_AND_LIMITS.md`
> **Componente:** Herramienta de Terminal y Motor Nativo (`src/bin/gaje-cli.rs`)

---

## 1. 📋 Resumen Ejecutivo

`gaje-cli` es el binario de control e inferencia nativa en **Rust puro** del protocolo GAJE. Proporciona una interfaz de línea de comandos de alto rendimiento para interactuar directamente con modelos cuantizados en formato plano **`.gaje.flat`** y bases de datos genómicas **`.gaje`** sin intermediación de Python.

---

## 2. 🟢 Capacidades Operativas (¿Qué SÍ hace `gaje-cli`?)

```
                                  GAJE-CLI NATIVO (Rust)
                                             │
      ┌──────────────────┬───────────────────┼───────────────────┬──────────────────┐
      ▼                  ▼                   ▼                   ▼                  ▼
 INFERENCIA CHAT    INSPECCIÓN MMAP     IMPORTACIÓN GGUF    EVALUACIÓN CE/PPL   PRESETS NATIVOS
 (CPU ~30 tok/s)    (HeaderV2, dims)    (Conversión Q4_0)   (Pérdida en disco)  (Born-Genomic)
```

### ⚡ A. Inferencia y Chat Interactivo (Zero-Copy)
* **Comando:** `gaje-cli --model <path.flat> --prompt "<texto>"`
* **Arranque en Frío:** Carga instantánea vía `mmap` en **menos de $0.75\text{ ms}$**.
* **Throughput Certificado (Ryzen 7 5800H):**
  * **SmolLM2 135M:** `28 - 32 tok/s` (RAM: $\approx 140\text{ MB}$).
  * **Qwen2.5 1.5B:** `11 - 12 tok/s` (RAM: $\approx 1.2\text{ GB}$).
  * **Qwen2.5 3B:** `3 - 4 tok/s` (RAM: $\approx 2.2\text{ GB}$).
* **Samplers Nativos:** Control estocástico por temperatura, penalización de repetición por $n$-gramas y física lagrangiana de mínima acción.

### 🔍 B. Inspección Profunda de Arquitectura (`--inspect`)
* **Comando:** `gaje-cli --model <path.flat> --inspect`
* Lee y valida dinámicamente la cabecera binaria `FlatHeaderV2` y el `ArchitectureDescriptor`: dimensiones $d_{model}$, número de cabezas de atención, capas de transformador, constantes RoPE y formato de cuantización (Q4_0, Q8_0, FP32).

### 🔄 C. Transmutación e Importación GGUF (`--import`)
* **Comando:** `gaje-cli --import <modelo.gguf> --output <modelo.gaje> --threshold 0.15`
* Transforma modelos estándar GGUF de Hugging Face a formato genómico `.gaje` o `.flat` con inyección de anclas de estabilidad en FP16.

### 🔤 D. Tokenización en Vivo (`--tokenize`)
* **Comando:** `gaje-cli --tokenize "<texto>"`
* Muestra la segmentación de texto en tokens IDs mediante el tokenizer integrado en Rust sin requerir Python.

### 🧬 E. Inicialización de Organismos "Born-Genomic" (`--init --preset`)
* **Comando:** `gaje-cli --init <path.gaje> --preset <tipo>`
* Genera estructuras neuronales genómicas desde cero con identificador único DNI:
  * `micro_organism` (128 embd / 2 capas / 4 cabezas)
  * `gold_embryo` (384 embd / 8 capas / 6 cabezas)
  * `silver_adult` (512 embd / 12 capas / 8 cabezas)
  * `titan` (1024 embd / 36 capas / 16 cabezas)

### 💾 F. Inyección DNI y Resonancia Contextual (`--dni-ingest`)
* **Comando:** `gaje-cli --model <path> --dni-ingest <archivo.txt> --intensity 0.01`
* Permite inyectar vectores de conocimiento mediante resonancia de fase compleja.

---

## 3. 🔴 Límites y Fuera de Alcance (¿Qué NO hace `gaje-cli`?)

1. **NO realiza Pre-entrenamiento Masivo desde Cero (Estilo Llama/GPT):**
   * No está diseñado para pre-entrenar modelos con billones de tokens de internet. Su propósito es compresión extrema (4-bit Q4_0), inferencia en CPU y fine-tuning focalizado (IQAT / Destilación).
2. **NO ejecuta en Clústeres GPU Distribuidos (Multi-Nodo):**
   * No implementa CUDA, ROCm, NCCL ni DeepSpeed. Opera **100% en CPU local multinúcleo con paralelismo Rayon e intrínsecos SIMD (AVX2/FMA)**.
3. **NO usa Mutación Genética a Ciegas en Modelos Grandes (Archivado ⚠️):**
   * Los comandos heredados de evolución estocástica a 2-bits (`--evolve`, `--gens`) están congelados para redes $\ge 135\text{M}$ por la maldición de la dimensionalidad ($4^{135\text{M}}$ estados).
4. **NO es un Servidor HTTP / Web UI por sí solo:**
   * Es una herramienta de terminal de bajo nivel. La interfaz web gráfica y la API REST se sirven mediante `examples/ui/web_ui/server.py`.
5. **NO entrena el Vocabulario (`lm_head`) en Producción:**
   * Para evitar el colapso del diccionario multilingüe en español y chino, el fine-tuning del motor congela el `lm_head` por diseño.

---

## 4. 📊 Matriz de Estado Operativo de `gaje-cli`

| Funcionalidad | Subcomando / Flag | Estado |
| :--- | :--- | :---: |
| **Inferencia Rápida** | `--model <path> --prompt "<txt>"` | 🟢 **Certificado** |
| **Inspección de Metadatos** | `--inspect` | 🟢 **Certificado** |
| **Tokenización Nativa** | `--tokenize "<txt>"` | 🟢 **Certificado** |
| **Importación GGUF** | `--import <path>` | 🟢 **Certificado** |
| **Evaluación de Pérdida** | `--eval <corpus>` | 🟢 **Certificado** |
| **Inicialización de Organismos** | `--init <path> --preset <preset>` | 🟢 **Certificado** |
| **Evolución 2-Bits (Grandes)** | `--evolve <target>` | 🔴 **Congelado / R&D** |
