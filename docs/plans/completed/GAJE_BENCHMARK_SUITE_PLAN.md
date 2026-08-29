# 📊 Plan de Implementación: Suite Unificada `gaje benchmark`

> **Versión:** v1.6.0-alpha (Silver Adult)  
> **Fecha:** 20 de agosto de 2026  
> **Estado:** 📝 Especificación Técnica y Plan de Ejecución  
> **Ubicación:** `docs/plans/GAJE_BENCHMARK_SUITE_PLAN.md`  
> **Componentes Asociados:** `src/bin/gaje-cli.rs`, `scripts/gaje_benchmark.py`, `docs/reports/`  

---

## 1. 🎯 Objetivo del Proyecto

Construir una **suite de benchmarking oficial, reproducible y automatizada** integrada nativamente en el ecosistema GAJE. La suite permitirá certificar y comparar cualquier modelo en formato plano `.gaje.flat` a través de un único comando:

```bash
gaje-cli benchmark --models models/production/*.gaje.flat --format markdown --output docs/reports/BENCHMARK_OFFICIAL.md
```

---

## 2. 📐 Las 4 Dimensiones de Evaluación Certificadas

```
                             GAJE BENCHMARK SUITE
                                       │
        ┌───────────────────┬──────────┴───────────┬───────────────────┐
        ▼                   ▼                      ▼                   ▼
 1. LATENCIA & INICIO  2. RENDIMIENTO CPU    3. HUELLA DE MEMORIA  4. CALIDAD GENERATIVA
 • Cold-start (mmap)   • Prompt Eval (tok/s)  • Peak RAM (RSS MB)   • Diversidad (d1/d2)
 • TTFT (ms)           • Token Gen (tok/s)    • Ratio de Compresión • % Degeneración (0%)
```

### ⏱️ Dimensión 1: Latencia y Arranque
* **Cold-Start Time ($\mu\text{s}$ / $\text{ms}$):** Tiempo transcurrido desde la invocación del comando hasta que el `mmap` expone el modelo listo en memoria.
* **Time-To-First-Token (TTFT en $\text{ms}$):** Tiempo requerido para procesar el prompt de entrada y emitir el primer token autoregresivo.

### ⚡ Dimensión 2: Rendimiento de Cómputo CPU
* **Prompt Processing Speed ($\text{tok/s}$):** Velocidad de prefill procesando contextos de 32, 128 y 512 tokens.
* **Sustained Generation Speed ($\text{tok/s}$):** Rendimiento de decodificación token a token en un hilo y con paralelismo Rayon (multinúcleo).

### 💾 Dimensión 3: Huella de Memoria Física
* **Peak Resident Set Size (RSS en $\text{MB}$):** Medición real del consumo de RAM física durante la generación continua.
* **Ratio de Compresión vs FP32:** Porcentaje de ahorro de memoria frente a la referencia dorada sin cuantizar.

### 🧠 Dimensión 4: Calidad y Retención Semántica
* **Diversidad Léxica ($d_1, d_2$):** Fracción de unigramas y bigramas únicos generados.
* **Tasa de Degeneración ($\%\text{ de loops}$):** Porcentaje de respuestas que caen en bucles repetitivos o balbuceo.
* **Banco Factual Multilingüe:** Verificación A/B con 25 prompts held-out en Español, Inglés, Chino y Razonamiento Algebraico en LaTeX.

---

## 3. 🧪 Matriz de Modelos a Certificar

| Modelo | Arquitectura | Parámetros | Precisión Híbrida | Archivo Flat |
| :--- | :---: | :---: | :---: | :--- |
| **SmolLM2-135M Base** | SmolLM2 | 135M | Q4_0 + FP32 | `models/production/smollm2_4bit.gaje.flat` |
| **SmolLM2-135M Quality** | SmolLM2 | 135M | Q4_0 + FP32 | `models/production/smollm2_4bit_quality.gaje.flat` |
| **Qwen2-0.5B Core** | Qwen2 (GQA) | 500M | Q4_0 + Q8_0 | `models/production/qwen2_0_5b_q4_0_q8_0_embd.gaje.flat` |
| **Qwen2.5-1.5B Mid** | Qwen2.5 | 1.5B | Q4_0 + Q8_0 | `models/production/qwen2_5_1_5b_q4_0_q8_0_embd.gaje.flat` |
| **Qwen2.5-3B Max** | Qwen2.5 | 3.0B | Q4_0 + Q8_0 | `models/production/qwen2_5_3b_q4_0_q8_0_embd.gaje.flat` |

---

## 4. 🏛️ Arquitectura de la Herramienta

La suite se implementará en dos capas complementarias:

### 4.1 Capa Nativa en Rust (`src/bin/gaje-cli.rs` + `src/bench/mod.rs`)
* Mide métricas de latencia de ultra-precisión en microsegundos usando `std::time::Instant` y lecturas directas de `/proc/self/statm` para RSS.
* Genera salida directa en terminal formateada en ASCII/Unicode o JSON crudo.

### 4.2 Harness de Evaluación Automatizado (`scripts/gaje_benchmark.py`)
* Orquesta corridas comparativas completas.
* Aplica el banco de prompts estandarizado (`data/eval/benchmark_prompts.json`).
* Exporta tablas Markdown automáticas listas para ser insertadas en `README.md` y `EMPIRICAL_TRUTH_STATE.md`.

---

## 5. 📅 Fases de Ejecución

```
                                CRONOGRAMA DE EJECUCIÓN
                                           │
         ┌───────────────────┬─────────────┴─────────────┬───────────────────┐
         ▼                   ▼                           ▼                   ▼
      FASE 1              FASE 2                      FASE 3              FASE 4
  Banco de Prompts    Módulo de Métricas         CLI Runner Nativo    Reporte Oficial
  (25 prompts JSON)   (RSS, TTFT, tok/s, d1/d2)  (`gaje benchmark`)   (Tabla Markdown)
```

### 🔹 Fase 1: Creación del Banco de Prompts Estandarizado
* Crear `data/eval/benchmark_prompts.json` con 25 casos de prueba categorizados:
  * 10 Factuales en Español (Geografía, Biología, Historia).
  * 5 Razonamiento / Ecuaciones en LaTeX.
  * 5 Programación y Sintaxis (Python).
  * 5 Multilingües (Inglés / Chino).

### 🔹 Fase 2: Módulo de Telemetría y Métricas en Rust/Python
* Implementar recolectores de:
  * `cold_start_us`: Tiempo de carga mmap.
  * `ttft_ms`: Tiempo hasta el primer token emitido.
  * `throughput_tok_s`: Tokens por segundo sostenidos.
  * `peak_rss_mb`: Memoria física máxima alcanzada.
  * `distinct_1_2`: Diversidad léxica objetiva.

### 🔹 Fase 3: Integración del Subcomando en `gaje-cli` y Script Python
* Extender `gaje-cli` para aceptar el subcomando `benchmark`.
* Construir `scripts/gaje_benchmark.py` con soporte para exportar en JSON y Markdown.

### 🔹 Fase 4: Ejecución de la Corrida Oficial y Publicación
* Ejecutar la corrida completa en la máquina de referencia (AMD Ryzen 7 5800H / 16 threads / Fedora Linux).
* Generar `docs/reports/BENCHMARK_OFFICIAL_v1_6.md` y sincronizar en `develop`.

---

## 6. 🏆 Criterios de Éxito y Aceptación

1. **Ejecución de 1 Solo Comando:** La suite completa debe ejecutarse sin intervención manual (`python3 scripts/gaje_benchmark.py`).
2. **Determinismo y Repetibilidad:** Tres corridas consecutivas del benchmark deben arrojar una desviación de throughput menor al $\pm 3\%$.
3. **Cero Dependencias Externas en Inferencia:** Todo el cálculo de generación y mmap debe operar exclusivamente sobre los binarios nativos de GAJE.
