# 🧬 Plan de Implementación de Producción: Q4_0 Nano/Pico (PPL < 10 & Certificación Factual)

> **Versión:** `v1.7.1-production`  
> **Fecha:** Septiembre 2026  
> **Estado:** 🚀 `LISTO PARA EJECUCIÓN (PRIORIDAD 1)`  
> **Objetivo:** Consolidar los modelos `gaje_nano_0_5b.flat` y `gaje_pico_135m.flat` con PPL < 10, throughput > 19-32 tok/s, 0 NaN/Inf y retención de paridad factual mediante destilación DNI y `gaje-cli distill`.

---

## 1. 🎯 Objetivos y Criterios de Éxito (Definition of Done)

1. **Perplejidad Held-Out (PPL):**
   * $\text{PPL} < 10.0$ en el corpus de validación `unified_distilled_corpus.jsonl` (reducción drástica desde los ~45.0 de variantes exploratorias).
2. **Fidelidad Semántica y Paridad Factual:**
   * $100\%$ de acierto en los 20 prompts de control de paridad (ej. `Alemania → Berlín`, `Planeta más grande → Júpiter`, `Ebullición del agua → 100°C`).
   * Eliminación absoluta de bucles repetitivos y degeneración léxica sin penalización destructiva.
3. **Subcomando Nativo Soberano (`gaje-cli distill`):**
   * Integración de la herramienta de destilación CLI en Rust (`src/bin/gaje-cli.rs`), orquestando el `DistillationGraph` y los shaders GPU WGSL sin dependencias externas de Python en producción.
4. **Rendimiento e Integridad Física:**
   * Throughput de generación $\ge 19\text{ tok/s}$ en `nano_0_5b` y $\ge 30\text{ tok/s}$ en `pico_135m` sobre CPU/GPU combinada.
   * `gaje-cli audit` completamente verde (0 NaN, 0 Inf, entropía de pesos equilibrada).

---

## 2. 🏛️ Arquitectura del Pipeline de Destilación DNI

```
                               ┌──────────────────────────────────────────────┐
                               │   data/unified_distilled_corpus.jsonl        │
                               │   (150 secuencias calibradas de alta pureza) │
                               └──────────────────────┬───────────────────────┘
                                                      │
                                                      ▼
                       ┌─────────────────────────────────────────────────────────────┐
                       │                   gaje-cli distill                          │
                       │           (src/bin/gaje-cli.rs & DistillationGraph)         │
                       └──────────────────────┬──────────────────────────────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
        ┌─────────────────────────┐                       ┌─────────────────────────┐
        │  MODELO MAESTRO 3B      │                       │  MODELO ALUMNO Q4_0     │
        │  (Qwen2.5 3B .flat)     │                       │  (nano 0.5B / pico 135M)│
        └────────────┬────────────┘                       └────────────┬────────────┘
                     │ Forward (Q4_0 / FP16)                           │ Forward Batched
                     ▼                                                 ▼
             [ Logits Maestro P_t ]                            [ Logits Alumno P_s ]
                     │                                                 │
                     └────────────────────────┬────────────────────────┘
                                              │
                                              ▼
                         ┌───────────────────────────────────────────┐
                         │   Shader GPU WGSL (kl_divergence.wgsl)    │
                         │   $\mathcal{L}_{KD} = D_{KL}(P_t \parallel P_s)$ $(\alpha=0.7)$  │
                         └────────────────────┬──────────────────────┘
                                              │
                                              ▼
                         ┌───────────────────────────────────────────┐
                         │      Refinamiento SFT / STE lm_head       │
                         │   (Actualización FP32 en VRAM zero-copy)  │
                         └────────────────────┬──────────────────────┘
                                              │
                                              ▼
                               ┌─────────────────────────────┐
                               │ models/production/          │
                               │   gaje_nano_0_5b.flat (v2)  │
                               └─────────────────────────────┘
```

---

## 3. 🛠️ Fases de Ejecución Técnica

### 🔹 Fase 1: Preparación y Curaduría del Corpus de Destilación
* **Archivo:** `scripts/data_processing/build_distilled_corpus.py` o subcomando `gaje-cli dataset-build`.
* **Acciones:**
  1. Unificar y filtrar `data/distill/gemma4_distillation_dataset.jsonl` y `data/latam_corpus.jsonl`.
  2. Extraer **150 secuencias multi-dominio** (ciencia, geografía, lógica de programación, traducción y síntesis factual) delimitadas con tokens de chat estándar (`<|im_start|>` / `<|im_end|>`).
  3. Particionar en **Train (80%)** y **Held-out Validation (20%)** para medición objetiva de PPL.

### 🔹 Fase 2: Implementación del Subcomando CLI `gaje-cli distill` (✅ COMPLETADO)
* **Archivos:** [`src/bin/gaje-cli.rs`](file:///data/data/com.termux/files/home/develop/gaje-semantic-compression/src/bin/gaje-cli.rs), [`src/nn/distiller/teacher.rs`](file:///data/data/com.termux/files/home/develop/gaje-semantic-compression/src/nn/distiller/teacher.rs).
* **Estado:** Integrado y certificado en el binario release `gaje-cli`.
* **Estructura CLI Implementada:**
  ```rust
  #[derive(Args, Debug)]
  struct DistillArgs {
      /// Ruta del modelo maestro (.flat v2)
      #[arg(short, long)]
      teacher: String,

      /// Ruta del modelo alumno a afinar (.flat v2)
      #[arg(short, long)]
      student: String,

      /// Dataset en formato JSONL delimitado
      #[arg(short, long)]
      dataset: String,

      /// Épocas de entrenamiento (por defecto: 3)
      #[arg(short, long, default_value_t = 3)]
      epochs: usize,

      /// Tasa de aprendizaje (learning rate)
      #[arg(short, long, default_value_t = 0.001)]
      lr: f32,

      /// Ponderación de divergencia KL vs SFT (alpha)
      #[arg(long, default_value_t = 0.7)]
      alpha: f32,

      /// Tamaño de batch para procesamiento en GPU/CPU
      #[arg(long, default_value_t = 32)]
      batch_size: usize,

      /// Archivo de salida para el modelo refinado (.flat)
      #[arg(short, long)]
      output: String,
  }
  ```
* **Conexión Interna:**
  * Enlazar con `DistillationGraph::new(...)` de `src/nn/distiller/graph.rs`.
  * Utilizar el despachador `GpuPipeline` (`src/compute/gpu/pipeline.rs`) para ejecutar `kl_divergence.wgsl` y la retropropagación acotada al `lm_head` FP32 del alumno.

### 🔹 Fase 3: Ejecución de la Destilación DNI & Optimización
* **Acciones:**
  1. Ejecutar destilación de `deepseek_r1_1_5b_q4_0.gaje.flat` (o Qwen2.5 3B) $\to$ `gaje_nano_0_5b.flat`.
  2. Ejecutar destilación $\to$ `gaje_pico_135m.flat`.
  3. Validar convergencia de Loss ($\mathcal{L}_{total} = \alpha \mathcal{L}_{KD} + (1-\alpha) \mathcal{L}_{CE}$) en cada época sin divergencias numéricas.

### 🔹 Fase 4: Auditoría, Benchmarks y Certificación Oficial
* **Acciones:**
  1. `gaje-cli audit --model models/production/gaje_nano_0_5b.flat`: Comprobar ausencia de `NaN`, `Inf` y coherencia de cabecera `FlatHeaderV2`.
  2. `gaje-cli bench --model models/production/gaje_nano_0_5b.flat --tokens 64`: Medir TTFT, decode throughput (tok/s) y PPL.
  3. Ejecutar los 20 prompts de evaluación factual con temperatura 0.0 (greedy) y registrar los resultados en `docs/reports/Q4_0_PRODUCTION_CERTIFICATION_v1_7_1.md`.
  4. Actualizar [`docs/meta/EMPIRICAL_TRUTH_STATE.md`](../meta/EMPIRICAL_TRUTH_STATE.md) reflejando el nuevo estado certificado.

---

## 4. 🧪 Escenarios BDD (Behavior-Driven Development)

```gherkin
Característica: Destilación DNI y Certificación de Producción Q4_0
  Como desarrollador del ecosistema GAJE
  Quiero destilar y certificar los modelos nano_0_5b y pico_135m
  Para garantizar PPL < 10 y respuestas factuales exactas en producción

  Escenario: Destilación DNI acelerada vía CLI nativo
    Dado un modelo maestro "models/production/deepseek_r1_1_5b_q4_0.gaje.flat"
    Y un modelo alumno "models/production/gaje_nano_0_5b.flat"
    Y un corpus limpio "data/unified_distilled_corpus.jsonl"
    Cuando ejecuto "gaje-cli distill --teacher ... --student ... --epochs 3 --alpha 0.7"
    Entonces el proceso finaliza con código de salida 0
    Y la pérdida total decrece monotonicamente entre épocas
    Y el archivo resultante contiene tensores Q4_0 válidos y lm_head FP32 actualizado

  Escenario: Certificación de PPL y paridad factual en inferencia
    Dado el modelo destilado "models/production/gaje_nano_0_5b.flat"
    Cuando ejecuto el benchmark con "gaje-cli bench --tokens 32"
    Entonces la Perplejidad (PPL) en held-out es menor a 10.0
    Y la generación greedy ante "¿Cuál es la capital de Alemania?" responde "Berlín"
    Y la generación greedy ante "¿Cuál es el planeta más grande?" responde "Júpiter"
    Y la auditoría con "gaje-cli audit" reporta 0 NaNs y 0 Infs
```

---

## 5. 📅 Plan de Acción Inmediato

1. **Paso 1:** Añadir el comando `Distill(DistillArgs)` en `src/bin/gaje-cli.rs` y conectar la lógica con `src/nn/distiller/`.
2. **Paso 2:** Preparar y validar el dataset limpio `data/unified_distilled_corpus.jsonl`.
3. **Paso 3:** Ejecutar la corrida de destilación y generar los pesos afinados.
4. **Paso 4:** Certificar formalmente con la suite de auditoría y benchmarks.
