# 🧬 GAJE Evaluation Harness: Diseño de Arquitectura y Especificación Técnica

> **Versión:** v1.6.0-alpha (Silver Adult)
> **Fecha:** 19 de agosto de 2026
> **Estado:** 📝 Plan Aprobado / Diseño de Arquitectura
> **Ubicación:** `docs/plans/EVAL_HARNESS_DESIGN.md`

---

## 1. 🎯 Objetivo y Motivación

El **GAJE Evaluation Harness** es un sistema automatizado de evaluación multidimensional para modelos cuantizados y adaptados en formato plano **`.gaje.flat`**.

Su propósito es erradicar la evaluación subjetiva ("probar prompts a mano") y proporcionar una **certificación matemática y cualitativa instantánea** cada vez que se exporta o re-entrena un checkpoint de modelo.

---

## 2. 🏛️ Arquitectura de las 4 Dimensiones

El harness somete a cualquier modelo `.gaje.flat` (y opcionalmente a un modelo base de control) a cuatro pruebas rigurosas e independientes:

```
                     ┌──────────────────────────────────────────┐
                     │          GAJE EVAL HARNESS               │
                     │       (scripts/eval_harness.py)          │
                     └────────────────────┬─────────────────────┘
                                          │
        ┌─────────────────────┬───────────┴───────────┬─────────────────────┐
        ▼                     ▼                       ▼                     ▼
 1. FACTUAL & OOD      2. ANTI-LOOPS           3. HARDWARE & SPEED   4. KL DIVERGENCE
 (¿Sabe responder?)    (¿Cero repetición?)     (tok/s, RAM, mmap)   (¿Olvido catastrófico?)
```

---

### 📊 Dimensión 1: Factualidad y Preservación OOD (Out-of-Domain)
* **Objetivo:** Evaluar la precisión de las respuestas tanto en conocimiento dentro del corpus (*in-domain*) como fuera del corpus (*OOD*), validando que el entrenamiento no haya causado **olvido catastrófico** (*catastrophic forgetting*).
* **Métrica:** Tasa de acierto de palabras clave/entidades objetivo ($Accuracy \in [0.0, 1.0]$).
* **Categorías de Evaluación (20 prompts fijos):**
  1. **Geografía y Capitales:** (París, Madrid, Tokio, etc.).
  2. **Ciencia y Naturaleza:** (Punto de ebullición del agua, fotosíntesis, gravedad).
  3. **Código y Sintaxis:** (Funciones Python, bucles `for`, listas).
  4. **Traducción Directa:** (Español $\leftrightarrow$ Inglés).
  5. **Conocimiento OOD:** Hechos deliberadamente omitidos del entrenamiento de destilación.

---

### 🔁 Dimensión 2: Coherencia y Detección de Lazos (Anti-Gibberish)
* **Objetivo:** Identificar degeneraciones de lenguaje, frases cortadas y bucles infinitos de retroalimentación (como *"Por,,,,,"* o *"en el mar es el mar"*).
* **Métricas:**
  * **Diversidad de N-gramas ($NDIV_3, NDIV_4$):** Proporción de 3-gramas y 4-gramas únicos respecto al total.
    $$NDIV_n = \frac{|\text{N-gramas únicos}|}{|\text{N-gramas totales}|}$$
  * **Tasa de Bucles ($LoopRate$):** Detección de subsecuencias repetidas $\ge 3$ veces consecutivas.
  * **Cierre Limpio (EOS Check):** Verificación de que el modelo emite correctamente el token de fin de secuencia (`<|im_end|>` o `<|endoftext|>`).

---

### ⚡ Dimensión 3: Benchmarking Físico de Hardware
* **Objetivo:** Garantizar que el modelo mantenga el estándar de inferencia ultra-rápida en CPU sin regresiones de rendimiento.
* **Métricas:**
  * **Throughput E2E:** Velocidad de generación sostenida en CPU (`tokens/segundo`).
  * **Time-to-First-Token (TTFT):** Latencia en milisegundos desde la recepción del prompt hasta el primer token emitido.
  * **Cold Start (`mmap`):** Tiempo en milisegundos para mapear el archivo `.gaje.flat` a memoria.
  * **Peak Memory Footprint (RAM):** Memoria física residente (RSS) utilizada por el proceso en MB.

---

### 📐 Dimensión 4: Divergencia de Logits ($D_{KL}(P_{base} \parallel P_{candidate})$)
* **Objetivo:** Medir cuantitativamente la distancia probabilística entre las predicciones del modelo base original y el modelo adaptado.
* **Fórmula:**
  $$D_{KL}(P \parallel Q) = \sum_{x \in \mathcal{V}} P(x) \log\left(\frac{P(x)}{Q(x) + \epsilon}\right)$$
* **Criterio de Decisión:**
  * $D_{KL} \le 0.5$: Adaptación suave y controlada (mantiene la distribución previa).
  * $0.5 < D_{KL} \le 1.5$: Adaptación fuerte (cambio significativo de estilo/idioma).
  * $D_{KL} > 2.5$: **Alerta de Colapso / Corrupción de Pesos** (rechazar checkpoint).

---

## 3. 🧮 Algoritmo de Puntuación Global (GAJE Score 0 - 100)

El harness consolidará los resultados en una única nota global ponderada:

$$\text{GAJE Score} = 0.35 \cdot S_{\text{factual}} + 0.30 \cdot S_{\text{coherence}} + 0.20 \cdot S_{\text{loops}} + 0.15 \cdot S_{\text{speed}}$$

| Rango de Score | Clasificación | Veredicto |
| :---: | :---: | :--- |
| **$90 - 100$** | **Diamond Grade** | 🏆 Listo para producción / Release oficial |
| **$75 - 89$** | **Silver Grade** | 🟢 Estable / Modelo de alta calidad funcional |
| **$50 - 74$** | **Bronze Grade** | 🟡 Aceptable para pruebas / Requiere más afinación |
| **$< 50$** | **Degenerado** | 🔴 Rechazado (Descartar checkpoint) |

---

## 4. 🛠️ Interfaz de Línea de Comandos (CLI) y Uso

El script principal residirá en `scripts/eval_harness.py` y se invocará de forma sencilla:

```bash
# Evaluación individual de un checkpoint
python scripts/eval_harness.py \
  --model models/production/smollm2_4bit_clean.gaje.flat

# Evaluación comparativa A/B contra el modelo base
python scripts/eval_harness.py \
  --model models/production/smollm2_4bit_clean.gaje.flat \
  --baseline models/production/smollm2_4bit.gaje.flat \
  --out docs/reports/harness_smollm2_clean.md
```

---

## 5. 📋 Formato del Reporte de Salida

El harness generará una tabla comparativa en consola y un reporte estructurado en Markdown:

```markdown
# 🧬 GAJE Evaluation Report: smollm2_4bit_clean.gaje.flat

| Dimensión de Evaluación | Modelo Base | Modelo Candidato | Variación | Estado |
| :--- | :---: | :---: | :---: | :---: |
| **Factualidad Global (20 Prompts)** | 65.0% | **85.0%** | +20.0% | 🟢 Superado |
| **Diversidad N-gramas ($NDIV_4$)** | 0.72 | **0.94** | +0.22 | 🟢 Superado |
| **Inmunidad a Lazos / Loops** | 80.0% | **100.0%** | +20.0% | ✅ Cero Loops |
| **Throughput CPU** | 30.5 tok/s | **30.2 tok/s** | -1.0% | 🟢 Óptimo |
| **Cold Start mmap** | 0.81 ms | **0.78 ms** | -0.03 ms | 🟢 Instantáneo |
| **Consumo de Memoria RAM** | 142 MB | **142 MB** | 0.0 MB | 🟢 Estable |
| **Divergencia KL ($D_{KL}$)** | 0.0000 | **0.3842** | +0.3842 | 🟢 Suave |
| **Puntuación Global (Score)** | **67.2 / 100** | **88.6 / 100** | **+21.4 pts** | 🏆 **Silver Grade** |
```

---

## 6. 📅 Plan de Implementación

1. **Fase 1 (Banco de Prompts y Métricas):** Crear el archivo JSON con los 20 prompts de prueba calibrados con respuestas ideales y palabras clave (`tests/fixtures/eval_harness_prompts.json`).
2. **Fase 2 (Motor del Harness):** Escribir `scripts/eval_harness.py` integrando la inferencia nativa PyO3 de `GenomicLLM`.
3. **Fase 3 (Generación de Reportes):** Añadir exportadores en formato Markdown y JSON.
4. **Fase 4 (Integración con el Flujo de Entrenamiento):** Integrar la llamada al harness al final de `export_trained.rs` para certificar automáticamente cada export producido.
