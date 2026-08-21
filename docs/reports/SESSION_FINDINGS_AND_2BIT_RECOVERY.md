# 🧬 Reporte Integral de Hallazgos: Diagnóstico de Entrenamiento Q4_0 y Rescate del Paradigma Neuromórfico 2-Bit

> **Versión:** v1.6.0-alpha (Silver Adult)
> **Fecha:** 19 de agosto de 2026
> **Autor:** Antigravity AI & Erick Aguilar
> **Estado:** 🟢 Aprobado / Verdad Empírica Consolidada
> **Ubicación:** `docs/reports/SESSION_FINDINGS_AND_2BIT_RECOVERY.md`

---

# PARTE I: Hallazgos de la Sesión de Entrenamiento y Calidad Q4_0

## 1. 🔬 Mecanismo de Backpropagation en Rust (Validado)
* **Gradient Check Numérico:** Confirmado contra diferencias finitas en `f64` con error relativo $\le 0.06$.
* **Bugs Críticos Corregidos:**
  1. *Transpose en `backward_core`:* Corrección del mapeo de nibbles (par $\rightarrow$ alto, impar $\rightarrow$ bajo).
  2. *Straight-Through Estimator (STE):* Corrección en `refine_with_grads_core` (se eliminó la división por `centroid_counts`; el gradiente real del centroide es la suma $\sum g \cdot x$).
* **Punto Dulce de Entrenamiento:** **8 bloques**, $lr \approx 2\times 10^{-4}$, $\text{grad\_clip} = 1.0 \rightarrow$ Held-out CE: **`2.427`** vs Baseline **`2.559`** ($\Delta = -0.13$).
* **Escalado por Capas (*Layer-wise Decay*):** La regla $lr_b = lr \cdot \text{decay}^{(n-1-b)}$ permite entrenar hasta 24 bloques de forma matemáticamente estable sin NaNs.

---

## 2. ⚡ Rendimiento y Diagnóstico de Cómputo
* **Viabilidad de Escala:** La corrida de 30k tokens sobre `dataset_1000.txt` se completó en **5.7 horas** en CPU (descartando la creencia previa de que tomaría semanas).
* **Aceleración Per-Secuencia:** El entrenamiento con caché reseteado por pareja `{"prompt": ..., "answer": ...}` es **$\approx 150\times$ más rápido**, procesando 22k tokens en apenas **~2 minutos**.

---

## 3. 🎯 Diagnóstico de Calidad y Causa Raíz
* **Hipótesis de Tokenización Refutada:** El CE base de `corpus_unified` fue medido en **2.99** (y no ~4.0). El tokenizer nunca fue el problema.
* **El CE NO Correlaciona con la Generación:**
  * `dataset_1000.txt` bajó el CE en **$-1.05$** (mayor ganancia numérica), pero produjo colapso y repetición severa (*"Por,,,,,"*).
  * `distill` bajó el CE en solo **$-0.08$**, pero retuvo coherencia total y texto estructurado.
* **Causa Raíz Identificada:** Un stream continuo de texto sin separadores premia la continuación del ruido local. Al aislar por secuencias delimitadas independientes, el CE base baja de **4.53** a **2.83**.
* **Lección Transversal:** **La entropía cruzada (CE) es una métrica auxiliar; la capacidad generativa es la única métrica de éxito real.**

---

## 4. 🏆 Veredicto del Harness Generativo Objetivo (`eval_generation.py`)

| Modelo | Corpus / Configuración | Diversidad Lexical | % Degeneradas | Veredicto |
| :--- | :--- | :---: | :---: | :--- |
| **`smollm2_4bit_quality.gaje.flat`** | **Destilación 1520 tok (`lm_head` congelado)** | **Máxima** | **0.0% (Cero loops)** | 🏆 **Campeón de Generación** |
| **`smollm2_4bit.gaje.flat`** | Base HuggingFace sin entrenar | Media | 12.5% - 25.0% | 🟡 Línea de control |
| **`smollm2_4bit_clean / big / trained`** | Entrenamientos masivos del cuerpo (22k - 30k) | Baja | Alta repetición | 🔴 Sobreajustado |

---

# PARTE II: Hoja de Ruta para el Rescate del Paradigma 2-Bit y Neuromórfico

## 1. 🛑 Por Qué Falló el Intento Anterior en Redes de 135M+
1. **La Maldición de la Dimensionalidad:** Con 135 millones de pesos en 2-bits, el espacio de búsqueda combinatorial es astronómico ($4^{135,000,000}$). La mutación estocástica a ciegas no converge en tiempo finito.
2. **Deriva Exponencial Multicapa:** Una pérdida de similitud del 3% por capa se acumula a través de 120 proyecciones:
   $$0.97^{120} \approx 0.02 \quad (\text{Colapso a ruido en logits finales})$$
3. **Costo de Inferencia en CPU:** Evaluar poblaciones genéticas sobre 135M tomaba 7 minutos por generación.

---

## 2. 🚀 Las 4 Rutas Viables de Rescate Científico

```
                      RUTAS DE RESCATE PARA 2-BITS
                                   │
      ┌────────────────┬───────────┴───────────┬────────────────┐
      ▼                ▼                       ▼                ▼
  1. MICRO-ORGANISMOS  2. BITNET b1.58 (STE)   3. ANCLAS 5% FP16 4. HARDWARE REAL
    (1M - 5M params)     (Gradientes, no azar)   (Híbrido Spiking) (FPGA / Loihi)
```

### 🟢 Ruta 1: Micro-Embriones Nativos (1M a 5M de parámetros)
* **Concepto:** Diseñar y entrenar micro-redes nacidas directamente en 2-bits (*Born-Genomic*) desde cero para tareas de borde ultraligeras.
* **Casos de Uso:** Clasificadores de audio en tiempo real, detección de anomalías en sensores IoT, *Keyword Spotting* en microcontroladores.
* **Viabilidad:** En 1M-5M de parámetros, una generación genética toma **$< 1\text{ segundo}$**, permitiendo convergencia evolutiva en horas.

### 🟢 Ruta 2: Gradientes en 2-Bits / Ternario (Enfoque BitNet b1.58)
* **Concepto:** Abandonar la mutación aleatoria a ciegas y aplicar el motor de **Backpropagation en Rust con Straight-Through Estimator (STE)** directamente sobre cuantización ternaria ($\{-1, 0, +1\}$ o $\{00, 01, 10, 11\}$).
* **Viabilidad:** La red recibe la dirección matemática exacta de optimización por gradiente en cada paso, evitando el azar combinatorial.

### 🟢 Ruta 3: Esqueleto de Estabilidad Híbrido (Anclas 5% en FP16)
* **Concepto:** Mantener el 95% de los pesos en 2-bits y preservar el **Top 5% de pesos con mayor magnitud** (proyecciones críticas) en **FP16**.
* **Impacto Comprobado:** Eleva la similitud coseno por capa de $0.76$ a **$> 0.97$**, frenando el colapso semántico en la decodificación.

### 🟢 Ruta 4: Spiking Neural Networks (SNN) en Aceleradores Reales (FPGA / Loihi)
* **Concepto:** Aprovechar los módulos existentes de *Timing Wheel* y física lagrangiana (`src/compute/lagrangian.rs`, `src/nn/spiking/`) en hardware no-von Neumann.
* **Viabilidad:** En CPUs x86 la emulación secuencial de spikes es costosa, pero en FPGAs o procesadores neuromórficos reales consume **microjoules ($\mu\text{J}$) por inferencia**.

---

## 3. 🗺️ Matriz de Convivencia Estratégica en GAJE

| Paradigma | Estado | Rol en el Ecosistema GAJE |
| :--- | :---: | :--- |
| **Q4_0 + FP32/Q8_0 Híbrido** | 🟢 **Producción** | Motor de inferencia ultrarrápido ($30\text{ tok/s}$) para LLMs de 135M a 3B en CPU comercial. |
| **2-Bit / Neuromórfico Spiking** | 🔬 **Investigación R&D** | Reservado para micro-embriones IoT (1-5M), aceleración BitNet con gradientes Rust y chips neuromórficos. |
