# 🧬 Estrategia de Destilación y Transmutación de DeepSeek y Gemma en el Ecosistema GAJE

> **Versión:** v1.6.0-alpha (Silver Adult)  
> **Fecha:** 20 de agosto de 2026  
> **Estado:** 📝 Plan Estratégico y Especificación de Arquitectura  
> **Ubicación:** `docs/plans/DISTILLATION_DEEPSEEK_GEMMA_STRATEGY.md`  
> **Componentes Asociados:** `scripts/export_gaje_flat.py`, `scripts/generate_distill_corpus.py`, `src/io/loader.rs`  

---

## 1. 🎯 Visión General

La arquitectura modular y el soporte multimodelo de GAJE permiten aprovechar dos de las familias de modelos abiertos más potentes del mundo: **DeepSeek (DeepSeek-R1, DeepSeek-Coder)** y **Google Gemma (Gemma-2 2B/9B)**.

Este documento formaliza las **dos rutas de integración**:
1. **Destilación de Razonamiento (*Sequence-Level Distillation*):** Transferir habilidades cognitivas de razonamiento paso a paso (*Chain-of-Thought* / CoT) hacia los micro-estudiantes de 135M y 0.5B.
2. **Transmutación Directa a Formato Plano (*Native Ingestion*):** Convertir modelos GGUF de DeepSeek y Gemma a formato `.gaje.flat` para inferencia local a alta velocidad vía `mmap` zero-copy.

---

## 2. 🏛️ Las Dos Rutas de Integración

```
                    INTEGRACIÓN DE DEEPSEEK Y GEMMA EN GAJE
                                       │
        ┌──────────────────────────────┴──────────────────────────────┐
        ▼                                                             ▼
 RUTA 1: DESTILACIÓN DE COMPORTAMIENTO                RUTA 2: TRANSMUTACIÓN DIRECTA
 (Black-Box SFT / Micro-Estudiantes)                 (White-Box .gaje.flat Nativo)
 • Maestro: DeepSeek-R1 / Gemma-2-27B                • DeepSeek-R1-Distill-Qwen (1.5B)
 • Estudiante: SmolLM2-135M (140 MB)                 • Gemma-2-2B (Arquitectura GeGLU)
 • Habilidad: Razonamiento <think>...</think>        • Ejecución nativa en CPU (12 tok/s)
```

---

## 3. 🔍 Ruta 1: Destilación de Razonamiento DeepSeek-R1 en Micro-Modelos

### 💡 Concepto:
DeepSeek-R1 demostró que el razonamiento estructurado explícito mediante etiquetas `<think> ... </think>` permite a modelos más pequeños auto-corregir sus respuestas lógicas.

### ⚙️ Pipeline de Producción:
1. **Generación del Corpus (`scripts/generate_distill_corpus.py`):**
   * El maestro (DeepSeek-R1) recibe problemas de matemáticas, lógica y programación.
   * Emite la respuesta con su traza de pensamiento interna:
     ```text
     <think>
     Para saber la edad del hijo:
     1. El padre tiene 36 años.
     2. x = 12 representa la diferencia de edad...
     </think>
     El hijo tiene 24 años.
     ```
2. **Entrenamiento en GAJE (`examples/export_trained.rs`):**
   * Se entrena el cuerpo del estudiante **SmolLM2-135M** o **Qwen2 0.5B** con `train_lm_head = false` y 8 bloques.
3. **Resultado:**
   * Un **micro-modelo de 140 MB** capaz de formular pensamientos previos antes de emitir la conclusión final.

---

## 4. 🧬 Ruta 2: Transmutación Directa a Formato Plano `.gaje.flat`

### A. DeepSeek-R1-Distill-Qwen (1.5B y 7B) — ¡Compatibilidad Inmediata! 🚀
* **Estructura Arquitectónica:** Los modelos destilados de DeepSeek-R1 basados en Qwen utilizan **la arquitectura estándar Qwen2.5** (SwiGLU, RMSNorm, GQA con $n_{\text{head\_kv}}=2$, tensores fusionados `fused_qkv` y `fused_gate_up`).
* **Estado en GAJE:** **100% Soportado.**
* **Procedimiento:**
  ```bash
  python3 scripts/export_gaje_flat.py \
    --input models/source/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf \
    --output models/production/deepseek_r1_1_5b_q4_0_q8_0_embd.gaje.flat \
    --hybrid-v2
  ```
* **Rendimiento Esperado en CPU:** **`11 - 12 tok/s`** con consumo de $\approx 1.2\text{ GB}$ de RAM y arranque submilisegundo.

---

### B. Google Gemma / Gemma-2 (2B y 9B) — Adaptación de Motor 🛠️
Gemma-2 tiene particularidades matemáticas que requieren soporte en `src/io/loader.rs` y `src/nn/block/`:

1. **GeGLU Activation:**
   * Reemplaza SwiGLU por activación GELU aproximada con compuerta:
     $$\text{GeGLU}(x) = \text{GELU}(x \cdot W_{\text{gate}}) \odot (x \cdot W_{\text{up}})$$
2. **RMSNorm con Offset Unitario ($+1$):**
   * En Gemma, los pesos de normalización se aplican sumando 1.0 a los parámetros cargados:
     $$y = \frac{x}{\text{RMS}(x)} \odot (1.0 + \gamma)$$
3. **Logit Soft-Capping ($30.0 \cdot \tanh$):**
   * En las capas de atención y el `lm_head`, los logits se confinan para evitar explosión numérica:
     $$\text{Logits}_{\text{capped}} = 30.0 \cdot \tanh\left(\frac{\text{Logits}}{30.0}\right)$$

---

## 5. 📊 Comparativa de Opciones

| Modelo / Enfoque | Tipo | Tamaño RAM | Velocidad CPU | Dificultad Técnica |
| :--- | :---: | :---: | :---: | :---: |
| **DeepSeek-R1-Distill-1.5B Nativo** | Transmutación .flat | $\approx 1.2\text{ GB}$ | $11 - 12\text{ tok/s}$ | 🟢 **Inmediata (0 días)** |
| **Destilación CoT en SmolLM2-135M** | Destilación SFT | $\approx 140\text{ MB}$ | $30\text{ tok/s}$ | 🟢 **Baja (1 día)** |
| **Gemma-2-2B Nativo** | Extensión de Motor | $\approx 1.8\text{ GB}$ | $8 - 10\text{ tok/s}$ | 🟡 **Media (2-3 días)** |

---

## 6. 📅 Plan de Acción y Recomendación

### Fase Inmediata:
1. Convertir y certificar **DeepSeek-R1-Distill-Qwen-1.5B** como nuevo maestro local en formato `.gaje.flat`.
2. Utilizar este DeepSeek 1.5B como generador del corpus de razonamiento `<think>` para destilar el micro-modelo **SmolLM2-135M-Thinking**.
