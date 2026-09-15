# 📊 Evaluación Empírica de CoT + .gmem: Ablación de Variables (Canal vs. Andamiaje)

> **Clasificación:** Reporte de Investigación Empírica (`docs/research/`)  
> **Fecha:** 15 de septiembre de 2026  
> **Modelo Evaluado:** `Qwen2.5-0.5B-Instruct` cuantizado en [`models/production/qwen2_5_0_5b.gaje`](file:///data/data/com.termux/files/home/develop/gaje-semantic-compression/models/production/qwen2_5_0_5b.gaje) (1.5 GB, 24 capas Q4_0, GTOK 151,936 tokens)  
> **Entorno de Ejecución:** ARM64 (Linux/Termux), inferencia nativa en Rust (`gaje-core`), greedy decoding `temp = 0.0`, `repetition_penalty = 1.15`.  
> **Harness de Prueba:** [`tests/test_cot_gmem_baseline.rs`](file:///data/data/com.termux/files/home/develop/gaje-semantic-compression/tests/test_cot_gmem_baseline.rs)

---

## 1. Matriz Exhaustiva de Ablación (8 Condiciones)

Para aislar si el rendimiento del andamiaje se debía a la estructura neurosimbólica de CoT (`[THINKING]...[ANSWER]`) o a la variable de confusión del **canal de inyección** y la **supresión de preámbulo**, se evaluaron 8 condiciones sobre los 20 prompts canónicos:

| ID | Canal | Formato Sintáctico | Aciertos | Exactitud (%) | Hipótesis que Evalúa / Resultado |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **A** | — | Pregunta directa | 6 / 20 | **30.0%** | **Baseline Directo**: Memoria paramétrica no guiada. |
| **B** | — | *"think step by step"* (libre) | 1 / 20 | **5.0%** | **Colapso CoT**: Dilución procedimental vacía. |
| **D** | `system` | `Knowledge: {fact}` (plano) | 8 / 20 | **40.0%** | **RAG Clásico**: Vacilación y preámbulos (*"To answer..."*). |
| **C5** | `system` | `[THINKING][QUERY]->fact[ANSWER]` | 11 / 20 | **55.0%** | **Formato en System**: Reduce preámbulos pero satura contexto. |
| **C2** | `assistant`| `[QUERY: X] -> fact\n[ANSWER] ` | 10 / 20 | **50.0%** | **Tags sin Thinking**: Los subtokens de corchetes restan precisión. |
| **C1** | `assistant`| `[THINKING][QUERY]->fact\n[ANSWER]`| 11 / 20 | **55.0%** | **Andamiaje CoT Completo**: Efecto aparente original. |
| **C3** | `assistant`| `Fact: {fact}\nAnswer: ` | 12 / 20 | **60.0%** | **Estructura Limpia**: Sin tags especiales, lenguaje natural. |
| **C4** | `assistant`| **`{fact}\nAnswer: ` (Plano)** | **13 / 20** | **`65.0%`** | **GANADOR ABSOLUTO**: Inyección en canal asistente + prefijo directo. |

---

## 2. Descomposición Ortogonal de Variables

### A. Efecto de la Etiqueta `[THINKING]` ($C_1 - C_2 = +5.0\%$)
* Con `[THINKING]`: 11 / 20 (55.0%).
* Sin `[THINKING]`: 10 / 20 (50.0%).
* **Diagnóstico**: La palabra `[THINKING]` no induce razonamiento real; el delta de 1 prompt es ruido de tokenización dentro del margen de varianza.

### B. Efecto de los Tags Especiales vs. Formato Genérico ($C_3 - C_2 = +10.0\%$)
* Con tags de corchetes `[QUERY: ...][ANSWER]`: 10 / 20 (50.0%).
* Con palabras clave en inglés estándar `Fact: ... Answer: `: 12 / 20 (60.0%).
* **Diagnóstico**: Los corchetes y subtokens fragmentados (`[`, `QUERY`, `:`, `]`) **perjudican al modelo**. El modelo Instruct de 0.5B responde mucho mejor a secuencias naturales de lenguaje (`Fact: ... Answer: `).

### C. Efecto de la Estructura vs. Plano ($C_4 - C_3 = +5.0\%$ y $C_4 - C_1 = +10.0\%$)
* Andamiaje CoT ($C_1$): 11 / 20 (55.0%).
* Formato Plano Directo ($C_4$): **13 / 20 (65.0%)**.
* **Diagnóstico**: **$C_4 > C_1$**. El andamiaje pseudo-CoT no solo no aporta ningún beneficio adicional sobre el formato plano en el turno del asistente, sino que **reduce la precisión en 10 puntos porcentuales**. La parafernalia sintáctica de "simular pensamiento" distrae la atención local de la red.

### D. Efecto del Canal de Inyección ($C_4 - D = +25.0\%$)
* Canal `system` ($D$): 8 / 20 (40.0%).
* Canal `assistant` ($C_4$): **13 / 20 (65.0%)**.
* **Diagnóstico**: **El factor determinante del rendimiento es el canal de inyección y el prefijo de asistente.**
  1. Cuando el hecho se coloca en el `system prompt`, el modelo inicia su turno de asistente en frío, generando preámbulos vacíos (*"To answer the question 'In what year...' I will use my knowledge..."*) que consumen los tokens antes de llegar al número.
  2. Cuando el hecho se inyecta en el turno del asistente seguido inmediatamente de `Answer: `, la distancia atencional entre el hecho y la generación es de 1 token, forzando la extracción directa del dato fáctico.

---

## 3. Veredicto Arquitectónico para GAJE Helix

1. **Cierre Formal de la Línea Pseudo-CoT:**  
   Queda descartada la necesidad de implementar parsers de razonamiento o envolturas `[THINKING]` en modelos sub-1B.
2. **Regla de Producto y Diseño Óptimo:**  
   La arquitectura de integración de la memoria hipocampal (`.gmem`) debe operar mediante **inyección en el turno del asistente con prefijo forzado**:
   ```text
   <|im_start|>user
   {Pregunta del usuario}<|im_end|>
   <|im_start|>assistant
   {Hecho recuperado de .gmem}
   Answer: 
   ```
   Esta estrategia es más rápida, no requiere tokens adicionales de andamiaje, elimina parsers complejos y maximiza la exactitud en edge devices.
