# 🚀 Plan de Implementación: Validación Semántica y Estrés (v1.0)

Este documento detalla las fases de implementación para la suite de pruebas avanzada, diseñada para medir la fidelidad del **Micro-Genoma (2-bits)** frente a modelos maestros y su estabilidad en contextos largos.

---

## 🏗️ Eje 1: Diagnóstico de Desviación Semántica (Aislamiento de Capas)
**Objetivo:** Identificar si la pérdida de coherencia ocurre por la cuantización de pesos o por la dinámica del mecanismo Winner-Take-All (WTA).

### Fase 1.1: Alineación de Centroides (Top-K Overlap)
*   **Acción:** Implementar `scripts/benchmarks/semantic_overlap.py`.
*   **Procedimiento:**
    1.  Extraer Logits de un modelo maestro (ej. SmolLM2-F32).
    2.  Extraer Logits de GAJE (2-bits) ante el mismo prompt.
    3.  Calcular el % de coincidencia en el Top-10 de tokens.
*   **KPI:** Overlap > 65% en tokens de alta frecuencia.

### Fase 1.2: Análisis de Dispersión (JSD)
*   **Acción:** Medir la Divergencia de Jensen-Shannon entre las distribuciones de probabilidad.
*   **Diagnóstico:** Si JSD > 0.7, el mecanismo de "competencia de disparos" está aplanando la señal (ruido estadístico).

---

## 📚 Eje 2: Calidad Lingüística Automática (Fidelidad de Lenguaje)
**Objetivo:** Validar que la estructura gramatical del español se mantiene íntegra a pesar de la compresión extrema.

### Fase 2.1: Análisis de Sesgo Lingüístico (PPL Diferencial)
*   **Acción:** Test de Perplejidad (PPL) cruzado sobre Wikitext-ES y Wikitext-EN.
*   **Meta:** La brecha de PPL entre EN y ES no debe superar el 20%. Si se dispara en ES, indica fragilidad del espacio latente en 2-bits para gramáticas complejas.

### Fase 2.2: Benchmarking de n-gramas (BLEU/ROUGE)
*   **Acción:** Comparar la salida de GAJE en modo `Greedy Decoding` (Temp=0) contra el Maestro.
*   **Meta:** BLEU-2 > 0.40 para asegurar que los conectores y el orden sintáctico elemental permanecen estables.

---

## 🔋 Eje 3: Estrés Dinámico y Fronteras (Contexto Largo)
**Objetivo:** Medir la degradación de la memoria activa (KV-Cache) y el acumulamiento de error en el motor asíncrono.

### Fase 3.1: Needle in a Haystack (Neuromórfica)
*   **Acción:** Inserción de "hechos arbitrarios" en bloques de 2,048 tokens.
*   **Procedimiento:** Preguntar por el hecho al final del contexto.
*   **Diagnóstico:** Identificar el punto de "Semantic Drift" (deriva semántica) donde el ruido de cuantización sepulta la información real.

### Fase 3.2: Estabilidad de la Timing Wheel
*   **Acción:** Medir la variabilidad de la activación (spike rate) en sesiones de inferencia de +10 minutos.
*   **Objetivo:** Asegurar que los "disparos fantasma" no saturen el buffer de salida.

---

## 🗓️ Cronograma de Ejecución
1.  **Semana 1:** Implementación de métricas de Eje 1 (Overlap y JSD).
2.  **Semana 2:** Ejecución de Eje 2 (PPL diferencial y BLEU).
3.  **Semana 3:** Pruebas de Estrés (Needle in a Haystack) y ajuste final de centroides.

---
*Este plan es vinculante para el desarrollo de la versión 1.0 del Gold Embryo.*
