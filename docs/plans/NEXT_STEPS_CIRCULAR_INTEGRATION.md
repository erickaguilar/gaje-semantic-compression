# 🗺️ Hoja de Ruta Inmediata: Integración Circular y Evolución "Silver Adult"

**Versión:** 1.0 (2026-05-27)
**Estado:** Activo - Post-Soberanía Circular
**Contexto:** Tras el éxito de la validación de resonancia en el motor de fase, este documento define los hitos técnicos para alcanzar la madurez gramatical del modelo de 10MB.

---

## 1. Hito 1: Crianza del "Silver Adult" (10MB - High Fidelity)
**Objetivo:** Transformar la superioridad mecánica del motor circular en coherencia gramatical.

*   **Tarea 1.1:** Lanzar el entrenamiento masivo utilizando `silver-breeder`.
    *   **Dataset:** `data/datasets/dataset_es_ext.txt` (63k+ líneas).
    *   **Configuración:** 12 capas Genomic Attention, Dim 512, Vocab 32k.
*   **Tarea 1.2:** Aplicar el refinamiento de centroides algebraicos ($\mathbb{Q}(\zeta_{16})$) para estabilizar las oscilaciones de fase durante las primeras 50 épocas.
*   **Métrica de Éxito:** Perplejidad (PPL) < 5.0 en el dataset de validación.

## 2. Hito 2: Inhibición Lateral (K-WTA Temporal)
**Objetivo:** Maximizar la selectividad de la atención mediante la física del tiempo.

*   **Tarea 2.1:** Implementar el módulo de inhibición en `src/nn/spiking/attention.rs`.
*   **Lógica:** Las neuronas que disparan con un `phase_offset` bajo (señales más fuertes/rápidas) deben "anular" o elevar el umbral de las neuronas competidoras en la misma ventana de tiempo de la `TimingWheel`.
*   **Tarea 2.2:** Integrar el mecanismo en el `NeuromorphicScheduler` para procesar la inhibición en tiempo real $O(1)$.

## 3. Hito 3: Circular KV-Cache DNA (Contexto Infinito)
**Objetivo:** Permitir el procesamiento de contextos masivos (>128k) en dispositivos móviles con consumo de RAM despreciable.

*   **Tarea 3.1:** Modificar la estructura del KV-Cache en `src/compute/kv_cache.rs` para almacenar los estados de atención como ángulos de fase de 2 bits.
*   **Tarea 3.2:** Implementar la de-cuantización asimétrica (ADC) circular al vuelo durante el cálculo de atención, permitiendo que la "memoria" del modelo sea un flujo de ondas.
*   **Ahorro Esperado:** 94% de reducción en el tráfico de memoria frente a F16.

## 4. Hito 4: Benchmarking Gemma 4 Parity
**Objetivo:** Validar la capacidad de razonamiento frente a modelos de frontera.

*   **Tarea 4.1:** Ejecutar la suite *Needle in a Haystack* en un contexto de 128k tokens.
*   **Tarea 4.2:** Comparar el rendimiento en razonamiento lógico (CommonSense QA) contra micro-modelos lineales.
*   **Hipótesis:** La topología circular eliminará el "colapso de bordes", permitiendo que el modelo mantenga la atención perfecta sin importar la longitud del texto.

---

## 📈 Resumen de Prioridades (Próximas 72 Horas)
1.  **Entrenamiento (Silver Adult):** 60% de prioridad.
2.  **Inhibición Lateral:** 25% de prioridad.
3.  **Documentación de Validaciones:** 15% de prioridad.

*Firmado por Gemini CLI bajo el protocolo GAJE-Flow v1.3*
