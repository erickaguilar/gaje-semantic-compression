# 🗺️ Plan de Implementación: Estabilidad Circular y Toroidal

**Versión:** 1.5 (Junio 2026)
**Estatus:** Plan Unificado de Fase y Recirculación
**Referencia Teórica:** `docs/research/TOROIDAL_ENERGY_FLOW_THEORY.md` y `CIRCULAR_TOPOLOGY_THEORY.md`

Este documento unifica los hitos técnicos para alcanzar la madurez del motor de fase, transformando la superioridad mecánica en coherencia gramatical y estabilidad dinámica.

---

## 1. Crianza del "Silver Adult" (10MB - High Fidelity)
**Objetivo:** Alcanzar coherencia gramatical mediante entrenamiento circular masivo.

*   **Entrenamiento Masivo:** Uso de `silver-breeder` con datasets extensos (63k+ líneas).
*   **Refinamiento de Centroides:** Estabilización de oscilaciones de fase en $\mathbb{Q}(\zeta_{16})$ durante las primeras épocas.
*   **Métrica de Éxito:** Perplejidad (PPL) < 5.0.

## 2. Estabilidad Dinámica e Inhibición (K-WTA)
**Objetivo:** Maximizar la selectividad de la atención y amortiguar el ruido de cuantización.

*   **Inhibición Lateral Temporal:** Neuronas con señales más fuertes (bajo `phase_offset`) anulan competidoras en la `TimingWheel`.
*   **Anchored Damping:** Las **Stability Anchors (F16)** actúan como reguladores activos (tipo LayerNorm biológico) para normalizar la salida si la entropía de fase es crítica.
*   **Implementación:** Refinar kernels en `src/nn/linear.rs` con funciones de amortiguación basadas en la proximidad a las anclas.

## 3. Memoria Semántica Recirculante (Closed-Loop DNI)
**Objetivo:** Pasar de inferencia lineal a un ciclo de destilación continua de conocimiento.

*   **KV-Cache Circular (Contexto Infinito):** Estados de atención almacenados como ángulos de fase de 2 bits, permitiendo contextos de >128k con RAM mínima.
*   **Pipeline de Re-Compresión:** Proceso de fondo que extrae embeddings relevantes de los logs de chat e inyecta conocimiento mediante DNI.
*   **Evolución Cerrada (Guided Monte Carlo):** Cache de mutaciones exitosas (Genealogy Cache) para acelerar la convergencia y auto-reparar contradicciones.

## 4. Flujo de Memoria Multicapa
**Objetivo:** Emular la jerarquía de memoria humana.

1.  **Capa Local (Instantánea):** Contexto inmediato en K-V Cache volátil.
2.  **Capa de Sesión (Media):** Gestionada por base de datos vectorial ligera.
3.  **Capa Profunda (Genómica/DNA):** Conocimiento integrado permanentemente en pesos de 2 bits.

---

## 📈 Resumen de Prioridades
1.  **Entrenamiento (Silver Adult):** Foco en reducción de PPL.
2.  **Inhibición Lateral:** Integración en `NeuromorphicScheduler`.
3.  **KV-Cache DNA:** Implementación de de-cuantización asimétrica circular.

---
*Firmado por Gemini CLI bajo el protocolo GAJE-Flow v1.5*
