# 🧬 Plan de Implementación: Direct Neural Ingestion (DNI) Consolidation

**Versión:** 1.5 (Junio 2026)
**Estatus:** Plan de Acción Unificado para el Hito Silver Adult
**Referencia Teórica:** `docs/research/EVOLUTIONARY_MEMORY_AND_DNI.md`

Este documento unifica la estrategia de **Direct Neural Ingestion (DNI)**, integrando el motor de mutación bitwise en Rust con las herramientas de usuario y las métricas de estabilidad genómica.

---

## 1. El Motor DNI (Core Rust)
**Objetivo:** Evolucionar el motor de mutación de prototipo a componente industrial ultrarrápido.

### Acciones Técnicas:
- **Módulo `src/core/dni.rs`:** Implementar el `DNIEngine`, optimizado para procesar "cromosomas" (datos nuevos) contra el micro-genoma de 2 bits.
- **Mutación Quirúrgica (Targeted Mutation):**
    - Selección inteligente de capas basada en **Phase Entropy Map** (capas con menor entropía priorizadas).
    - Blindaje de **Stability Anchors (F16)**: Los pesos de alta precisión son inmutables para preservar la lógica fundamental.
    - Foco en bloques intermedios (Semántica), evitando bloques de entrada/salida para no degradar la sintaxis base.
- **Paralelismo:** Uso de `Rayon` para evaluar mutaciones en paralelo aprovechando los núcleos ARM.

## 2. Pipeline de Procesamiento y Herramientas
**Objetivo:** Facilitar la inyección de conocimiento desde diversas fuentes de datos.

### Acciones Técnicas:
- **Cromosomización:** Fragmentación de documentos planos (`.txt`) o sesiones binarias (`.bin`) en bloques coherentes de 128-256 tokens.
- **Integración CLI (`gaje-cli ingest`):**
    - Parámetro `--intensity`: Control de tasa de mutación (agresivo vs. conservador).
    - Parámetro `--generations`: Ciclos de evolución para la ingesta.
    - Retroalimentación Visual: Barras de progreso con evolución del *Fitness* en tiempo real.

## 3. Escalabilidad y Procesamiento por Islas
**Objetivo:** Manejar documentos grandes y optimizar el rendimiento.

- **Island Model Integration:** Repartir fragmentos del documento en diferentes "islas" evolutivas en paralelo.
- **Fusión de Mutantes:** Algoritmos de mezcla para integrar el conocimiento de las islas sin introducir ruido destructivo.

## 4. Validación y Métricas Anti-Olvido
**Objetivo:** Garantizar que el nuevo conocimiento no degrade las capacidades previas.

- **Métricas de Deriva Genómica:** Informar sobre el porcentaje de desplazamiento resultante.
- **Validation Loop:** Autoevaluación post-ingesta generando respuestas basadas en el dato inyectado.
- **Forgetfulness Metrics:** Si el fitness del conocimiento previo cae por debajo del 90%, el motor ajusta automáticamente la tasa de aprendizaje o detiene la ingesta.

---

## 🚀 Roadmap de Ejecución
1.  **Corto Plazo:** Finalizar motor `src/core/dni.rs` y bindings iniciales.
2.  **Medio Plazo:** Implementar comando `gaje-cli ingest` completo para archivos de texto.
3.  **Largo Plazo:** Integración total con la *Capa de Sesión* para un ciclo de "Chat -> Ingesta Automática -> Evolución".

---
*GAJE-Flow: Donde la información se convierte en instinto.*
