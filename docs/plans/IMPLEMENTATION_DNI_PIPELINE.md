# 🧬 Plan de Implementación: Direct Neural Ingestion (DNI) Pipeline

**Fecha:** 29 de mayo de 2026
**Estatus:** Plan de Acción para el Hito Silver Adult (Q3 2026)
**Referencia Teórica:** `docs/research/EVOLUTIONARY_MEMORY_AND_DNI.md`

Este documento detalla la hoja de ruta para implementar la funcionalidad de "Ingestión Neuronal Directa", permitiendo que el conocimiento fluya desde la memoria de sesión hacia el genoma digital de 2 bits de forma permanente.

---

## 1. El Motor DNI (Core Rust)
**Objetivo:** Crear un motor de mutación bitwise ultrarrápido diseñado para la inyección granular de información.

### Acciones Técnicas:
- **Módulo `src/core/dni.rs`:** Implementar el `DNIEngine`, una versión simplificada del `IslandModelEngine` optimizada para procesar un solo "cromosoma" (dato nuevo) contra el modelo cargado.
- **Targeted Mutation:** Permitir la mutación selectiva de bloques específicos. Por defecto, se evitarán los bloques 0-2 (Embeddings/Sintaxis base) y los bloques finales (Output logic), enfocando la ingesta en los bloques intermedios (Semántica).
- **Paralelismo:** Uso de `Rayon` para evaluar múltiples mutaciones en paralelo directamente en la CPU del dispositivo móvil.

## 2. Heurísticas de Selección de Capas
**Objetivo:** Minimizar la interferencia catastrófica (olvido) mediante la elección inteligente de los pesos a modificar.

### Acciones Técnicas:
- **Phase Entropy Map:** Implementar una función que analice la dispersión de fase ($\mathbb{Q}(\zeta_{16})$) en cada capa. Las capas con menor entropía (más "espacio de fase libre") serán priorizadas para recibir nueva información.
- **Damping Control:** Durante la ingesta, las *Stability Anchors* de las capas seleccionadas aumentarán su "fuerza de atracción" para asegurar que los nuevos bits no se alejen de la estructura lógica fundamental.

## 3. Pipeline de Procesamiento de Datos
**Objetivo:** Convertir archivos de texto o sesiones binarias en un formato apto para la inyección.

### Acciones Técnicas:
- **Cromosomización:** Fragmentar los documentos de entrada en bloques coherentes de 128-256 tokens.
- **Interfaz CLI (`--dni-ingest`):**
    - Soportar entrada de archivos `.txt` planos.
    - Soportar entrada de sesiones binarias `.bin` generadas por la *Capa de Sesión*.
    - Parámetro `--intensity`: Controla el ratio de mutación (agresivo vs. conservador).

## 4. Validación y Cierre de Bucle
**Objetivo:** Asegurar que la ingesta ha sido exitosa sin degradar el modelo.

### Acciones Técnicas:
- **Validation Loop:** Tras cada paso de DNI, el sistema debe autoevaluarse generando una respuesta basada en el dato inyectado.
- **Forgetfulness Metrics:** Ejecutar un benchmark de "Resonancia de Identidad". Si el fitness del conocimiento previo cae por debajo de un umbral (ej. 90%), el motor debe ajustar automáticamente la tasa de aprendizaje evolutivo o detener la ingesta.

---

## 🚀 Roadmap de Ejecución

1.  **Fase 1 (Corto Plazo):** Creación del motor `src/core/dni.rs` y bindings iniciales.
2.  **Fase 2 (Medio Plazo):** Implementación del comando `--dni-ingest` en `gaje-cli` para archivos de texto.
3.  **Fase 3 (Largo Plazo):** Integración total con la *Capa de Sesión* para un ciclo de "Chat -> Ingesta Automática -> Evolución".

---
*GAJE-Flow: Donde la información se convierte en instinto.*
