# 🗺️ Plan de Implementación: Estabilidad Toroidal y Memoria Recirculante

**Fecha:** 29 de mayo de 2026
**Estatus:** Plan de Acción para el Hito Silver Adult (Q3 2026)
**Referencia Teórica:** `docs/research/TOROIDAL_ENERGY_FLOW_THEORY.md`

Este documento detalla los 4 pilares de optimización para transformar el motor GAJE en un sistema dinámico autoestabilizado con eficiencia energética y cognitiva superior.

---

## 1. Memoria Semántica Recirculante (Closed-Loop DNI)
**Objetivo:** Pasar de una inferencia lineal a un ciclo de destilación continua de conocimiento.

### Acciones Técnicas:
- **Pipeline de Re-Compresión:** Implementar un proceso de fondo que tome los *logs* de las sesiones de chat, extraiga los embeddings más relevantes y los inyecte mediante **Direct Neural Ingestion (DNI)** en los pesos genómicos.
- **Diferenciación Semántica:** El sistema debe identificar qué información es redundante y qué conceptos son "estables" para evitar la sobre-escritura innecesaria del ADN digital.
- **Métrica:** Reducción del tamaño de la base de conocimientos externa en un 90% mediante la integración directa en el modelo.

## 2. Estabilidad Dinámica (Anchored Damping)
**Objetivo:** Utilizar las anclas de precisión para amortiguar el ruido de la cuantización agresiva de 2 bits.

### Acciones Técnicas:
- **Feedback de Anclas:** Configurar las **Stability Anchors (F16)** no solo como puntos de referencia estáticos, sino como reguladores activos que actúen como un `LayerNorm` biológico.
- **Control de Entropía:** Si el sistema detecta que la entropía de fase supera un umbral crítico (caos semántico), las anclas deben ejercer una fuerza de atracción mayor para normalizar la salida.
- **Implementación:** Refinar los kernels en `src/nn/linear.rs` para incluir funciones de amortiguación (damping) basadas en la proximidad a las anclas.

## 3. Evolución Cerrada (Guided Monte Carlo Engine)
**Objetivo:** Transformar el motor genético en un proceso de aprendizaje que recuerde mutaciones exitosas.

### Acciones Técnicas:
- **Genealogy Cache:** Crear un registro ligero de las mutaciones XOR que históricamente han mejorado el Fitness del modelo en tareas de lógica.
- **Mutación Guiada:** El motor Monte Carlo priorizará las "rutas genéticas" conocidas por su estabilidad, reduciendo el tiempo de convergencia en el entrenamiento de bordes (on-device training).
- **Auto-Reparación:** Introducir un bucle de predicción-verificación-corrección. Si una respuesta generada contradice la memoria profunda, el sistema induce una recuperación de contexto adicional antes de finalizar el token.

## 4. Flujo de Memoria Multicapa (Hierarchical Context)
**Objetivo:** Organizar la información en capas según su persistencia y relevancia temporal, emulando la memoria humana.

### Estructura de Capas:
1.  **Capa Local (Instantánea):** Contexto de los últimos 2-5 minutos. Reside en la memoria de trabajo volátil (K-V Cache).
2.  **Capa de Sesión (Media):** Contexto de las últimas horas. Gestionada mediante una base de datos vectorial ligera en memoria RAM.
3.  **Capa Profunda (Genómica/DNA):** Conocimiento de semanas o meses. Integrada permanentemente en los pesos de 2 bits del archivo `.gaje`.

---

## 🚀 Roadmap de Ejecución

1.  **Fase 1 (Corto Plazo):** Implementación de la Capa de Sesión y el buffer de recirculación semántica.
2.  **Fase 2 (Medio Plazo):** Refinamiento de los kernels de amortiguación (Stability Anchors) para control de entropía.
3.  **Fase 3 (Largo Plazo):** Automatización total del motor DNI para la auto-reparación y evolución guiada por memoria histórica.

---
*Este plan es vinculante para el desarrollo del ecosistema GAJE-Flow v1.x.*
