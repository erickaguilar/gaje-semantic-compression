# 🗺️ Hoja de Ruta: Desafíos Críticos e Industrialización (v0.9.5+)

Este documento detalla las carencias actuales del proyecto y los pilares necesarios para transformar el motor GAJE de un entorno de investigación (Termux) a un producto de IA de borde (Edge AI) de grado comercial.

---

## 1. Genomic Training Nativo (El Bucle de Retroalimentación)
Aunque contamos con el `SpikingEvolutionEngine` para mutaciones bitwise, el proyecto todavía depende de modelos pre-entrenados (GGUF) para su estructura base.

*   **El Problema:** Falta un mecanismo de **Backpropagation Genómico Híbrido** más robusto.
*   **La Solución:** Evolucionar el actual `refine_centroids` hacia un sistema de aprendizaje continuo (**Life-long Learning**). El organismo debe ser capaz de aprender de un flujo constante de datos del usuario sin sufrir **olvido catastrófico** (catastrophic forgetting), permitiendo que el modelo "crezca" orgánicamente en el dispositivo.

## 2. Abstracción de Hardware: JNI/FFI (El "Mundo Real")
El motor actual es extremadamente rápido en Termux, pero permanece "atrapado" en la interfaz de terminal.

*   **El Problema:** No existe una forma sencilla de integrar el motor en aplicaciones móviles nativas.
*   **La Solución:** Implementar una capa de **Bindings JNI (Java Native Interface)** pulida. El objetivo es que un desarrollador de Android pueda importar GAJE como una librería dinámica (`.so`) y llamar a funciones como `gaje.generate("Hola")` desde Kotlin sin necesidad de conocer los intrínsecos de SIMD NEON o la gestión de memoria en Rust.

## 3. Gestión de Energía Consciente (Thermal & Power Awareness)
En dispositivos móviles, el uso sostenido de todos los núcleos al 100% provoca sobrecalentamiento y el cierre forzoso por parte del sistema operativo.

*   **El Problema:** El scheduler actual no distingue entre tipos de núcleos ni estados de batería.
*   **La Solución:** Crear un **Scheduler consciente de la arquitectura big.LITTLE**. El motor debe ser capaz de:
    *   Mover tareas de fondo (como la indexación de memoria semántica) a los núcleos de eficiencia (**LITTLE**).
    *   Reservar los núcleos de alto rendimiento (**big**) exclusivamente para la interacción activa (chat en tiempo real).

## 4. Memoria de Largo Plazo (Semantic RAG Nativo)
Contamos con compresión de 2 bits para los pesos, pero la memoria contextual de la conversación sigue siendo efímera.

*   **El Problema:** La persistencia de experiencias pasadas es limitada.
*   **La Solución:** Implementar el **Plan de RAG Nativo** (ver `docs/plans/NATIVE_SEMANTIC_RAG_PLAN.md`). Esto incluye integrar `redb` directamente con el motor de spikes, permitiendo que cada pensamiento se guarde en ADN de 2 bits para una recuperación ultra-eficiente mediante ADC (Asymmetric Distance Computation).

## 5. Dashboard de Resonancia (Salud Genómica)
A medida que el modelo muta y se adapta localmente, es vital monitorizar su integridad.

*   **El Problema:** Es difícil diagnosticar si el modelo está "mejorando" o "degradándose" tras las mutaciones.
*   **La Solución:** Desarrollar un **Dashboard de Resonancia** (una expansión del actual `server.py`). Esta herramienta debe visualizar:
    *   La **entropía** de la red neuronal.
    *   La **salud de los centroides** (fidelidad de la señal).
    *   Mapas de calor que indiquen qué áreas del "ADN" del modelo están cambiando más rápido.

---

## Conclusión
La resolución de estos cinco puntos marcará la transición de GAJE de un protocolo experimental a un **Edge AI SDK** soberano, eficiente y capaz de operar de forma autónoma en el bolsillo del usuario.
