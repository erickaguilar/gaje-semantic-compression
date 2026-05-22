# 🚀 Plan de Implementación: Motor Temporal de Fase (Phase-GAJE)

Este plan detalla la transición del motor de disparos simples a un sistema de **Codificación Temporal y Potenciales Graduados**.

## Fase 1: Actualización de Infraestructura (Estructura de Datos)
*   **Modificar `SpikeEvent`:** Añadir campos `intensity: f32` y `phase_offset: u8` (0-15 para sub-ticks).
*   **Extender `TimingWheel`:** Implementar soporte para micro-pasos dentro de cada slot.
*   **Refactorizar `NeuromorphicScheduler`:** Ajustar la inyección de spikes para que acepte intensidad y fase.

## Fase 2: Dinámica Neuronal Graduada (Layer Logic)
*   **Actualizar `GajeNeuromorphicLayer::check_spikes`:**
    *   Calcular `residual = current_potential - threshold`.
    *   Generar spikes con intensidad proporcional al residuo.
*   **Optimizar `integrate_batch` (NEON):**
    *   Modificar el kernel para que el incremento del centroide sea multiplicado por la intensidad del spike entrante.
    *   *Nota:* Esto introduce una multiplicación, pero solo en el momento del spike, no en toda la matriz.

## Fase 3: Codificación de Fase (Phase/Latency Coding)
*   **Implementar Mapeo de Tiempo:** Crear una función que traduzca el potencial de membrana a un `phase_offset` (latencia).
*   **Validación de Latencia:** Asegurar que las neuronas con mayor energía se procesen primero en la Timing Wheel.

## Fase 4: Inhibición Lateral e Inteligencia K-WTA
*   **Módulo de Inhibición:** Crear un mecanismo en `SpikingAttention` que "anule" eventos futuros en la Timing Wheel si una neurona de alta confianza ya ha disparado para esa posición.
*   **Sincronización:** Implementar ventanas de inhibición temporal para emular el comportamiento del Softmax.

## Fase 5: Validación y Calibración
*   **Benchmark de Perplejidad:** Comparar la coherencia del modelo con y sin Codificación de Fase.
*   **Prueba de Resonancia v2:** Verificar si el `Identity Cloner` converge más rápido con neuronas graduadas.

---
**Prioridad Actual:** Fase 1 y 2 (Estructuras y Intensidad).
