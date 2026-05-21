# 🧬 Plan de Destilación Neuromórfica: SmolLM (f16) → GAJE Spiking (2-bit)

Este documento define el protocolo para transferir el conocimiento de un modelo denso de alta precisión (SmolLM f16) hacia el motor de emulación neuromórfica de 2-bits.

## 1. Objetivo Estratégico
Lograr la "congelación" del conocimiento de SmolLM en un formato neuromórfico asíncrono, reduciendo el consumo energético en 10x y la memoria en 8x, manteniendo la coherencia semántica mediante resonancia de disparos (spikes).

## 2. Arquitectura de Destilación

### Fase A: El Maestro (Dense f16)
*   **Modelo:** SmolLM (135M o 360M) cargado en precisión f16.
*   **Función:** Actuar como el generador de "Patrones de Resonancia". Para cada input, el maestro genera activaciones densas que servirán como el objetivo (target) de la evolución.

### Fase B: El Estudiante (Spiking 2-bit)
*   **Modelo:** Spiking Transformer Block inicializado aleatoriamente.
*   **Cuantización:** 2-bits por peso empaquetados.
*   **Mecanismo:** Neuronas LIF que deben aprender a disparar en sincronía con las magnitudes de activación del maestro.

## 3. El Proceso de Resonancia Genómica

1.  **Inyección de Datos:** Pasar un dataset de alta calidad (ej. `dataset_es.txt`) por SmolLM f16.
2.  **Captura de Activaciones:** Extraer los vectores de activación de las capas clave (Atención y FFN).
3.  **Mapeo de Fitness (SRL - Spiking Resonance Loss):**
    *   Si una activación f16 es alta -> El fitness del estudiante mejora si dispara un spike.
    *   Si una activación f16 es baja -> El fitness del estudiante mejora si la neurona permanece en silencio.
4.  **Evolución Bitwise:** El `SpikingEvolutionEngine` corre mutaciones XOR masivas en paralelo para que el modelo de 2-bits replique el mapa de actividad eléctrica del maestro.

## 4. Implementación Técnica

- [ ] **Módulo `src/nn/spiking/distiller.rs`**: Controlador que gestiona el flujo entre el `NativeLoader` (maestro) y el `NeuromorphicScheduler` (estudiante).
- [ ] **Ajuste de Umbrales Dinámicos:** Los umbrales (Thresholds) del estudiante se calibran según la media de activación del maestro por capa.
- [ ] **Exportación `.gaje`**: Guardado del resultado en un archivo genómico listo para su despliegue en Edge devices.

## 5. Métricas de Éxito
*   **Preservación de Señal:** >90% de correlación entre los disparos del estudiante y las activaciones del maestro.
*   **Eficiencia:** Procesamiento de destilación a una tasa de >500 tokens/segundo en el emulador.
*   **Integridad:** Cero descompresión a f32 durante la ejecución del estudiante.

---
*Este plan establece la base para la creación de micro-modelos neuromórficos derivados de arquitecturas comerciales.*
