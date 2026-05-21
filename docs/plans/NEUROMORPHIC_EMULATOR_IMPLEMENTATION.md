# 🗺️ Plan Maestro: Emulador de Spiking Transformer (GAJE-Flow)

Este documento detalla la hoja de ruta para la implementación del motor neuromórfico nativo en Rust, optimizado para pesos de 2-bits y contextos masivos.

## Fase 1: Fundamentos Atómicos y Tipos de Datos (Semana 1)
**Objetivo:** Establecer la infraestructura de memoria para los pesos GAJE de 2-bits.

- [ ] **Módulo `src/core/types.rs`**: Implementar el enum `GajeWeight2Bit` con soporte para empaquetado de bits (4 pesos por `u8`).
- [ ] **Módulo `src/nn/spiking/neuron.rs`**: 
    - Implementación de la estructura `SpikingNeuron` (LIF).
    - Métodos `integrate` (suma directa de centroides) y `check_spike`.
- [ ] **Unit Tests**: Validar que la integración de 4 spikes produce un disparo correcto según el umbral.

## Fase 2: Motor de Eventos y Programador (Semana 1-2)
**Objetivo:** Implementar la cola de prioridad asíncrona para evitar el procesamiento denso.

- [ ] **Módulo `src/compute/event_queue.rs`**:
    - Estructura `SpikeEvent` (timestamp, source_neuron_id, target_layer_id).
    - Implementación de `std::collections::BinaryHeap` para la gestión de eventos en el tiempo.
- [ ] **Módulo `src/compute/scheduler.rs`**:
    - Lógica de propagación de retardos (`Δt`).
    - Manejo de ventanas de tiempo para simulación en tiempo real vs. tiempo acelerado.

## Fase 3: Capas Neuromórficas (Atención y FFN) (Semana 2)
**Objetivo:** Traducir la arquitectura Transformer a disparos discretos.

- [ ] **Módulo `src/nn/spiking/attention.rs`**:
    - Implementación de "Spiking Self-Attention". 
    - Los scores de atención se convierten en probabilidades de disparo o frecuencias de spike.
- [ ] **Módulo `src/nn/spiking/ffn.rs`**:
    - Capas lineales de disparos rápidos usando los pesos comprimidos de 2-bits.
- [ ] **Módulo `src/nn/spiking/block.rs`**: Ensamblaje del bloque Transformer neuromórfico.

## Fase 4: Motor Evolutivo Bitwise (Semana 3)
**Objetivo:** Acelerar el entrenamiento/ajuste mediante evolución en Rust.

- [ ] **Módulo `src/core/evolution.rs`**:
    - Función de Fitness basada en `Spike Frequency Accuracy` (SFA).
    - Operadores de mutación bitwise directos sobre el buffer de pesos (AND, OR, XOR para cambiar estados de 2-bits).
- [ ] **Paralelismo**: Integración con `Rayon` para evaluar múltiples "organismos" (variantes del modelo) en paralelo.

## Fase 5: Integración y Benchmarking (Semana 4)
**Objetivo:** Conectar el cargador de modelos y medir rendimiento real.

- [ ] **Módulo `src/io/loader.rs`**: Adaptar el cargador GGUF para inicializar `SpikingNeuron` con pesos de 2-bits.
- [ ] **Benchmark Suite**: 
    - Comparativa de consumo de CPU/Energía vs. Inferencia densa tradicional.
    - Test de contexto de 1M de tokens (RoPE 1,000,000.0).

---

## Métricas de Éxito
1. **Velocidad**: >10x más rápido que la inferencia flotante en tareas de contexto largo.
2. **Eficiencia**: <20% de uso de CPU en estados de baja densidad de información (redundancia).
3. **Memoria**: Mantener el límite de 2-bits por peso sin descompresión a f32.
