# 🏗️ SDD: Software Design Document - GAJE-Flow Core

Este documento detalla la arquitectura técnica y las decisiones de diseño del motor de compresión semántica y ejecución neuromórfica.

## 1. Arquitectura de Memoria: SoA (Structure of Arrays)

Para maximizar el throughput SIMD (AVX2/NEON), el sistema utiliza una organización **SoA** en lugar de AoS (Array of Structures).

- **Componente:** `src/nn/spiking/`
- **Diseño:** Los potenciales de membrana, umbrales y estados de las neuronas se almacenan en vectores contiguos paralelos.
- **Beneficio:** Permite que un solo ciclo de CPU procese múltiples neuronas mediante instrucciones vectoriales, eliminando el "cache miss" por dispersión de objetos.

## 2. Formato de Datos: Genoma Digital de 2-Bits

La compresión se basa en la representación de pesos como bases nitrogenadas digitales.

| Base | Representación | Significado Semántico |
| :--- | :---: | :--- |
| **A** (Adenina) | `00` | Inhibición fuerte / Peso mínimo |
| **C** (Citosina) | `01` | Inhibición leve |
| **G** (Guanina) | `10` | Excitación leve |
| **T** (Timina) | `11` | Excitación fuerte / Peso máximo |

## 3. Motor Neuromórfico (Zero-Mult Engine)

El motor de inferencia implementa el modelo **LIF (Leaky Integrate-and-Fire)** simplificado para eliminar multiplicaciones.

- **Proceso:**
    1. Recepción de eventos vía **Timing Wheel** ($O(1)$).
    2. Suma directa de centroides al potencial de membrana.
    3. Disparo (Spike) si se supera el umbral dinámico.
    4. Reset homeostático.

## 4. Interfaz de Puente (Rust-Python)

- **Rust (Core):** Responsable del cálculo intensivo, gestión de memoria y paralelismo masivo.
- **Python (Research):** Utilizado para orquestación, visualización de experimentos y preparación de datasets sintéticos.
