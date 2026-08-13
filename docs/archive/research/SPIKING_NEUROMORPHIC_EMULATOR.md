# 🧠 Emulador de Spiking Transformer Basado en Eventos (Neuromórfico)

Este documento define el plano arquitectónico para acoplar la lógica de compresión de 2-bits (GAJE) con un motor de simulación neuromórfica nativo en Rust.

## 1. La Estructura de la Neurona Discreta (LIF) en Rust

En lugar de utilizar tensores densos y operaciones de multiplicación de matrices, la arquitectura implementa neuronas biológicas de tipo Leaky Integrate-and-Fire (LIF). Dado que los pesos de GAJE son de 2-bits, la integración matemática se reduce a **cero multiplicaciones**, utilizando sumas directas de los centroides.

```rust
// Representación de los 4 estados posibles de los centroides de 2-bits
#[derive(Copy, Clone, Debug)]
pub enum GajeWeight2Bit {
    State00 = 0, // Centroide calibrado más bajo
    State01 = 1,
    State10 = 2,
    State11 = 3, // Centroide calibrado más alto
}

// Estructura de una neurona neuromórfica emulada
pub struct SpikingNeuron {
    pub membrane_potential: f32,    // El "voltaje" interno de la neurona
    pub threshold: f32,             // "Anclas FFN" que estabilizan el disparo
    pub decay: f32,                 // Fuga de energía en el tiempo (Leaky)
    pub weights: Vec<GajeWeight2Bit>, // Pesos hiper-comprimidos en memoria continua
}

impl SpikingNeuron {
    // Integrar un impulso eléctrico entrante (Spike)
    pub fn integrate(&mut self, input_index: usize, centroides: &[f32; 4]) {
        // ¡Cero Multiplicaciones!
        // Al llegar un spike (1), sumamos el valor real del centroide de 2-bits.
        let weight_state = self.weights[input_index] as usize;
        self.membrane_potential += centroides[weight_state];
    }

    // Verificar si la neurona debe "disparar" un token o señal
    pub fn check_spike(&mut self) -> bool {
        if self.membrane_potential >= self.threshold {
            self.membrane_potential = 0.0; // Resetear voltaje (período refractario)
            true  // Dispara un Spike (1)
        } else {
            self.membrane_potential *= self.decay; // Disipación de energía
            false // No dispara (0)
        }
    }
}
```

## 2. Procesamiento de Contexto Masivo (RoPE Alto) Basado en Eventos

Para manejar contextos masivos (ej. 1,000,000 de tokens), se abandona la iteración tradicional de matrices. En su lugar, el sistema emplea una **Cola de Prioridad Basada en Eventos** (`std::collections::BinaryHeap`).

*   **Asincronía Neuromórfica:** Si una neurona en la capa de atención se dispara en el milisegundo `T`, calcula a qué neuronas de la capa FFN afectará y encola esos impulsos con un tiempo futuro `T + Δt`.
*   **Eficiencia Extrema:** La CPU (o ARM en dispositivos Edge) solo ejecuta ciclos de reloj cuando ocurren "picos" (spikes) de actividad. Las partes redundantes del contexto son saltadas pasivamente, logrando una velocidad de simulación órdenes de magnitud superior.

## 3. Fitness Evolutivo Acelerado por Memoria Continua

La optimización evolutiva (algoritmos genéticos / Monte Carlo) corre directamente sobre este emulador en Rust.

*   **Población de Hilos Neuromórficos:** Se crean múltiples variantes de la red.
*   **Nueva Función de Loss/Fitness:** Se evalúa qué tan rápido y con qué precisión de frecuencia disparan las neuronas de salida el token correcto, reemplazando la función de pérdida de coma flotante tradicional.
*   **Mutación a Nivel de Bits:** Al representar los estados genómicos en memoria continua (`GajeWeight2Bit`), Rust permite mutaciones directas usando operaciones lógicas ultra-rápidas (Bitwise `AND`/`OR`/`XOR`). Esto reduce el tiempo de procesamiento de cientos de generaciones de horas a minutos.

## 🎯 Impacto y Visión

Este middleware transforma el formato `.gaje` de un simple archivo de compresión a un **plano genómico ejecutable**. Demuestra que estos modelos están listos para operar de forma nativa en la próxima generación de procesadores neuromórficos, superando las limitaciones térmicas y de von Neumann del silicio tradicional.
