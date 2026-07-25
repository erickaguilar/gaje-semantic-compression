# 🏭 Fase 6: Industrialización de la Arquitectura GAJE (Motor Neuromórfico)

**Estado:** Planificado (A partir de v0.8.0)
**Objetivo:** Transformar el prototipo de Spiking Transformer en un motor de inferencia de grado industrial, maximizando la eficiencia de la CPU (SIMD) y eliminando los cuellos de botella algorítmicos en simulaciones de contexto masivo (1M+ tokens).

---

## 1. Diseño Orientado a Datos: Structure of Arrays (SoA)
El diseño inicial basado en una estructura por neurona (`SpikingNeuron`) genera una dispersión de datos en la memoria RAM, lo que provoca constantes fallos de caché (cache misses) cuando la CPU intenta procesar miles de neuronas.

### La Mejora (Optimización SIMD)
Implementar una arquitectura **AoS a SoA**. En lugar de vectores de objetos complejos, los datos se separan en vectores planos contiguos. Esto permite que la CPU use instrucciones SIMD (AVX2/NEON) para procesar múltiples potenciales de membrana en un solo ciclo de reloj.

```rust
// El cerebro de tu emulador: Memoria plana y contigua
pub struct GajeNeuromorphicLayer {
    // Potenciales alineados para caché L1/L2
    pub membrane_potentials: Vec<f32>,
    pub thresholds: Vec<f32>,
    pub decays: Vec<f32>,

    // Almacenamiento masivo de pesos de 2-bits empaquetados
    // ¡4 pesos por cada byte (u8)! Compresión extrema en RAM.
    pub packed_weights: Vec<u8>,
}

impl GajeNeuromorphicLayer {
    pub fn integrate_batch(&mut self, active_spike_index: usize, centroides: &[f32; 4]) {
        for (i, potential) in self.membrane_potentials.iter_mut().enumerate() {
            let byte_index = (i * active_spike_index) / 4;
            let bit_shift = ((i * active_spike_index) % 4) * 2;

            let weight_bits = (self.packed_weights[byte_index] >> bit_shift) & 0b11;
            *potential += centroides[weight_bits as usize];
        }
    }
}
```

## 2. Reemplazar la Cola de Prioridad por un "Timing Wheel"
Para gestionar contextos masivos (ej. 1,000,000 de tokens), una cola de prioridad basada en `BinaryHeap` se vuelve prohibitiva, ya que cada inserción cuesta $O(\log N)$.

### La Mejora (Soporte Real para 1M+ Contexto)
Implementar un algoritmo de **Timing Wheel** (Rueda de Tiempo), el estándar en kernels de sistemas operativos. Se trata de un buffer circular indexado directamente por el paso de simulación (tick) actual.
- **Costo de inserción:** $O(1)$ constante.
- **Costo de ejecución:** $O(1)$. La CPU avanza al siguiente índice y procesa los eventos secuencialmente sin reordenamiento costoso.

## 3. Paralelización basada en Actores (Rayon / Tokio MPSC)
El entrenamiento paralelo (Path Integral Breeding) o la inferencia de múltiples linajes no debe depender de bloqueos de memoria (Mutexes) que estancan los hilos.

### La Mejora (Escalabilidad de Núcleos)
Utilizar la librería `rayon` para paralelizar el procesamiento de las capas sobre múltiples núcleos de manera determinista, o emplear canales `tokio::sync::mpsc` para abstraer los bloques de la red como actores independientes.

```rust
use rayon::prelude::*;

pub fn evaluate_lineages(lineages: &mut Vec<GajeNeuromorphicLayer>, centroides: &[f32; 4]) {
    // Distribuye la carga en todos los núcleos automáticamente
    // Sin riesgo de Data Races gracias a las garantías de Rust.
    lineages.par_iter_mut().for_each(|layer| {
        layer.integrate_batch(0, centroides);
    });
}
```

---
*Este documento establece los estándares de ingeniería para la próxima iteración principal del núcleo neuromórfico (v0.9.0).*
