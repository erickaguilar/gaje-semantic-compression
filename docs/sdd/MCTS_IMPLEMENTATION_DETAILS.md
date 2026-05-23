# 🏗️ SDD: Detalles de Implementación MCTS-Genómico

Este documento describe la estructura técnica del motor MCTS para la optimización de pesos genómicos.

## 1. Estructura de Datos (SoA)

Para evitar la fragmentación de memoria y maximizar el rendimiento, el árbol no utilizará punteros (`Box<Node>`). En su lugar, usaremos un `MctsTree` basado en vectores planos.

```rust
pub struct MctsTree {
    // Índices de los padres
    pub parents: Vec<usize>,
    // Hijos: Vector de vectores (o un sistema de offsets para ser más SoA)
    pub children: Vec<Vec<usize>>,
    // Estadísticas PUCT
    pub n_visits: Vec<u32>,
    pub q_values: Vec<f32>,
    pub p_prior: Vec<f32>,
    // Estado del genoma en este nodo (o delta respecto al padre)
    pub genomic_deltas: Vec<GenomicDelta>,
}
```

## 2. El Kernel de Selección (PUCT)

La función de selección buscará el índice `i` que maximice:
`Q[i] + C_PUCT * P[i] * (sqrt(total_n) / (1 + n_visits[i]))`

## 3. Evaluación Neuromórfica

El motor MCTS llamará al `LifEngine` para evaluar la aptitud.
- **Entrada:** Genoma mutado.
- **Salida:** Score de resonancia (basado en la estabilidad de los spikes y el error MSE).

## 4. Paralelismo con Rayon

Utilizaremos `rayon` para realizar múltiples simulaciones en paralelo desde la raíz, combinando los resultados al final de cada iteración.
