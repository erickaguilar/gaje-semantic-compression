# Plan de Arquitectura Struct-of-Arrays (SoA) para Inferencia Híbrida Genómica

## 🎯 Objetivo
Resolver la complejidad en tiempo de ejecución introducida por la Fase 12 (metabolismo dinámico) donde se mezclan precisiones de 2-bit (Base), 4-bit (Epigenético) y 6-bit (Triplete). El objetivo es evitar el *branching* (condicionales `if`) durante los bucles internos de multiplicación matricial en Rust, maximizando así el throughput de las instrucciones SIMD NEON.

## 🧠 El Problema: Array of Structs (AoS) Dinámico
Si se diseña la memoria de forma entrelazada o dinámica:
```rust
// MAL DISEÑO (Lento)
for block in layer {
    if block.is_epigenetic {
        compute_4bit(block); // Rompe el pipeline SIMD
    } else if block.is_triplet {
        compute_6bit(block); // Rompe el pipeline SIMD
    } else {
        compute_2bit(block);
    }
}
```
Esto causaría un severo *branch prediction penalty* en el hardware y haría casi imposible vectorizar la carga de memoria.

## 🛠️ La Solución: Perfilado Estático y Struct of Arrays (SoA)

### 1. Perfilado Estático en Python (Export Time)
La decisión de qué partes del modelo reciben 4-bit o 6-bit NO se toma en tiempo real. 
El `SignalToNoiseBalancer` de Python analiza el modelo *una vez* y define de forma permanente la "Máscara Metabólica". Python exporta tres buffers distintos para cada capa:

### 2. Disposición de Memoria SoA en Rust
En lugar de mezclar precisiones, cada capa en Rust se compone de tres vectores planos y contiguos:

```rust
// DISEÑO OPTIMIZADO (Rápido)
pub struct SoAGenomicLayer {
    // El 95%+ de los datos: Matriz Densa 2-bit
    pub base_strands: Vec<u8>,       
    pub base_centroids: Vec<f32>,

    // El ~4% de los datos: Sparse Anchors 4-bit
    // Formato comprimido (ej. CSR) o denso parcial
    pub epi_strands: Vec<u8>,        
    pub epi_indices: Vec<u32>,       // Para sumar al resultado base

    // El <1% de los datos: Super Anchors 6-bit
    pub tri_strands: Vec<u8>,        
    pub tri_indices: Vec<u32>,       
}
```

### 3. Pipeline de Ejecución SIMD (Inference Time)
Durante el *forward pass*, las operaciones se realizan por fases secuenciales sin condicionales internos:

1.  **Fase 1: Multiplicación Base (SIMD Pura)**
    *   Ejecutar la multiplicación de matrices (MatMul) sobre el `base_strands` (2-bit) para **todos** los elementos.
    *   Este bucle es ciego, lineal y utiliza el 100% de la capacidad vectorial NEON.
    *   *Resultado:* `base_out` (Vector f32 preliminar).

2.  **Fase 2: Corrección Epigenética (Scatter-Add)**
    *   Iterar sobre `epi_strands` (4-bit) usando `epi_indices`.
    *   Desempaquetar el bloque 4-bit a f32, multiplicarlo por su entrada correspondiente, y **sumar** el resultado al índice específico en `base_out`.
    *   *Nota:* Como son muy pocos datos (~4%), esta fase es extremadamente rápida y compensa los errores de la Fase 1 en ubicaciones clave.

3.  **Fase 3: Refinamiento de Tripletes (Scatter-Add)**
    *   Idéntico a la Fase 2, pero para `tri_strands` (6-bit) usando `tri_indices`.
    *   Esto corrige los Outliers críticos que destrozarían la activación SwiGLU.

## 📊 Beneficios de la Arquitectura SoA
1.  **SIMD Ininterrumpido:** La Fase 1 (donde reside el 95% del cómputo) se ejecuta sin un solo condicional, garantizando latencia predecible.
2.  **Cero Branching Penalty:** Las Fases 2 y 3 solo se ejecutan sobre subconjuntos predefinidos, sin tener que "preguntar" a los datos qué precisión tienen.
3.  **Caché Amigable:** Leer vectores planos predecibles (`base_strands`) maximiza el ratio de acierto en L1/L2 del procesador (vital en dispositivos móviles Android).
4.  **Escalabilidad Matemática:** Este enfoque aprovecha la propiedad distributiva del producto punto: `(A_base + A_epi) * X = (A_base * X) + (A_epi * X)`.
