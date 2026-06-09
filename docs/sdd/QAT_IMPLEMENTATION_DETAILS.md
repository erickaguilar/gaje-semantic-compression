# 📐 SDD: QAT Implementation Details (Native Rust)

## 1. Estructura de Tensores Cuantizados
Los tensores en el motor Rust dejarán de ser simples `Vec<f32>` para usar una estructura de bloques:

```rust
struct QuantizedTensor {
    data: Vec<u8>,        // Datos comprimidos (2, 4 u 8 bits)
    scales: Vec<f32>,     // Factores de escala por canal/bloque
    zeros: Vec<f32>,      // Puntos cero (offsets)
    precision: BitDepth,  // Enum: Q2, Q4, Q8
}
```

## 2. Kernels de Computación
Para maximizar la velocidad, la de-cuantización se fusionará con la multiplicación de matrices (Gemm):

*   **Fusión de Operaciones**: `output = (quantized_weight - zero) * scale * activation`.
*   **SIMD Optimization**: Uso de instrucciones `AVX2`/`NEON` para procesar múltiples pesos de 4 bits en un solo ciclo de reloj.

## 3. Manejo de Activaciones Estáticas
Las activaciones también serán cuantizadas para reducir el ancho de banda de memoria:
*   Durante el entrenamiento (QAT), el modelo aprenderá el rango máximo de las activaciones en cada capa.
*   En inferencia, usaremos estos rangos fijos para evitar el escaneo del tensor de entrada.

## 4. Cambios en `src/compute/`
Se añadirán los siguientes sub-módulos:
*   `src/compute/quant/mod.rs`: Lógica genérica de cuantización.
*   `src/compute/quant/kernels.rs`: Implementaciones altamente optimizadas para ARM y x86.
```
