# 🗺️ QAT Integration Roadmap: Operation Precise Compression

## 1. Visión General
El objetivo de este plan es integrar **Quantization-Aware Training (QAT)** en el pipeline de GAJE para permitir modelos de ultra-baja precisión (4-bit/2-bit) que mantengan la coherencia semántica (PPL < 15.0) en dispositivos con recursos limitados.

## 2. Fases de Implementación

### Fase 1: Simulación en Python (Research Bridge)
*   **Implementar `QATSimulator`**: Un wrapper en `python/gaje/nn/` que simule el redondeo de pesos y activaciones durante el forward pass.
*   **Cálculo de Factores de Escala Estáticos**: Algoritmos para determinar los rangos óptimos de cuantización durante el entrenamiento.
*   **Validación de Gradientes**: Asegurar que el "Straight-Through Estimator" (STE) funcione correctamente para permitir el aprendizaje a pesar de la cuantización.

### Fase 2: Evolución del Formato `.gaje`
*   **Metadatos de Precisión**: Actualizar la cabecera del formato para incluir `bit_depth` por lóbulo.
*   **Bloques de Escala**: Almacenar factores de escala (`scales`) y desplazamientos (`zeros`) junto a los tensores comprimidos.

### Fase 3: Optimización del Motor en Rust (Soberanía Nativa)
*   **Kernels de De-cuantización al Vuelo**: Optimizar `src/compute/` para realizar la de-cuantización directamente en los registros de la CPU/NPU.
*   **Soporte de Activaciones Estáticas**: Implementar el path de ejecución que usa los factores de escala pre-calculados, eliminando la necesidad de cálculos de rango en tiempo real.

### Fase 4: Entrenamiento y Certificación
*   **Fine-tuning con QAT**: Re-entrenar el modelo `Silver Adult` usando el simulador QAT.
*   **Benchmark Comparativo**: Comparar PPL y latencia entre `F32` (Baseline) vs `QAT-4bit` vs `QAT-2bit`.

## 3. Entregables Técnicos
1.  `docs/sdd/QAT_IMPLEMENTATION_DETAILS.md`: Especificación de los kernels de Rust.
2.  `python/gaje/nn/_impl/qat_wrapper.py`: Lógica de simulación.
3.  `src/nn/quantized_layer.rs`: Implementación nativa.

## 4. Métricas de Éxito
*   **Tamaño del Modelo**: < 100MB para el modelo `Silver Adult` (4-bit).
*   **Precisión**: Degradación de PPL < 2% respecto al modelo F32 original.
*   **Latencia**: Mejora del 30% en inferencia en dispositivos Android (Termux).
