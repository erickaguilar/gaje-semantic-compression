# 📈 Estrategia de Optimización de Perplexity (PPL)

## 1. Diagnóstico de la Fase 2
Las pruebas iniciales de Perplexity arrojaron valores críticos (>1M). El análisis técnico revela que el modelo genómico mantiene la **energía semántica** (Fidelidad de Logits 0.94), pero pierde la **coherencia secuencial** debido a la falta de un loop de atención completo y preciso.

## 2. Recomendaciones de Ingeniería

### A. Implementación de KV-Cache Genómico (Rust)
- **Problema:** El recálculo de la atención en Python es lento e impreciso para secuencias largas.
- **Solución:** Mover el almacenamiento de los tensores Key (K) y Value (V) al motor de Rust en formato genómico (2-bit).
- **Impacto:** Reducción de la latencia y mejora de la precisión al mantener el historial completo de la secuencia.

### B. Kernel de Atención Multi-Cabeza (MHA) Nativo
- **Problema:** La aproximación de atención escalar actual no captura los matices de Qwen2.
- **Solución:** Implementar el producto escalar de matrices (Dot-product Attention) en Rust usando instrucciones SIMD para operar sobre los pesos genómicos de forma paralela.

### C. Calibración de Centroides Max-Lloyd
- **Problema:** Los centroides globales [±1.51, ±0.45] son genéricos.
- **Solución:** Implementar entrenamiento de centroides por bloque (Block-wise Quantization) para que cada capa tenga su propio codebook óptimo.
- **Impacto:** Reducción del ruido de cuantización en las capas FFN.

### D. Safe SiLU & Clipping
- **Estado:** ✅ Implementado.
- **Resultado:** Se eliminaron los overflows en la función de activación, estabilizando la señal a través de 24 bloques.

## 3. Próximos Pasos
1. Integrar `GenomicAttention` en `src/lib.rs`.
2. Validar PPL nuevamente tras la implementación del KV-Cache.
3. Avanzar hacia MMLU una vez que PPL sea < 100.
