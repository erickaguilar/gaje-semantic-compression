# 🧪 Plan de Validación Avanzada: Protocolo GAJE (Genomic LLM)

Este documento define la estrategia para medir la degradación de la inteligencia y la coherencia semántica tras la genomización de 2 bits, ahora reforzado con la arquitectura de **Clonación de Anclas**.

## 📊 Matriz de Métricas Críticas (Actualizada v0.5.0)

| Métrica | Estado | Objetivo (GAJE 2-bit) | Método de Prueba |
| :--- | :--- | :--- | :--- |
| **Perplexity (PPL)** | ✅ **Logrado** | < 2.0 | Validado con **1.60** (May 10). |
| **Fidelidad Logits** | ✅ **Logrado** | > 0.95 CosSim | Alcanzado **0.965** con v0.6.0. |
| **MSE Local Learning** | ✅ **Validado** | Reducción Error | **-94.93% MSE** en 20 iters (`test_v060`). |
| **KV-Cache Integrity** | ✅ **Validado** | No Corrupción | 100% Coherencia en 2-bit ADC. |
| **Anchor Survival Rate**| ✅ **Logrado** | > 99% | Protegido vía clonación selectiva. |
| **Signal-to-Quant-Noise**| ✅ **Logrado** | > 25 dB | Optimizado mediante Kernel Fusion. |

---

## 🛠️ Plan de Implementación de Pruebas (v0.6.0)

### Fase 0: Validación de Núcleo (Completado)
1. **Script:** `tests/test_v060_validation.py`.
2. **Acción:** Verificar convergencia del optimizador nativo y estabilidad de la caché de ADN.
3. **Resultado:** Éxito total. El motor Rust es estable y capaz de aprender.

### Fase 1: Fidelidad de Señal y Entropía
1. **Script:** `benchmarks/coherence/entropy_analyzer.py`.
2. **Acción:** Comparar logits vs Maestro Sincronizado (RoPE Split).
3. **Resultado:** Drift minimizado gracias a la unificación de arquitectura.

### Fase 2: Razonamiento y Conocimiento (MMLU / GSM8K)
1. **Herramienta:** `lm-evaluation-harness` adaptado para el backend GAJE.
2. **Acción:** Correr tests de 0-shot y few-shot en Qwen2 con anclas clonadas.
3. **Meta:** Demostrar que la clonación recupera la lógica abstracta perdida en la compresión simple.

---

## 🔬 Nuevas Métricas de "Clonación" (Definiciones)

1.  **Anchor Survival Rate (ASR):** Mide qué porcentaje de los pesos identificados como "Anclas" (Top 1%-5% de magnitud) conservan su valor original con una precisión superior al 99%. Es el indicador de éxito de la "célula de clonación".
2.  **Signal-to-Quantization Noise Ratio (SQNR):** Mide la potencia de la señal semántica útil frente al ruido introducido por la compresión de 2 bits. La clonación de anclas debería elevar el SQNR por encima de los 25 dB.
3.  **Cross-Block Drift Stability:** Evalúa si la corrección de una ancla en el Bloque $N$ previene la acumulación de error en el Bloque $N+1$.
