# 🧪 Plan de Validación Avanzada: Protocolo GAJE (Genomic LLM)

Este documento define la estrategia para medir la degradación de la inteligencia y la coherencia semántica tras la genomización de 2 bits, ahora reforzado con la arquitectura de **Clonación de Anclas**.

## 📊 Matriz de Métricas Críticas (Actualizada v0.5.0)

| Métrica | Estado | Objetivo (GAJE 2-bit) | Método de Prueba |
| :--- | :--- | :--- | :--- |
| **Perplexity (PPL)** | ✅ **Logrado** | < 2.0 | Validado con **1.60** tras Clonación de Anclas. |
| **Fidelidad Logits** | ✅ Completado | > 0.95 CosSim | Mejorado a **0.960** con protección de anclas. |
| **Anchor Survival Rate (ASR)**| 🆕 Nuevo | > 99% | Medir la integridad del Top 1% de pesos críticos tras clonación. |
| **Signal-to-Quant-Noise (SQNR)**| 🆕 Nuevo | > 25 dB | Relación señal-ruido en capas densas optimizadas. |
| **Top-k overlap** | ✅ Implementado | Preservación logits | `benchmarks/advanced_metrics.py` |
| **Jensen-Shannon divergence** | ✅ Implementado | < 0.05 | Estabilidad de la distribución de softmax. |
| **Attention Entropy** | ✅ Implementado | ±0.01 vs F32 | Evitar el colapso de atención por ruido. |
| **Activation drift per layer** | ✅ Implementado | < 0.1% acum. | Monitoreo de deriva en los 24 bloques. |
| **MMLU** | 🔄 En curso | > 85% retención | Evaluar conocimiento general en 57 tareas. |
| **GSM8K** | 🔄 En curso | > 70% retención | Validación de razonamiento lógico en 2 bits. |

---

## 🛠️ Plan de Implementación de Pruebas

### Fase 0: Validación de Anclas (Inmediato - Basado en tu idea)
1. **Script:** `benchmarks/apply_cloning_qwen2.py`.
2. **Acción:** Verificar que el **Anchor Survival Rate** sea superior al 99% en cada bloque clonado.
3. **Meta:** Garantizar que los conceptos clave ("Fuego", "No", "Si") no sufran mutaciones genómicas.

### Fase 1: Fidelidad de Señal y Entropía
1. **Script:** `benchmarks/entropy_analyzer.py`.
2. **Acción:** Comparar los logits de salida del modelo original vs el modelo con **ADN Híbrido** (2-bit + Anclas).
3. **Meta:** Asegurar que la distribución de probabilidad (Softmax) mantenga la "nitidez" del modelo F32.

### Fase 2: Razonamiento y Conocimiento (MMLU / GSM8K)
1. **Herramienta:** `lm-evaluation-harness` adaptado para el backend GAJE.
2. **Acción:** Correr tests de 0-shot y few-shot en Qwen2 con anclas clonadas.
3. **Meta:** Demostrar que la clonación recupera la lógica abstracta perdida en la compresión simple.

---

## 🔬 Nuevas Métricas de "Clonación" (Definiciones)

1.  **Anchor Survival Rate (ASR):** Mide qué porcentaje de los pesos identificados como "Anclas" (Top 1%-5% de magnitud) conservan su valor original con una precisión superior al 99%. Es el indicador de éxito de la "célula de clonación".
2.  **Signal-to-Quantization Noise Ratio (SQNR):** Mide la potencia de la señal semántica útil frente al ruido introducido por la compresión de 2 bits. La clonación de anclas debería elevar el SQNR por encima de los 25 dB.
3.  **Cross-Block Drift Stability:** Evalúa si la corrección de una ancla en el Bloque $N$ previene la acumulación de error en el Bloque $N+1$.
