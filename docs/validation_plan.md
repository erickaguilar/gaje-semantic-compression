# 🧪 Plan de Validación Avanzada: Protocolo GAJE (Genomic LLM)

Este documento define la estrategia para medir la degradación de la inteligencia y la coherencia semántica tras la genomización de 2 bits.

## 📊 Matriz de Métricas Críticas

| Métrica | Estado | Objetivo (GAJE 2-bit) | Método de Prueba |
| :--- | :--- | :--- | :--- |
| **Fidelidad Logits** | ✅ Completado | > 0.90 CosSim | Validado con **0.9456** CosSim y **0.0125** KL Div. |
| **Perplexity (PPL)** | ⚠️ En Mejora | < 1.1x vs F32 | Inicial: >1M. Requiere integración de KV-Cache y MHA en Rust. |
| **Top-k overlap** | ✅ Implementado | Preservación logits | `benchmarks/advanced_metrics.py` |
| **Jensen-Shannon divergence** | ✅ Implementado | Mejor estabilidad que KL | `benchmarks/advanced_metrics.py` |
| **Attention entropy** | ✅ Implementado | Detectar colapso | `benchmarks/advanced_metrics.py` |
| **Activation drift per layer** | ✅ Implementado | Localizar degradación | `benchmarks/advanced_metrics.py` |
| **Token repetition score** | ✅ Implementado | Loops autoregresivos | `benchmarks/advanced_metrics.py` |
| **Semantic consistency score** | ✅ Implementado | Coherencia narrativa | `benchmarks/advanced_metrics.py` |
| **MMLU** | Faltante | > 85% retención | Evaluar conocimiento general en 57 tareas. |
| **GSM8K** | Faltante | > 70% retención | Resolver problemas matemáticos de primaria para validar razonamiento lógico en 2 bits. |
| **Hallucination Rate** | Faltante | < 5% incremento | Comparar respuestas basadas en hechos (NQ Dataset) entre original y genómico. |
| **Long-context Coherence** | Faltante | 128k tokens | Validar el "Needle In A Haystack" usando KV-Cache genómico comprimido. |
| **Attention Fidelity** | Faltante | > 0.90 CosSim | Medir la similitud entre el mapa de atención original y el mapa de atención genómico (Q*K). |
| **Multilingual Robustness** | Faltante | > 80% similitud | Validar si la compresión de 2 bits afecta más a ciertos idiomas (English vs Spanish vs Chinese). |
| **Token Entropy** | Faltante | ±0.05 bits | Analizar la distribución de probabilidad de salida para detectar colapsos de modo. |

---

## 🛠️ Plan de Implementación de Pruebas

### Fase 1: Fidelidad de Señal y Entropía (Inmediato)
1. **Script:** `benchmarks/entropy_analyzer.py`.
2. **Acción:** Comparar los logits de salida del modelo original vs genomizado para la misma frase.
3. **Meta:** Asegurar que la distribución de probabilidad (Softmax) no se aplane excesivamente.

### Fase 2: Razonamiento y Conocimiento (MMLU / GSM8K)
1. **Herramienta:** `lm-evaluation-harness` adaptado para el backend GAJE.
2. **Acción:** Correr tests de 0-shot y few-shot en Qwen2 genomizado.
3. **Meta:** Demostrar que los "pesos de ADN" conservan la lógica abstracta.

### Fase 3: Estrés de Contexto Largo
1. **Acción:** Llenar el KV-Cache genómico con documentos técnicos y pedir un dato específico al final.
2. **Meta:** Validar que el ahorro del 93.75% en RAM permite búsquedas precisas en contextos de +100k tokens.

## 📈 Dashboard de Seguimiento de Calidad
*Este dashboard debe actualizarse tras cada mejora en el algoritmo de entrenamiento de centroides (Max-Lloyd/Block-Quant).*

- **Baseline (F32):** Inteligencia de referencia (100%).
- **GAJE Alpha (Current):** Estimado 80-85% de retención de inteligencia.
- **GAJE Beta (Target):** >95% de retención con compresión 16x.

---

## 🔍 Detalle de Métricas Técnicas (Nuevas)

Para asegurar la integridad del modelo tras la compresión de 2 bits, se implementan las siguientes métricas de monitoreo profundo:

1.  **Top-k Overlap:** Mide cuántos de los top-k tokens (ej. k=10) se mantienen iguales entre el modelo F32 y GAJE. Es vital para asegurar que la "intención" inmediata del modelo no cambie.
2.  **Jensen-Shannon Divergence (JSD):** A diferencia de KL, JSD es simétrica y siempre acotada, lo que proporciona una métrica de estabilidad superior para comparar distribuciones de softmax muy divergentes.
3.  **Attention Entropy:** Un colapso en la entropía de atención indicaría que el modelo se está enfocando en un solo token (posiblemente basura) debido a la cuantización agresiva.
4.  **Activation Drift per Layer:** Permite localizar exactamente en qué capa (bloque transformer) se empieza a perder la fidelidad de la señal, facilitando el "fine-tuning" selectivo de centroides.
5.  **Token Repetition Score:** Crucial para detectar si el modelo de 2 bits entra en bucles infinitos de repetición (loops autoregresivos), un síntoma común de degradación semántica.
6.  **Semantic Consistency Score:** Utiliza un modelo de embeddings externo para validar que el significado global de la respuesta generada sea consistente con la intención original, más allá de la coincidencia exacta de tokens.
