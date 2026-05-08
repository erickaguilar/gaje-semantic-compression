# 🧪 Plan de Validación Avanzada: Protocolo GAJE (Genomic LLM)

Este documento define la estrategia para medir la degradación de la inteligencia y la coherencia semántica tras la genomización de 2 bits.

## 📊 Matriz de Métricas Críticas

| Métrica | Estado | Objetivo (GAJE 2-bit) | Método de Prueba |
| :--- | :--- | :--- | :--- |
| **Fidelidad Logits** | ✅ Completado | > 0.90 CosSim | Validado con **0.9456** CosSim y **0.0125** KL Div. |
| **Perplexity (PPL)** | ⚠️ En Mejora | < 1.1x vs F32 | Inicial: >1M. Requiere integración de KV-Cache y MHA en Rust. |
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
