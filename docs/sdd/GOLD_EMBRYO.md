# 🧬 SDD: Arquitectura del Embrión de Oro v2.0 (10 MB High-Fidelity)

## 1. Resumen Ejecutivo
El "Embrión de Oro v2.0" evoluciona de la meta original de 4MB a un objetivo de **10 MB**. Este cambio estratégico responde a la necesidad de mayor resolución semántica para resolver problemas de coherencia (Semantic Drift) encontrados en la versión micro. 10MB representa el equilibrio perfecto entre compresión extrema y capacidad de razonamiento funcional.

## 2. Especificaciones Técnicas (Actualizadas)
| Parámetro | Valor (v1.0) | Valor (v2.0) | Ventaja |
| :--- | :--- | :--- | :--- |
| **Embedding (`n_embd`)** | 384 | **512** | Mayor densidad vectorial |
| **Bloques (`n_blocks`)** | 8 | **12** | Mayor profundidad lógica |
| **Cabezas (`n_head`)** | 6 | **8** | Mejor atención multi-foco |
| **Vocabulario** | 16,384 | **32,768** | Soporte gramatical completo |
| **Límite de Tamaño** | ~4 MB | **10 MB** | Espacio para Anclas Críticas |

## 3. Análisis de Memoria (Matemática de los 10 MB)

### A. Embeddings & LM Head
*   Vocab (32,768) * n_embd (512) = 16,777,216 pesos.
*   En 2-bits: 16.7M / 4 = **4,194,304 bytes (~4.2 MB)**. (Incluye Input y Output).

### B. Bloques de Inteligencia (12 Bloques)
*   Atención + FFN por bloque (aprox. 1.5M pesos) = 1.5M / 4 = ~375KB por bloque.
*   Total Bloques: 12 * 375KB = **4,500,000 bytes (~4.5 MB)**.

### C. Margen de Fidelidad (1.3 MB)
*   **Anclas de Oro (Top 5%):** 1 MB dedicado a pesos de alta precisión (16-bit) para proteger la "columna vertebral" del modelo.
*   **Metadatos y Topología:** 0.3 MB para el Grafo de Centroides (Fase 4.0).

**Total Final Proyectado:** **~10.0 MB**.

## 4. Estrategia de Crianza v2.0
1.  **Resolución de Deriva:** El aumento a 512 dimensiones reduce el ruido de colisión en el espacio latente.
2.  **Entrenamiento por Resonancia Relacional:** Los 12 bloques permiten una jerarquía de abstracción más clara (capas bajas: gramática, capas medias: hechos, capas altas: razonamiento).
3.  **Tokenizador Silver:** Migración a un vocabulario de 32k tokens para evitar la fragmentación excesiva de palabras.
