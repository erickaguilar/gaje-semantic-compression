# 🧬 SDD: Arquitectura del Embrión de Oro (GAJE v1.0)

## 1. Resumen Ejecutivo
El "Embrión de Oro" es la unidad mínima de inteligencia genómica diseñada para evolucionar de forma autónoma. Este documento justifica la viabilidad técnica de una arquitectura de 8 bloques bajo un límite de 10 MB utilizando el motor GAJE (2-bits).

## 2. Especificaciones Técnicas
| Parámetro | Valor |
| :--- | :--- |
| **Embedding (`n_embd`)** | 384 |
| **Bloques (`n_blocks`)** | 8 |
| **Cabezas (`n_head`)** | 6 (64 dim/head) |
| **Vocabulario** | 16,384 |
| **Cuantización** | 2-bit (GAJE-DNA) |

## 3. Análisis de Memoria (Matemática de los 10 MB)
La memoria en GAJE se calcula principalmente por la cantidad de "bases nitrogenadas" (pesos de 2 bits).

### A. Embeddings
*   Vocab (16,384) * n_embd (384) = 6,291,456 pesos.
*   En 2-bits: 6,291,456 / 4 = **1,572,864 bytes (~1.5 MB)**.

### B. Bloques de Atención (por bloque)
*   Q, K, V (3 * 384 * 384) = 442,368 pesos.
*   Output (384 * 384) = 147,456 pesos.
*   Total Atención: 589,824 pesos / 4 = **147,456 bytes**.

### C. Bloques FFN (por bloque)
*   Gate, Up (2 * 384 * 1024) = 786,432 pesos (asumiendo 1024 de expansión).
*   Down (1024 * 384) = 393,216 pesos.
*   Total FFN: 1,179,648 pesos / 4 = **294,912 bytes**.

### D. Totales Proyectados
*   Embeddings: 1.5 MB
*   Bloques (8 * (147k + 295k)): 8 * 442,368 bytes = **3,538,944 bytes (~3.5 MB)**.
*   LM Head: Igual que embeddings = **1.5 MB**.
*   **Total Estimado:** 1.5 + 3.5 + 1.5 = **6.5 MB**.

**Conclusión:** La arquitectura propuesta tiene un margen de ~3.5 MB para metadatos, anclas y centroides, cumpliendo holgadamente la meta de < 10 MB.

## 4. Contrato de Inicialización
*   **DNA:** Generado mediante `generate_random_dna` (XOR de alta entropía).
*   **Centroides:** Inicialización basal `[-1.5, -0.5, 0.5, 1.5]` normalizada por la desviación estándar teórica de Xavier/Kaiming para 384 dimensiones.
