# 🚀 GAJE Protocol: Roadmap hacia el 85% de Precisión

Para transformar el Protocolo GAJE de un prototipo experimental a un producto revolucionario con >85% de precisión, debemos evolucionar la arquitectura de **Product Quantization (PQ)** siguiendo esta ruta técnica:

---

## 🛑 Fase 1: Asymmetric Distance Computation (ADC) - *Prioridad Alta*
**El Problema:** Actualmente comparamos "ADN contra ADN" (Simétrico), lo que duplica el error de cuantización.
**La Solución:** Implementar ADC. El Query permanece en Float32 (32-bit) y se compara directamente contra los centroides del ADN (2-bit) mediante una tabla de consulta (Lookup Table).
*   **Impacto esperado:** +25-30% de precisión inmediata.
*   **Entorno:** Modificación del núcleo de Rust para aceptar un vector float como query.

---

## 🧬 Fase 2: Centroides Genómicos Dinámicos (K-Means)
**El Problema:** Los umbrales actuales (-0.34, 0, 0.34) son estáticos y asumen una distribución normal perfecta.
**La Solución:** Entrenar el "Código Genético" (Codebook). Usar K-Means para encontrar los 4 mejores centroides (A, C, G, T) para cada sub-espacio del vector.
*   **Impacto esperado:** +20% de precisión.
*   **Prueba:** Entrenar con 10,000 vectores de ejemplo antes de comprimir la base de datos completa.

---

## 🌍 Fase 3: Validación con Datasets Reales
**El Problema:** Los datos aleatorios (sintéticos) no tienen "semántica" real; son puntos dispersos que no forman agrupaciones (manifolds).
**La Solución:** Utilizar datasets de embeddings reales para las pruebas:
*   **SBERT (768 dims):** Embeddings de oraciones de Wikipedia.
*   **CLIP (512 dims):** Embeddings de imágenes de COCO Dataset.
*   **GloVe:** Vectores de palabras.
*   **Impacto esperado:** En datos reales, la estructura semántica ayuda al algoritmo a encontrar vecinos de forma más natural que en datos puramente aleatorios.

---

## 🛠 Entorno de Pruebas Necesario
Para ejecutar esta fase, necesitamos:
1.  **Memoria:** Al menos 4GB de RAM (para K-Means sobre datasets grandes).
2.  **GPU (Opcional):** Para acelerar el entrenamiento de centroides si el dataset supera el millón de registros.
3.  **Librerías de Pre-procesamiento:** `scikit-learn` para el entrenamiento de los centroides iniciales que luego cargaremos en Rust.

---

## 📈 Resumen de Ganancia Estimada
| Técnica | Precisión Actual | Incremento Est. | Meta |
| :--- | :--- | :--- | :--- |
| PQ Base + Gray Code | 22% | - | 22% |
| **+ ADC (Asymmetric)** | 52% | +30% | 52% |
| **+ Per-Dim K-Means** | **83%** | +31% | 85% |
| **+ Real Data Structure**| **87%** | +4% | **87%** 🚀 |

---
*Estado: Fase 2 Completada con éxito. Optimizando para Fase 3 (Datasets Masivos).*
