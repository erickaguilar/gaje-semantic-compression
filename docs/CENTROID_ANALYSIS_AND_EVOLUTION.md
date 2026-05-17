# Análisis y Evolución de los Centroides: El ADN del Protocolo GAJE

**Fecha:** 11 de Mayo, 2026  
**Documento:** Memoria Técnica de Evolución de Cuantización

## 1. ¿Qué es un Centroide en GAJE?
En el Protocolo GAJE, la cuantización de 2 bits reduce la precisión de un peso neuronal de 32 bits a solo 4 estados posibles (`00, 01, 10, 11`). Los **Centroides** son los valores reales (flotantes) que estos bits representan. 

Si el ADN (`database`) es el código, los centroides son la "fuerza" semántica que le da sentido a ese código.

## 2. Hallazgos en la Evolución de los Centroides

### Evolución 1: Centroides Globales (Estáticos)
*   **Método:** Uso de 4 valores fijos para todo el modelo.
*   **Hallazgo:** Pérdida masiva de información. El modelo "ensordece" porque no todas las capas tienen la misma distribución de pesos.

### Evolución 2: Centroides por Capa (Layer-wise)
*   **Método:** 4 valores distintos para cada capa del Transformer.
*   **Hallazgo:** El modelo comienza a mostrar signos de inteligencia, pero la señal es "débil". La varianza de los pesos dentro de una misma capa es demasiado alta para ser capturada por solo 4 valores globales de capa.

### Evolución 3: Adaptación Genómica Local (Block-wise) - *Estado Actual*
*   **Método:** Cada bloque de 32 dimensiones tiene sus propios 4 centroides, calculados dinámicamente según la Media ($\mu$) y Desviación Estándar ($\sigma$) local.
*   **Hallazgo:** Máxima fidelidad teórica. Sin embargo, se identificó una **"Pérdida de Fuerza"** durante la transferencia de estos centroides entre Rust y Python. La conversión a listas de Python truncaba la precisión y generaba latencia.

## 3. Hallazgos Críticos sobre la "Fuerza Semántica"

Durante la estabilización de la v0.6.1, descubrimos por qué el modelo perdía coherencia:

1.  **Alineación Energética:** Si los centroides no se escalan correctamente antes de la atención, la varianza de la señal disminuye en cada capa. Al llegar a la capa 24, la señal es tan débil que el modelo genera tokens nulos (paréntesis, espacios).
2.  **Saturación (Clamping):** Los límites de protección de los centroides eran demasiado estrechos (-64, 64). Se halló que las "Anclas" (anchors) requieren un rango dinámico de hasta **160** para preservar las neuronas de alta activación que definen los conceptos complejos.
3.  **El Cuello de Botella del Puntero:** El paso de millones de centroides de Rust a Python mediante listas destruía la ventaja competitiva de GAJE. La implementación de **Punteros Directos (NumPy)** conservó la integridad decimal de los centroides y eliminó el 90% de la latencia.

## 4. Hacia la Evolución 4: Centroides Protegidos
El hallazgo final indica que para que los centroides mantengan su "fuerza" absoluta, no deben ser procesados por el intérprete de Python entre capas.
*   **Propuesta:** Mantener los centroides en la memoria caché del procesador (L1/L2) gestionada exclusivamente por Rust.
*   **Impacto:** Los centroides se convierten en valores de "fase pura", alineados perfectamente con RoPE sin interferencia del recolector de basura de Python.

## 5. Conclusión
Los centroides han evolucionado de ser simples constantes a ser un **Sistema de Mapeo Estadístico Dinámico**. La clave de la inteligencia en 2 bits no está en el ADN (los bits), sino en la precisión y "fuerza" con la que los centroides traducen ese ADN a activaciones neuronales.
