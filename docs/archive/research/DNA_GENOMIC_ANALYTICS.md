# 🧬 Genoma Computacional: Análisis de Secuencias ADN (2-bit) en GAJE-Flow

## 📋 Resumen Ejecutivo

Este documento define el fundamento matemático, la interpretación de teoría de la información y las aplicaciones analíticas de la secuencia ADN de 2 bits en el protocolo **GAJE (Genomic Adaptive Joint Embedding)**.

---

## 1. Fundamento Matemático y Mapeo Cuaternario

En el motor GAJE, la representación en ADN no es una simulación biológica, sino una **biyección matemática exacta** entre el espacio de cuantización de 2 bits y un alfabeto cuaternario de 4 símbolos $\Sigma = \{A, C, G, T\}$:

$$\text{Mapeo Cuaternario}: \quad 00_2 \rightarrow \mathbf{A}, \quad 01_2 \rightarrow \mathbf{C}, \quad 11_2 \rightarrow \mathbf{G}, \quad 10_2 \rightarrow \mathbf{T}$$

- **Densidad de Almacenamiento**: 1 byte (8 bits) empaqueta **4 nucleótidos** ($\text{ej. } 11010010_2 = \mathbf{G}\text{-}\mathbf{C}\text{-}\mathbf{A}\text{-}\mathbf{T}$).
- **Representación de Cadenas**: Una secuencia visual de 128 letras representa 32 bytes contiguos (o 64 parámetros de 2 bits) del espacio de pesos latente.

---

## 2. Interpretación en Teoría de la Información

### A. Sesgo de Centroides y Contenido GC
La frecuencia observada de nucleótidos (ej. abundancia de $\mathbf{G}$ y $\mathbf{C}$ frente a $\mathbf{A}$ y $\mathbf{T}$) refleja directamente la distribución de los pesos alrededor de los centroides de cuantización:

- **Regiones GC-Rich ($\mathbf{G} = 11_2, \mathbf{C} = 01_2$)**: Representan tensores con activaciones de alta magnitud y centroides positivos.
- **Regiones AT-Rich ($\mathbf{A} = 00_2, \mathbf{T} = 10_2$)**: Representan zonas de supresión o pesos de baja magnitud.

### B. Entropía de Shannon por Capa
Se define la entropía cuaternaria de una secuencia en el bloque $L$ como:

$$H(L) = -\sum_{i \in \{A,C,G,T\}} P(x_i) \log_2 P(x_i)$$

- Si $H(L) \approx 2.0$, la capa utiliza de manera óptima los 4 estados de cuantización.
- Si $H(L) \ll 2.0$ (saturación de un solo nucleótido), indica un **colapso de varianza**, señalando redundancia que puede ser comprimida o recalibrada.

---

## 3. Líneas de Investigación y Análisis Omico en Redes Neuronales

| Método Analítico | Aplicación en IA / GAJE | Métrica de Evaluación |
| :--- | :--- | :--- |
| **Análisis de $k$-mers** | Medir la frecuencia de sub-secuencias repetitivas (ej. `"GCCG"`, `"CCGG"`) para detectar patrones de peso redundantes. | Conteo de frecuencia de $k$-mers por bloque. |
| **Complejidad de Lempel-Ziv** | Evaluar la compresibilidad simbólica de las capas de Atención vs. FFN SwiGLU. | Ratio de complejidad $LZ(L)$. |
| **Alineamiento de Secuencias (Needleman-Wunsch)** | Medir la deriva genómica entre dos checkpoints o modelos (ej. SmolLM2 vs Qwen2). | Distancia de Alineamiento / Similitud. |
| **Filogenética Computacional** | Construir árboles evolutivos de modelos de lenguaje basados en su distancia cuaternaria. | Matriz de Distancia Genómica. |
| **Mutagénesis Dirigida (SNPs)** | Alterar un solo nucleótido ($\mathbf{G} \rightarrow \mathbf{A}$) en `.gaje` para medir la sensibilidad semántica a nivel de parámetro individual. | Delta CosSim / Logit Shift. |

---

## 💡 Conclusión

La representación ADN en GAJE combina la **Teoría de la Información de Shannon** con el **Análisis de Secuencias Discretas**. Ofrece una interfaz intuitiva para el usuario final y proporciona un marco riguroso para diagnosticar la salud y entropía de las redes neuronales comprimidas.
