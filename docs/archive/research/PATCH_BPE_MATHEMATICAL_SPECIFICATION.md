# 🧬 Especificación Matemática y Arquitectura de Patch-BPE Jerárquico para GAJE-Flow

**Fecha:** 30 de mayo de 2026
**Estatus:** Propuesta de Investigación / Diseño de Arquitectura (v2.0-alpha)
**Clasificación:** Confidencial - Protocolo GAJE-Flow

---

## 1. Introducción y Motivación

En modelos de lenguaje de escala ultraligera (10MB a 50MB) optimizados para dispositivos de borde (móviles/IoT), los tokenizadores tradicionales basados en subpalabras (como BPE de HuggingFace) presentan un cuello de botella de almacenamiento inaceptable.

Para un vocabulario típico de $V = 32,768$ tokens y una dimensión de embedding de $d = 512$, las matrices de mapeo de entrada y salida (`lm_head`) consumen:

$$\text{Parámetros} = 2 \times (V \times d) = 2 \times (32,768 \times 512) = 33,554,432$$

Aún cuantizados a **2 bits por peso**, estas capas consumen aproximadamente **8.4 MB**, representando más del **80% del presupuesto total** de un modelo Silver de 10 MB. Esto reduce drásticamente el espacio disponible para los bloques de atención (la capacidad lógica del modelo).

Por otro lado, los enfoques libres de tokenizadores a nivel de byte en crudo (*Byte-level Token-Free*) resuelven el problema del tamaño del vocabulario (reduciendo $V$ a 256 bytes, lo que equivale a solo 64 KB en disco), pero multiplican la longitud de la secuencia por un factor de $4\text{x}$ a $6\text{x}$. Dado que la complejidad de la atención es cuadrática $O(T^2)$ con respecto a la longitud de secuencia $T$, esto satura la capacidad de cómputo de las CPUs móviles y aumenta exponencialmente el tamaño del KV-Cache.

**Patch-BPE (Patch-level Byte Encoding)** resuelve esta contradicción mediante una arquitectura jerárquica de dos niveles: procesa bytes a nivel local mediante un codificador superficial y opera a nivel de parches continuos de alta dimensionalidad en el transformador de fase profunda.

---

## 2. Diagrama de la Arquitectura Jerárquica

La separación de responsabilidades entre el procesamiento de bytes locales (alta frecuencia, baja dimensionalidad) y los parches semánticos globales (baja frecuencia, alta dimensionalidad) se estructura del siguiente modo:

```mermaid
graph TD
    IN[Texto en UTF-8] --> |Stream de Bytes| BYTES[Byte Stream: B_1, B_2, ..., B_N]
    BYTES --> |Segmentación Fija de tamaño P| PACKETS[Parches de Bytes: P_k]

    subgraph Codificador de Parches (Local - FP16/INT8)
        PACKETS --> |Concatenación / One-Hot| OH[Representación Matricial: B_k]
        OH --> |Proyección Lineal Encoder| ENC[MLP / Proyección lineal]
    end

    ENC --> |Secuencia de Vectores Continuos: e_k| CORE[GAJE Toroidal Transformer: 2-bit]

    subgraph Decodificador de Parches (Local - Autoregresivo)
        CORE --> |Vector Predicho: h_next| DEC[Proyección lineal Decoder]
        DEC --> |Predicción de Distribución Multivariada| OUT_BYTES[Bytes Reconstruidos]
    end

    OUT_BYTES --> |Stream de salida| TEXT_OUT[Texto en UTF-8]
```

---

## 3. Formulación Matemática de Patch-BPE

### A. Proyección de Parches de Entrada (Patch Encoder)
Sea un stream de bytes de entrada representado como enteros de 8 bits sin signo: $b_n \in \{0, 1, ..., 255\}$. Agrupamos los bytes en bloques no superpuestos de tamaño de parche fijo $P$ (por ejemplo, $P = 4$ bytes).

Para el parche $k$-ésimo, definimos el bloque de bytes como:

$$\mathbf{p}_k = [b_{(k-1)P + 1}, b_{(k-1)P + 2}, ..., b_{kP}] \in \mathbb{N}^{P}$$

Cada byte del bloque se proyecta en una representación *one-hot* $\mathbf{v}_n \in \{0, 1\}^{256}$. Concatenamos los vectores del bloque para formar la matriz de parches:

$$\mathbf{B}_k = [\mathbf{v}_1, \mathbf{v}_2, ..., \mathbf{v}_P] \in \{0, 1\}^{P \times 256}$$

El codificador de parches (un proyector lineal compacto) aplana esta matriz y la mapea al espacio continuo de embedding de fase de la red $\mathbf{e}_k \in \mathbb{R}^{d}$:

$$\mathbf{e}_k = \text{LayerNorm}\left( \text{Flatten}(\mathbf{B}_k) \cdot \mathbf{W}_{enc} + \mathbf{b}_{enc} \right)$$

Donde:
*   $\mathbf{W}_{enc} \in \mathbb{R}^{(P \times 256) \times d}$ es la matriz de proyección del codificador.
*   $\mathbf{b}_{enc} \in \mathbb{R}^{d}$ es el vector de sesgo.
*   $d$ es la dimensión de embedding del transformador principal ($n_{embd}$).

Para $P = 4$ y $d = 512$, el número de parámetros del codificador es:

$$\text{Params}_{enc} = (4 \times 256) \times 512 = 524,288 \text{ parámetros}$$

Aún en precisión media (FP16), esto representa únicamente **1.04 MB** de peso en disco, comparado con los 4.19 MB de la matriz de embeddings tradicional.

---

### B. Inferencia Toroidal y Propagación
Una vez generados los embeddings de parches $\mathbf{e}_k$, se inyectan a la topología de fase toroidal de GAJE en el campo ciclotómico $\mathbb{Q}(\zeta_{16})$, donde la onda semántica se propaga capa por capa de atención genómica:

$$\mathbf{h}_k^{(l)} = \text{GAJE-Block}\left(\mathbf{h}_k^{(l-1)}\right) \quad \text{donde } \mathbf{h}_k^{(0)} = \mathbf{e}_k$$

Como la longitud de la secuencia de parches es exactamente $T_{patch} = \frac{T_{bytes}}{P}$, la carga de cómputo en las capas de atención se reduce en un factor de:

$$\text{Aceleración} = \left(\frac{T_{bytes}}{T_{patch}}\right)^2 = P^2$$

Para un tamaño de parche $P=4$, el transformador principal procesa la atención de manera **16 veces más rápida** que un transformador de bytes tradicional, reduciendo el tamaño del KV-cache toroidal en un **75%**.

---

### C. Decodificación de Bytes (Patch Decoder)
Dado el vector de estado oculto de salida del transformador principal para el parche predicho $\mathbf{h}_{next} \in \mathbb{R}^{d}$, el decodificador local reconstruye autoregresivamente los $P$ bytes individuales del bloque.

Modelamos la probabilidad conjunta del parche de salida como el producto de las probabilidades de los bytes condicionales:

$$P(\mathbf{p}_{next} \mid \mathbf{h}_{next}) = \prod_{n=1}^{P} P(b_n \mid b_{<n}, \mathbf{h}_{next})$$

Para evitar la computación pesada de redes recurrentes (RNNs) en dispositivos móviles, se emplea un decodificador paralelo multivariante o una red feed-forward jerárquica con dependencias causales compactas:

$$\mathbf{o}_n = \text{Softmax}\left( [\mathbf{h}_{next}; \mathbf{c}_{<n}] \cdot \mathbf{W}_{dec, n} + \mathbf{b}_{dec, n} \right)$$

Donde:
*   $\mathbf{c}_{<n}$ es el contexto de los bytes previamente predichos en el mismo parche.
*   $\mathbf{W}_{dec, n}$ son sub-matrices de proyección ligeras.

---

## 4. Tabla Comparativa de Parámetros y Almacenamiento

| Métrica / Dimensión | Arquitectura BPE Tradicional (GAJE v1.0) | Arquitectura Patch-BPE (GAJE v2.0) | Beneficio / Diferencial |
| :--- | :---: | :---: | :---: |
| **Tamaño de Vocabulario ($V$)** | 32,768 | **256 (Fijo)** | Reducción de 128x en dimensionalidad de salida |
| **Peso de Embeddings en Disco** | ~8.4 MB (2-bit) | **~130 KB (2-bit)** | **~64x más ligero** |
| **Bloques de Inteligencia en 10MB** | 12 bloques | **~48 bloques** | **4x más capas de razonamiento** |
| **Soporte de Caracteres Especiales** | Limitado al vocabulario prefijado | **100% UTF-8 (Absoluto)** | Inmune a palabras desconocidas/emojis |
| **Longitud de Contexto en CPU** | $T$ tokens | $T_{bytes}/P$ parches | Aceleración matemática en atención cuadrática |

---

## 5. Próximos Pasos para su Implementación en GAJE-Flow

Para materializar esta arquitectura en futuras versiones del motor, el plan de trabajo requiere:

1.  **Módulo FFI y Lector de Bytes en Rust:**
    *   Escribir el buffer de streaming de entrada en [loader.rs](file:///home/erickaguilar/Documentos/gaje-semantic-compression/src/io/loader.rs) para convertir secuencias UTF-8 directas a bloques de bytes planos con padding dinámico en el extremo final de la secuencia.
2.  **Kernel de Proyección Lineal SIMD:**
    *   Implementar en `src/compute/kernels.rs` las rutinas de proyección aplanada (Flatten + Dot Product) usando vectorización AVX/NEON para acelerar la generación de embeddings locales en menos de 0.5 milisegundos.
3.  **Alineación del Entrenamiento por Pérdida de Fase:**
    *   Diseñar un optimizador híbrido en [trainer.rs](file:///home/erickaguilar/Documentos/gaje-semantic-compression/src/nn/trainer.rs) para guiar el aprendizaje jerárquico ajustando la pérdida de Cross-Entropy a nivel de bytes dentro de la proyección espacial del parche continuo.
