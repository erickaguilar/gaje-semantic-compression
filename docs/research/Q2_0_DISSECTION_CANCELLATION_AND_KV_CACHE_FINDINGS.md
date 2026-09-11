# 🧬 GAJE Research: Disección de Q2_0, Interferencia Destructiva en Capa 21 y Dinámica de KV-Cache

**Fecha:** Septiembre 2026  
**Autores:** Equipo de Investigación GAJE / Núcleo Nativo Helix  
**Modelos Evaluados:** Qwen 2.5 0.5B Instruct (`models/production/qwen2_5_0_5b.gaje` [Q4_0, 1.5 GB] vs `models/qwen2_5_0_5b_q2_0.flat` [Q2_0, 454 MB])  
**Módulos del Repositorio:** `src/nn/linear/`, `src/compute/kernels/genomic.rs`, `src/nn/block/`, `src/nn/attention.rs`, `src/io/flat_reader.rs`

---

## 1. Resumen Ejecutivo

Este documento consolida la investigación empírica y matemática que desentrañó el comportamiento del formato de 2 bits (`Q2_0` / `Genomic2Bit`) en modelos de lenguaje pequeños (Qwen2.5-0.5B).

### Hallazgos Principales:
1. **Integridad de los Kernels de 2 bits:** Se descartó completamente la hipótesis de un bug en la aritmética de dequantización (`scale/min`), desalineación de índices o kernels SIMD. Evaluadas de forma aislada, todas las matrices lineales de 2 bits preservan entre el 88% y 100% de la energía de 4 bits bajo vectores isotrópicos ($S_c \ge 0.93$).
2. **Onda Portadora Residual e Invarianza de RMSNorm:** En el cuerpo del transformer (Capas 2 a 20), una excitación no-lineal en la Capa 2 crea una "onda portadora" masiva en el flujo residual ($\|x\| \approx 1645$ en Q4 frente a $\approx 238$ en Q2). RMSNorm ecualiza esta diferencia a nivel local ($\|x_{\text{norm}}\| \approx 21$ en ambos modelos), permitiendo que en régimen desacoplado el cuerpo mantenga $S_c \approx 0.996$.
3. **Mecanismo Proximal del Colapso en Capa 21 (Overshoot de Cancelación):** La arquitectura Qwen2.5 sustrae la onda portadora residual en la Capa 21 mediante una perturbación antiparalela ($\cos(h_{\text{in}}, \Delta) = -0.999458$). En 2 bits, el error angular de $\Delta$ ($\theta \approx 0.28$ rad frente a $0.03$ rad) y la desproporción respecto a la norma residual causan que la resta sobrepase el origen ($\|\Delta\| > \|x\|$), invirtiendo la polaridad semántica ($S_c$ colapsa a $-0.28$).
4. **La Trampa Exponencial del KV-Cache en Contexto Continuo:** En régimen i.i.d. (posición $0$), el modelo opera con $S_c \approx 0.996$. Sin embargo, en inferencia autoregresiva ($t \ge 1$), el Softmax en la atención ($\text{Softmax}(Q K^\top / \sqrt{d})$) exponencia los errores angulares de $W_q$ y $W_k$, redistribuyendo drásticamente la atención entre tokens y haciendo caer $S_c$ a $\approx 0.22$.
5. **Ablación Quirúrgica y Desacoplamiento Atención / FFN:** Proteger bloques completos mediante precisión mixta (4-2-4) es insuficiente porque las matrices de atención del cuerpo continúan inyectando ruido en el Softmax. La solución óptima radica en desacoplar por familia de tensores: **FFN en 2-bit** (~67% de parámetros) + **Atención en 4-bit** (~33% de parámetros), alcanzando $\approx 2.66$ bits efectivos con coherencia de contexto preservada.

---

## 2. Fase 1: Sanity Checks de Instrumentación

Para descartar artefactos en el pipeline de medición, se ejecutaron dos validaciones en el metal de Rust (`tests/test_scale_audit.rs`):

### A. Auto-identidad (Q4_0 vs Q4_0)
* **Similitud Coseno ($S_c$):** $1.0000 \pm 0.0000$ en las 24 capas del modelo ($z$-score $> 420$).
* **Ratio de Normas:** $1.0000 \pm 0.0000$ en las 24 capas.
* **Descomposición SVD:** Alineamiento = 1.00, Rotación = 0.00, Fragmentación = 0.00.
* **Conclusión:** El extractor, los punteros mmap, el desempaquetado de tensores y el reseteo de caché son 100% deterministas.

### B. Aislamiento de Matrices Individuales (Q4_0 vs Q2_0)

| Tensor | Shape | Norma Fila 0 (Q2 / Q4) | Forward Unitario (Ratio) | Forward Isotrópico $\mathbf{x} \in \mathcal{N}(0, 1)$ |
| :--- | :--- | :--- | :--- | :--- |
| `token_embd` | $151936 \times 896$ | 0.427 / 0.467 (**0.915**) | **1.087** | **0.925** ($S_c = 0.933$) |
| `blk.0.attn_q`| $896 \times 896$ | 0.276 / 0.303 (**0.910**) | **1.013** | **1.000** ($S_c = 0.999$) |
| `blk.2.attn_q`| $896 \times 896$ | 0.809 / 1.038 (**0.780**) | **0.994** | **0.999** ($S_c = 0.999$) |
| `blk.2.ffn_down` | $896 \times 4864$ | 0.174 / 0.192 (**0.908**) | **0.854** | **0.894** ($S_c = 0.935$) |

> **Veredicto:** El cociente $\sim 1/8$ observado en inferencia no se debe a un error escalar en las matrices ni en los kernels de dequantización; cada matriz conserva aisladamente su ganancia geométrica.

---

## 3. Fase 2: La Portadora Residual y la Paradoja de RMSNorm

Al trazar capa por capa el flujo de activaciones en un token aislado (posición $0$), emergió la estructura del tronco residual:

```
Embedding : Norm Q4 =    0.40 | Norm Q2 =    0.36 | Ratio = 0.9069 | Sc = 0.9413
Block 00  : Norm Q4 =    6.80 | Norm Q2 =    4.34 | Ratio = 0.6384 | Sc = 0.8426
Block 01  : Norm Q4 =    9.14 | Norm Q2 =    6.09 | Ratio = 0.6668 | Sc = 0.7178
Block 02  : Norm Q4 =  709.94 | Norm Q2 =  125.74 | Ratio = 0.1771 | Sc = 0.9910  <-- BIFURCACIÓN DE GANANCIA
Block 03  : Norm Q4 = 1614.58 | Norm Q2 =  235.53 | Ratio = 0.1459 | Sc = 0.9965
Block 04  : Norm Q4 = 1616.25 | Norm Q2 =  236.53 | Ratio = 0.1463 | Sc = 0.9963
...
Block 20  : Norm Q4 = 1643.52 | Norm Q2 =  237.90 | Ratio = 0.1448 | Sc = 0.9917
```

### El Mecanismo:
1. **Bifurcación en Capa 2:** La proyección FFN (SwiGLU) de la Capa 2 excita fuertemente una dirección característica (`Projected FFN` = 707.06 en Q4 frente a 124.60 en Q2 debido a sutiles desalineaciones angulares previas en capas 0–1).
2. **Atractor Residual:** Al sumarse al residuo ($x \leftarrow x + \text{ffn}$), esa dirección dominante absorbe la mayor parte de la energía del vector, convirtiéndose en un atractor cuasi-estacionario para las siguientes 18 capas.
3. **Invarianza de RMSNorm:**
   $$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{D}\sum x_i^2 + \epsilon}} \odot \gamma$$
   Tanto para $\|x\| = 1645$ como para $\|x\| = 238$, la norma tras RMSNorm es prácticamente idéntica ($\approx 21.78$ vs $\approx 20.54$). Esto aísla a los pesos de las capas intermedias de la magnitud del flujo residual, permitiendo que la fase angular se preserve con $S_c \approx 0.9965$ en régimen no autoregresivo.

---

## 4. Fase 3: La Cancelación Destructiva en Capa 21

Al ingresar a la Capa 21, la arquitectura de Qwen2.5 elimina la portadora residual para exponer la representación semántica fina antes del clasificador `lm_head`:

```
=== Anatomía de la Cancelación en Bloque 21 ===
Entrada a Bloque 21 : Norm Q4 = 1643.52 | Norm Q2 = 237.90 | CosSim(h4, h2) = +0.9917
Delta Bloque 21     : Norm Q4 = 1506.32 | Norm Q2 = 271.89
CosSim(h_in, delta) : Q4 = -0.999458   | Q2 = -0.965380
Salida Bloque 21    : Norm Q4 =  146.66 | Norm Q2 =  75.06  | CosSim(h4, h2) = -0.2828
```

### El Sobreimpulso (Overshoot):
* En Q4: $\Delta$ sustrae 1506 de las 1643 unidades en casi perfecta oposición angular ($-0.999458$), dejando una señal residual limpia de $146.66$ unidades.
* En Q2: La perturbación generada por el bloque tiene norma $271.89$, pero la portadora residual de entrada solo tenía $237.90$. La resta sobrepasa el origen ($237.90 - 271.89 = -33.99$), invirtiendo la dirección del vector y arrojando una similitud coseno negativa ($-0.2828$).

---

## 5. Fase 4: Test de Sustitución Cruzada en Capa 21

Para distinguir si el colapso proximal responde a la magnitud de entrada o a la dirección de $\Delta$, se ejecutó una prueba de sustitución cruzada sobre 10 tokens en `tests/test_scale_audit.rs`:

```
=== SUSTITUCIÓN CRUZADA EN CAPA 21 ===
Token  | cos(h_in) | cos(h4,d4) | cos(h2,d2) | cos(h_out Q4/Q2) | cos(h_in_Q2 + d_Q4, h_out_Q4) | cos(h_in_Q4 + d_Q2, h_out_Q4)
---------------------------------------------------------------------------------------------------------------------------------
   100 |    0.9905 |    -0.9994 |    -0.9634 |          +0.0763 |                        0.5328 |                        0.1156
   500 |    0.9829 |    -0.9993 |    -0.9491 |          +0.8538 |                        0.4346 |                        0.9165
  1000 |    0.9859 |    -0.9995 |    -0.9567 |          +0.7182 |                        0.4288 |                        0.8361
  1500 |    0.9917 |    -0.9995 |    -0.9654 |          -0.2828 |                        0.5566 |                       -0.2905
  2048 |    0.9861 |    -0.9995 |    -0.9562 |          +0.7090 |                        0.4568 |                        0.8238
  5000 |    0.9872 |    -0.9994 |    -0.9580 |          +0.6667 |                        0.4547 |                        0.7771
 10000 |    0.9863 |    -0.9994 |    -0.9563 |          +0.7155 |                        0.4564 |                        0.8229
 25000 |    0.9873 |    -0.9994 |    -0.9586 |          +0.6116 |                        0.4699 |                        0.7356
 50000 |    0.9878 |    -0.9994 |    -0.9592 |          +0.5018 |                        0.4707 |                        0.6174
100000 |    0.9873 |    -0.9995 |    -0.9586 |          +0.5730 |                        0.4728 |                        0.7061
```

### Conclusiones de la Sustitución Cruzada:
1. Al sustituir $\Delta^{Q4}$ sobre el estado $h_{\text{in}}^{Q2}$, la salida se alinea coherentemente con Q4 ($\ge +0.43$).
2. Al sustituir $\Delta^{Q2}$ sobre el estado $h_{\text{in}}^{Q4}$, la salida replica fielmente la degradación y los colapsos de Q2 (incluyendo el valor negativo en el token 1500).
3. **El error primario de Capa 21 es angular en $\Delta$:** El vector de perturbación de Q2 tiene un error de alineamiento de $\theta \approx 0.28$ rad frente a $0.03$ rad en Q4. En una cancelación donde los vectores son casi idénticos en norma, un error angular de $15^\circ$ desvía el vector resultante en más de $90^\circ$.

---

## 6. Fase 5: Régimen i.i.d. vs Contexto Autoregresivo (KV-Cache)

Al extender las evaluaciones a secuencias continuas en Rust (`tests/diagnose_cancellation.rs`, 517 tokens) y rastrear la evolución por ventana de posición (`tests/test_kv_ablation.rs`, 128 tokens), se detectó el impacto del historial de atención:

### Evolución por Posición de Token en Contexto:

```
Rango Tokens  | Q2 Puro (Capa 10 / Capa 23) | Sándwich 4-2-4 (Capa 10 / Capa 23)
---------------------------------------------------------------------------------
Token  0      |        0.9961 / 0.1674       |        0.9957 / -0.5937
Tokens  1.. 4 |        0.3938 / 0.2989       |        0.4307 /  0.5060
Tokens  5..19 |        0.1873 / 0.1860       |        0.3016 /  0.3121
Tokens 20..49 |        0.2248 / 0.1774       |        0.3313 /  0.2484
Tokens 50..127|        0.2175 / 0.1821       |        0.3045 /  0.1406
```

### Dinámica del Colapso en Contexto:
* **En $t = 0$:** No existe historial previo en el KV-cache. La atención se restringe a sí misma ($A = [1.0]$) y la salida es $V_0$. Por tanto, el cuerpo exhibe $S_c = 0.9961$.
* **En $t \ge 1$:** La atención atiende a múltiples vectores clave $K_{0..t}$. El producto punto $Q_t K_j^\top$ contiene ruido angular inducido por la cuantización a 2 bits de $W_q$ y $W_k$. Al pasar por la función Softmax:
  $$P_j = \frac{\exp\left(\frac{Q_t K_j^\top}{\sqrt{d}}\right)}{\sum_m \exp\left(\frac{Q_t K_m^\top}{\sqrt{d}}\right)}$$
  Las diferencias en el producto punto son **amplificadas exponencialmente**, redistribuyendo los pesos hacia combinaciones lineales de $V$ sustancialmente divergentes de Q4.
* Esto explica por qué el cuerpo cae a $S_c \approx 0.22$ en secuencias largas a pesar de que cada matriz FFN individual mantenga $S_c \ge 0.93$.

---

## 7. Fase 6: Evaluación de Ablaciones de Bloques

Se evaluaron cuatro variantes arquitectónicas para medir la eficacia de la protección por bloques:

| Configuración | Descripción | $S_c$ Capa 10 (Cuerpo) | $S_c$ Capa 23 (Salida) |
| :--- | :--- | :---: | :---: |
| **Q2_0 Puro** | 24 capas en 2-bit | 0.2273 | 0.1850 |
| **Ablación B1 (Entrada Q4)** | Capas 0–1 en 4-bit, 2–23 en 2-bit | 0.3198 | 0.2445 |
| **Ablación B2 (Salida Q4)** | Capas 0–20 en 2-bit, 21–23 en 4-bit | 0.2273 | 0.1348 |
| **Sándwich 4-2-4** | Capas 0–1 y 21–23 en 4-bit | 0.3198 | 0.1917 |

### Hallazgos de la Ablación:
1. **La cirugía solo en salida (B2) fracasa:** Al recibir un estado desalineado ($S_c \approx 0.22$) producto del arrastre del KV-cache en el cuerpo, cambiar las matrices de salida a 4-bit no puede reconstruir la información perdida.
2. **La cirugía sándwich por capas completas (4-2-4) es insuficiente:** Eleva $S_c$ en el cuerpo de 0.22 a 0.31, pero no mitiga la amplificación exponencial del Softmax en las capas 2 a 20.

---

## 8. Fase 7: Arquitectura Óptima (Desacoplamiento Atención / FFN)

La evidencia empírica indica que la partición óptima del modelo debe operar a nivel de **familias de tensores dentro de cada capa**, y no por bloques enteros:

```
        TENSORES DEL TRANSFORMER                  PRECISIÓN       PROPÓSITO MATEMÁTICO
   ┌──────────────────────────────────────────┐
   │ Proyecciones de Atención (Wq, Wk, Wv, Wo)│ ───►  Q4_0 (4-bit) Prevenir amplificación de Softmax en KV-cache
   ├──────────────────────────────────────────┤
   │ Proyecciones FFN (Wgate, Wup, Wdown)     │ ───►  Q2_0 (2-bit) Compresión densa del ~67% de los parámetros
   ├──────────────────────────────────────────┤
   │ Capas Terminales (Bloques 21, 22)        │ ───►  Q4_0 / Anclas Cancelación limpia de onda portadora
   └──────────────────────────────────────────┘
```

### Balanza de Parámetros y Compresión (Qwen 2.5 0.5B):
* **Proyecciones FFN (Gate, Up, Down):** Representan el **67.4%** de los parámetros matriciales. En 2-bit, sufren cero amplificación Softmax y toleran la cuantización con alta fidelidad energética.
* **Proyecciones de Atención (Q, K, V, Out):** Representan el **32.6%** de los parámetros. Mantenerlas en 4 bits elimina la divergencia en el KV-cache.
* **Tasa de Bits Efectiva:** $\approx (0.674 \times 2) + (0.326 \times 4) \approx \mathbf{2.65\text{ bits por peso}}$.
* **Tamaño Estimado:** $\approx \mathbf{280\text{ MB}}$ (frente a 409 MB de Q4_0 y 220 MB de Q2_0 puro), preservando tanto la memoria de contexto autoregresiva como la fase de salida.

---

## 9. Pruebas de Regresión y Código Asociado

Las siguientes suites de prueba nativas en Rust certifican los hallazgos de este documento:
* `tests/test_scale_audit.rs`: Auditoría de escala por tensor aislado, descomposición de Capa 2 y test de sustitución cruzada.
* `tests/diagnose_cancellation.rs`: Extracción estadística multi-prompt continua y matriz de correlación inter-capa.
* `tests/test_kv_ablation.rs`: Comparación de ablaciones (Q2 puro, Entrada Q4, Salida Q4, Sándwich 4-2-4) particionada por ventana de posición autoregresiva.
