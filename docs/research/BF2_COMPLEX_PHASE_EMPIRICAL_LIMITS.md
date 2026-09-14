# 🧬 Hallazgo de Investigación: Límites Empíricos de BF2-Complex y la Invariancia del Presupuesto de Shannon

> **Fecha:** 14 de Septiembre de 2026  
> **Versión:** `GAJE Helix v1.8.0-research`  
> **Estado:** 🔬 `ARTEFACTO EXPERIMENTAL AISLADO — NO APTO PARA PRODUCCIÓN`  
> **Módulos Asociados:** [`src/compute/bf2.rs`](../../src/compute/bf2.rs), [`tests/test_bf2_complex_phase.rs`](../../tests/test_bf2_complex_phase.rs)  
> **Decisión Arquitectónica:** Archivar como micro-kernel de investigación; priorizar formalmente **Q4_0** como estándar de producción soberana.

---

## 1. Contexto y Objetivos del Experimento

El experimento **BF2-Complex** exploró la hipótesis de proyectar la cuantización de 2 bits sobre el plano complejo $\mathbb{C}$ utilizando las cuatro bases canónicas del ADN como fases ortogonales en el círculo unitario:
$$A = e^{i \pi/4}, \quad C = e^{i 3\pi/4}, \quad G = e^{i 5\pi/4}, \quad T = e^{i 7\pi/4}$$

Se implementó un micro-kernel aritmético en Rust para evaluar dos propiedades teóricas:
1. **Inferencia Zero-Multiplier:** Reducción de la multiplicación compleja $(\mathbf{w}_r + i\mathbf{w}_i)(x_r + ix_i)$ a sumas e inversiones de signo a partir de $s = x_r + x_i$ y $d = x_r - x_i$.
2. **Proyección por Densidad de Born:** Reemplazo de la exponencial de Softmax ($\exp(z)$) por la norma euclidiana al cuadrado ($|\Psi|^2 / \sum |\Psi|^2$).

---

## 2. Resultados Empíricos Positivos (Metal de Rust en ARM64)

Las pruebas unitarias y benchmarks en modo `--release` sobre un procesador móvil ARM64 certificaron:

1. **Identidad Algebraica Exacta:**
   * La reparametrización con sumas y diferencias $(s, d)$ arrojó una discrepancia máxima de **$0.0 \times 10^0$** frente a la multiplicación compleja formal en los 4 cuadrantes.
2. **Huella Física de Memoria:**
   * Una matriz de $1024 \times 512$ (524,288 pesos complejos) se comprimió de $4,096\text{ KB}$ (FP32) a **$192\text{ KB}$** (BF2 con escalas por bloque de 32), alcanzando una tasa de reducción de **`21.33x`**.
3. **Invariancia de Fase en ModReLU:**
   * `mod_relu` demostró que la fase angular $\theta = \text{atan2}(y_i, y_r)$ se preserva bit a bit mientras se atenúan magnitudes espurias por debajo del umbral de ruido.
4. **Throughput Aislado:**
   * GEMV en capa individual: $1.11\text{ ms} \to 876\,\mu\text{s}$ (**$1.27\text{x}$ de aceleración**).
   * Proyección de Born en vocabulario (49K tokens): $2.67\text{ ms} \to 492\,\mu\text{s}$ (**$5.42\text{x}$ más rápida**).

---

## 3. El Problema Estructural: La Ley de Shannon en el Plano Complejo

A pesar del rigor algebraico del micro-kernel, el esquema adolece del **mismo límite fundamental que condujo al colapso de Q2_0 y Q3_0**:

### A. Invariancia del Presupuesto de Información
Cuatro fases sobre el círculo unitario representan exactamente:
$$H = \log_2(4) = 2.0\text{ bits por peso complejo}$$

* El problema de la cuantización extrema no es la geometría de la constelación (recta real vs. círculo en $\mathbb{C}$), sino la **capacidad física del canal de información**.
* En transformers profundos (24–28 capas), la distorsión acumulada en los productos punto se amplifica exponencialmente a través de los mecanismos de atención.
* La evidencia empírica previa en GAJE estableció que:
  * **Q2_0 puro (2 bits):** $S_c \approx 0.180 - 0.235$ *(ruido ininteligible / gibberish)*.
  * **Q3_0 puro (3 bits):** $S_c \approx 0.380$ *(colapso sintáctico en modelo completo)*.
  * **Q4_0 (4 bits):** $S_c = 1.0000$ *(coherencia semántica íntegra)*.
* Rotar la constelación a $\mathbb{C}$ **no añade grados de libertad representacionales**. Sin bits adicionales, el modelo inevitablemente reproducirá el colapso de relación señal-ruido (SNR) de Q2_0.

### B. El Peligro Semántico de la Regla de Born
La aceleración de $5.42\text{x}$ al eliminar $\exp(z)$ genera dos efectos adversos no triviales:
1. **Pérdida de Contraste (Aplanamiento de Entropía):**
   * En Softmax: $\frac{e^{12}}{e^8} \approx 54.6$ *(decisión de token nítida y contrastada)*.
   * En Born: $\frac{12^2}{8^2} = 2.25$ *(distribución degenerada, selección difusa de tokens distractores)*.
2. **Inversión por Paridad Simétrica:**
   * $|\Psi|^2$ es una función par: $(-10.0)^2 = (+10.0)^2 = 100$.
   * Un logit fuertemente penalizado por el modelo (ej. $-10.0$) recibe la misma probabilidad que un logit óptimo ($+10.0$).
   * Sin un entrenamiento desde cero formulado enteramente bajo la física de amplitudes cuánticas, aplicar Born sobre logits entrenados bajo Boltzmann/Softmax destruye la coherencia de generación.

### C. Dilución de la Ganancia en el Modelo Completo
* El salto de $1.27\text{x}$ medido en una capa FFN aislada ($1024 \times 512$) se diluye drásticamente en el paso autorregresivo total frente a las latencias de lectura del KV-Cache, las operaciones de RMSNorm, RoPE y el bucle de muestreo.

---

## 4. Veredicto y Directriz Operativa (Camino 1: Consolidación)

1. **Conservación como Artefacto:**  
   [`src/compute/bf2.rs`](../../src/compute/bf2.rs) se conserva como micro-kernel experimental aislado y documentado, sin integración en el pipeline de exportación ni en las cabeceras `FlatHeaderV2`.
2. **Prioridad Soberana en Q4_0:**  
   El desarrollo y la optimización para silicio de borde (móviles, navegadores WASM y microservidores) se concentran en **Q4_0**, formato ya certificado empíricamente con $S_c = 1.0000$, paridad lingüística bilingüe y compatibilidad probada.
3. **Cierre de la Línea Sub-4 Bits sin Superposición:**  
   No se invertirá tiempo de ingeniería en desarrollar exportadores para BF2-Complex a menos que se demuestre experimentalmente una ventaja en la amplificación de error angular ($S_c$) en capas reales.
