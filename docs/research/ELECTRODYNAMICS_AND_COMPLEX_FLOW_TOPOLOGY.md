# ⚡ Modelado de Flujo Potencial y Topología de Campos Complejos en GAJE Helix

**Estado:** Documento de Investigación y Fundamentación Matemática Teórico-Operativa  
**Fecha:** 2026-08-29  
**Autores:** Erick Aguilar & Antigravity (Google DeepMind Pair Programming)

---

## 1. Fundamentación en Análisis Complejo

El flujo de información y gradientes dentro de la arquitectura de **GAJE (Genetic Adaptive Joint Embedding / DNA Semantic Compression)** se modela como un campo de flujo potencial a través del plano complejo extendido $\hat{\mathbb{C}}$ mediante el potencial complejo:

$$\Omega(z) = \Phi(x, y) + i \Psi(x, y)$$

donde:
* **$\Phi(x, y)$ (Potencial armónico / Voltaje):** Representa la señal semántica acumulada (norma de activación y energía del gradiente).
* **$\Psi(x, y)$ (Función de corriente / Líneas de flujo):** Representa las líneas de campo por donde viaja la información desde los embeddings hasta el vocabulario.
* **Velocidad del flujo:** $V(z) = \overline{\Omega'(z)} = \frac{\partial \Phi}{\partial x} - i \frac{\partial \Phi}{\partial y}$.

---

## 2. Síntesis del Mapeo Teórico-Operativo en GAJE

| Módulo GAJE | Formulación en Análisis Complejo / Campo | Efecto en la Dinámica del Sistema | Métrica de Control / Implementación |
| :--- | :--- | :--- | :--- |
| **Flujo Residual $\mathbf{x} + f(\mathbf{x})$** | Potencial holomorfo $\Omega(z) = \Phi + i\Psi$ en $\hat{\mathbb{C}}$ | Conducción laminar sin vórtices ni disipación de energía semántica. | Preservación del coseno de similitud local ($\cos \theta$). |
| **Jacobiano RMSNorm** | Proyección conforme $z/\vert{}z\vert{}$ sin polo de orden 2 | Elimina la singularidad artificial $\frac{1}{\text{RMS}^3}$ en la derivada. | Supresión de gradientes explosivos en el backward pass. |
| **Layer-Wise Decay (Últimos 8 blk)** | Red de adaptación de impedancias (Filtro Butterworth/Chebyshev) | Mitiga la reflexión de onda estacionaria ($S_{11} \approx 1$) entre el dieléctrico $Q4\_0$ y el sumidero $\vert{}V\vert{}$. | Decaimiento $\text{lr}_b = \text{lr} \cdot \text{decay}^{(n-1-b)}$ con $\text{decay} \approx 0.8$. |
| **Inhibición Lateral K-WTA** | Supresor de corrientes parásitas (*Eddy Currents*) | Fuerza esparsidad estricta anulando componentes ruidosos sub-umbral. | Proyección top-$K$ en activaciones intermedias. |
| **Memoria `.gmem` Zero-Copy** | Variedad toroidal compacta $T^2 = S^1 \times S^1$ ($\partial \Omega = \emptyset$) | Recirculación periódica sin dispersión de frontera ni fricción de I/O. | Estructura en anillo contiguo indexado por punteros mmap. |
| **`ForwardCache`** | Desacoplador capacitivo puro | Preserva la fase del fasor $\mathbf{g} = \vert{}\mathbf{g}\vert{}e^{i\theta}$, evitando desalineación en el backward. | Inmutabilidad de tensores intermedios durante el paso retrógrado. |

---

## 3. Coeficiente de Reflexión de Gradiente ($\Gamma_b$)

Para garantizar la estabilidad laminar y prevenir la transición al régimen turbulento (colapso a `NaN` o descalibración de centroides discretos), se define el **Coeficiente de Reflexión de Gradiente** por bloque:

$$\Gamma_b = \frac{\Vert{}\nabla \mathbf{x}_{\text{in}}^{(b)}\Vert{}_2}{\Vert{}\nabla \mathbf{x}_{\text{out}}^{(b)}\Vert{}_2}$$

### Heurística Operativa en Rust:
```rust
// Heurística de estabilidad laminar en el backward pass (src/nn/llm/forward.rs)
let gamma = grad_x_in_norm / (grad_x_out_norm + 1e-8);
if gamma > 1.20 {
    // Atenuación adaptativa para evitar régimen turbulento
    current_lr *= 0.85;
}
```

* **$\Gamma_b \approx 1.0$:** Acoplamiento de impedancias perfecto (flujo laminar conservativo).
* **$\Gamma_b \gg 1.0$:** Discontinuidad en centroides discretos $Q4\_0$ reflejando energía hacia capas tempranas (requiere estrangulamiento de tasa de aprendizaje local).

---

## 4. Conclusión

Este marco unifica la termodinámica de la información, el análisis complejo y la computación neuronal soberana, formalizando el porqué de la convergencia y robustez observada empíricamente en el motor **GAJE Helix**.
