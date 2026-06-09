# 🏛️ Capa de Formalización: Gradiente Natural Proximal y Métrica Heterogénea

**Estatus:** Documento de Rigor Matemático (Fundamentación GAJE v1.1)
**Objetivo:** Establecer el marco de optimización Riemanniana para el aprendizaje en arquitecturas de cuantización extrema (2 bits) con anclas de estabilidad.

---

## 1. La Ecuación Unificada de GAJE
La evolución del genoma semántico en el motor GAJE no sigue un descenso de gradiente estocástico (SGD) euclidiano, sino un paso de **Gradiente Natural Proximal (PNG)** sobre una variedad de parámetros no homogénea:

$$\theta_{t+1} = \text{Prox}_{\Phi, \Omega} \left( \theta_t - \eta \mathbf{M}(\theta)^{-1} \nabla_\theta \mathcal{L}(\theta) \right)$$

### Componentes de la Regla de Actualización:

1.  **$\nabla_\theta \mathcal{L}(\theta)$ (Vector de Gradiente):** El gradiente de la pérdida semántica con respecto a los parámetros, calculado sobre el corpus de refinamiento.
2.  **$\mathbf{M}(\theta)$ (Métrica Riemanniana de Fisher):** Una métrica heterogénea que precondiciona el gradiente. Se aproxima mediante la **Matriz de Información de Fisher Empírica (EFI)** diagonal, introduciendo un **Factor de Conformalidad $\Gamma$**:
    $$m_{ii} = \begin{cases} \Gamma \cdot (\mathbb{E}[\nabla_i \mathcal{L}^2] + \epsilon) & \text{si } \theta_i \in \text{Anchors (F16)} \\ \mathbb{E}[\nabla_i \mathcal{L}^2] + \epsilon & \text{si } \theta_i \in \text{Genomic (2-bit)} \end{cases}$$
    El factor $\Gamma \gg 1$ impone un "stiffening" (rigidez métrica) en las anclas, emulando la *Consolidación de Memoria Elástica* (EWC, Kirkpatrick 2017).
3.  **$\text{Prox}_{\Phi, \Omega}$ (Operador Proximal Compuesto):**
    *   **$\Phi$ (Proyección $\epsilon$-net):** Operador proximal que mapea los pesos al conjunto discreto de centroides $\{c_0, ..., c_3\}$ minimizando la distorsión en la métrica de Fisher local.
    *   **$\Omega$ (Proyección de Escasez):** Operador proximal de la pseudo-norma $\|\cdot\|_0$ (K-WTA), que impone la topología de activación rala.

---

## 2. Geometría Riemanniana Heterogénea
La arquitectura GAJE trata el espacio de pesos como una variedad donde la "facilidad de movimiento" (plasticidad) es inversamente proporcional a la importancia gramatical detectada.

*   **Stiffening Conforme:** Las anclas F16/F8 no son restricciones fijas ($g(\theta)=0$), sino regiones donde el tensor métrico $\mathbf{M}$ aumenta su valor, reduciendo la norma del gradiente natural en esas dimensiones ($\| \mathbf{M}^{-1} \nabla \mathcal{L} \|$).
*   **Aproximación por ε-nets:** Dado que la cuantización de 2 bits rompe el toroide continuo, modelamos el espacio discreto como una **$\epsilon$-net** óptima. Las anclas actúan como el término residual $\delta = \theta - \text{proj}_{\Phi}(\theta)$ que preserva la coherencia en zonas de alta curvatura de Fisher.

---

## 3. Convergencia y Operadores Compuestos
Dada la naturaleza no convexa del operador de escasez ($\Omega$) y la cuantización ($\Phi$), la convergencia del paso proximal compuesto se justifica bajo el marco de **ADMM (Alternating Direction Method of Multipliers)** o *Douglas-Rachford Splitting*. 

En la práctica de GAJE-Flow, esto significa que la actualización se realiza en fases alternas:
1.  Ajuste de los polos de resonancia (Centroides).
2.  Refinamiento del residuo de alta precisión (Anclas).
3.  Proyección de transparencia semántica (K-WTA).

---

## 4. Auditoría de Implementación y Deuda Técnica
Actualmente, el motor en Rust (`src/nn/linear.rs`) implementa una versión simplificada:
*   **Métrica:** SGD plano ($\mathbf{M} = \mathbf{I}$). **Acción:** Implementar estimación *lazy* de Fisher diagonal para evitar duplicar el uso de memoria en Android.
*   **Proyección:** Clamping heurístico en lugar de mapeo proximal formal. **Acción:** Refactorizar `refine_with_grads_core` para usar la métrica heterogénea calculada.
*   **Topología:** La escasez es reactiva (en el forward) pero no proactiva (en el refinamiento).

---

## 5. Referencias Clave
*   **Amari, S. (1998):** Natural Gradient Works Efficiently in Learning.
*   **Kirkpatrick, J. et al. (2017):** Overcoming catastrophic forgetting in neural networks (EWC).
*   **Martens, J. (2014):** New insights and perspectives on the natural gradient method.

---
*Este documento constituye la hoja de ruta para la certificación Nivel 2 (PPL < 15.0).*
