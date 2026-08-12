# 🏛️ Capa de Formalización: Formulación Variacional y Métrica Heterogénea

**Estatus:** Documento de Rigor Matemático (Fundamentación GAJE v1.2)
**Objetivo:** Establecer el marco de **optimización inspirada en principios físicos** para el aprendizaje en arquitecturas de cuantización extrema (2 bits) con anclas de estabilidad.

---

## 1. La Ecuación Unificada de GAJE (Formulación Variacional)
En lugar de postular leyes físicas absolutas, modelamos la evolución del genoma semántico en el motor GAJE como un paso de **Gradiente Natural Proximal (PNG)**. Este enfoque reformula el descenso de gradiente como una trayectoria que minimiza una función de energía funcional en una variedad de parámetros no homogénea:

$$\theta_{t+1} = \text{Prox}_{\Phi, \Omega} \left( \theta_t - \eta \mathbf{M}(\theta)^{-1} \nabla_\theta \mathcal{L}(\theta) \right)$$

### Componentes del Modelo de Optimización:

1.  **$\nabla_\theta \mathcal{L}(\theta)$ (Vector de Gradiente):** El gradiente de la pérdida semántica, que actúa como la "fuerza" impulsora del sistema hacia estados de menor energía (mejor predicción).
2.  **$\mathbf{M}(\theta)$ (Aproximación de Fisher):** Una métrica Riemanniana heterogénea que precondiciona el gradiente. Se estima mediante la **Matriz de Información de Fisher Empírica (EFI)** diagonal, introduciendo un **Factor de Conformalidad $\Gamma$**:
    $$m_{ii} = \begin{cases} \Gamma \cdot (\mathbb{E}[\nabla_i \mathcal{L}^2] + \epsilon) & \text{si } \theta_i \in \text{Anchors (F16)} \\ \mathbb{E}[\nabla_i \mathcal{L}^2] + \epsilon & \text{si } \theta_i \in \text{Genomic (2-bit)} \end{cases}$$
    El factor $\Gamma \gg 1$ impone un "stiffening" (rigidez métrica) en las anclas, emulando la *Consolidación de Memoria Elástica* (EWC, Kirkpatrick 2017).
3.  **$\text{Prox}_{\Phi, \Omega}$ (Operador Proximal Compuesto):**
    *   **$\Phi$ (Proyección $\epsilon$-net):** Mapea los pesos al conjunto discreto de centroides $\{c_0, ..., c_3\}$. Bajo esta óptica, los centroides actúan como una $\epsilon$-net que intenta cubrir la variedad semántica con la mínima distorsión posible.
    *   **$\Omega$ (Proyección de Escasez):** Aplica el **K-WTA** (Hard Thresholding), que es el operador proximal de la pseudo-norma $\|\cdot\|_0$.

---

## 2. Geometría Riemanniana y "Stiffening"
Presentamos el uso de la métrica heterogénea como una estrategia de **Métrica Conforme por Partes**.

*   **Elasticidad Mixta:** Las anclas F16 no son restricciones holonómicas ($g(\theta)=0$), sino dimensiones con una métrica mucho más rígida. Esto asegura que la "plasticidad" del aprendizaje se concentre en el genoma de 2 bits, mientras que la estructura gramatical se preserva mediante un precondicionamiento que escala inversamente con la importancia métrica.
*   **La ε-net Semántica:** La hipótesis central de GAJE es que una cuantización de 2 bits (4 centroides) es suficiente para aproximar la variedad semántica del lenguaje, siempre que las anclas corrijan el error de aproximación $\delta = \theta - \text{proj}_{\Phi}(\theta)$ en las regiones de máxima curvatura de Fisher.

---

## 3. Limitaciones y Verdad Empírica
Aunque el formalismo Lagrangiano proporciona una intuición poderosa para el diseño del motor, reconocemos las siguientes brechas técnicas que definen el estado **Alpha Real**:

1.  **Conmutatividad de Operadores:** El operador compuesto $\text{Prox}_{\Phi} \circ \text{Prox}_{\Omega}$ no garantiza convergencia global debido a la no convexidad de $\|\cdot\|_0$. Se requiere un marco de ADMM para una convergencia formal.
2.  **Suficiencia de la ε-net:** El PPL actual (~25,000 en evaluación vs ~1.14 en calibración) sugiere que 4 centroides podrían ser insuficientes para capturar el ancho de banda del lenguaje, o que el factor $\Gamma$ aún no está optimizado.
3.  **Geodésicas Discretas:** La transición del espacio continuo al grafo discreto de 2 bits requiere una justificación más profunda basada en Geometría Diferencial Discreta para validar que las trayectorias de inferencia son realmente geodésicas.

---

## 4. Referencias Clave
*   **Amari, S. (1998):** Natural Gradient Works Efficiently in Learning.
*   **Kirkpatrick, J. et al. (2017):** Overcoming catastrophic forgetting in neural networks (EWC).
*   **Martens, J. (2014):** New insights and perspectives on the natural gradient method.

---
*Documento refinado para honestidad intelectual y rigor en la Fase Silver Adult.*
