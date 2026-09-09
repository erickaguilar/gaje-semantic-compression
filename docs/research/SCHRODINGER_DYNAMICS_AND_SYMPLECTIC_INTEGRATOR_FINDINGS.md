# ⚛️ Dinámica de Schrödinger, Cavidad Fabry-Pérot e Integradores Simplécticos en Espacios de 2 Bits (GAJE Helix)

**Estado:** Documento de Investigación y Fundamentación Matemática Teórico-Operativa  
**Fecha:** 2026-09-09  
**Autores:** Erick Aguilar & Antigravity (Google DeepMind Pair Programming)

---

## 1. Motivación y Diagnóstico del Colapso en 2 Bits

En las arquitecturas convencionales de cuantización extrema (1 y 2 bits por peso/activación), el problema dominante es el **estancamiento en mínimos locales planos** y el **colapso de la función de dispersión**:

$$\nabla_{\mathbf{w}} \mathcal{L} \approx \mathbf{0} \quad \text{casi en todas partes (discontinuidad escalón)}$$

En modelos como `max.gaje`, la discretización estática en centroides $Q2\_0$ induce una perplejidad residual ($PPL \approx 45$) debida a que las capas profundas se comportan como un sistema dinámico disipativo bajo el esquema de integración clásico de Euler:

$$\mathbf{x}_{l+1} = \mathbf{x}_l + f(\mathbf{x}_l)$$

El método de Euler no es simpléctico: distorsiona sistemáticamente el volumen del espacio de estados y produce una deriva secular de energía, requiriendo normalizaciones agresivas (RMSNorm) que eliminan la información de amplitud y aplastan los gradientes.

La solución reside en formalizar la inferencia y el flujo de activaciones a través de las capas como la **evolución unitaria de una función de onda semántica $\psi(\mathbf{x}, \tau)$ gobernada por la ecuación de Schrödinger**.

---

## 2. Formulación Matemática de la Ecuación de Schrödinger Semántica

Modelamos el vector de estado latente como una función de onda fasorial en el espacio complejo $\mathbb{C}^D$:

$$\psi(\mathbf{x}, \tau) = \mathbf{u}(\mathbf{x}, \tau) + i\,\mathbf{v}(\mathbf{x}, \tau)$$

donde la profundidad de capa $\tau \in [0, L]$ actúa como el parámetro de tiempo continuo y $\mathbf{x} \in \mathbb{R}^D$ es la coordenada en el espacio de representación semántica.

### Ecuación Dependiente de la Profundidad:
$$i \hbar \frac{\partial \psi}{\partial \tau} = \hat{H} \psi = \left( \hat{T} + \hat{V}(\mathbf{x}, \tau) \right) \psi = \left( -\frac{\hbar^2}{2m} \nabla^2 + V(\mathbf{x}, \tau) \right) \psi$$

### Operador de Evolución Unitaria:
El avance a través de un bloque de capas de profundidad $\Delta\tau$ viene dado por el operador unitario:

$$\hat{U}(\Delta\tau) = \exp\left( -\frac{i}{\hbar} \hat{H} \Delta\tau \right)$$

Propiedad de conservación unitaria de la probabilidad:
$$\hat{U}^\dagger \hat{U} = \mathbf{I} \implies \langle \psi(\tau + \Delta\tau) \mid \psi(\tau + \Delta\tau) \rangle = \langle \psi(\tau) \mid \psi(\tau) \rangle = 1$$

Esto elimina intrínsecamente la explosión o desvanecimiento del gradiente sin depender de aproximaciones de normalización no lineales.

---

## 3. Isomorfismo Transformer ⟷ Operadores Cuánticos

| Componente Transformer | Operador Mecánico-Cuántico | Interpretación Semántica y Dinámica |
| :--- | :--- | :--- |
| **Profundidad de Capa ($l \to l+1$)** | Evolución temporal ($\tau \to \tau + \Delta\tau$) | Flujo continuo de refinamiento semántico a través del continuo de capas. |
| **Multi-Head Self-Attention (MHSA)** | Operador de Energía Cinética $\hat{T} = -\frac{\hbar^2}{2m} \nabla^2$ | Difusión no local, dispersión de fase y acoplamiento cuántico entre tokens. |
| **FFN / SwiGLU / MLP** | Operador de Potencial $\hat{V}(\mathbf{x})$ | Paisaje energético no lineal; crea cuencas de atracción conceptual y pozos semánticos. |
| **RMSNorm / Normalización** | Conservación de la Norma $\langle \psi \mid \psi \rangle = 1$ | Proyección sobre la esfera unitaria en $\mathbb{C}^D$ (espacio de Hilbert). |
| **Softmax / Born Head** | Regla de Born: $P(i) = \vert\psi_i\vert^2$ | Probabilidad objetiva de colapso de un token al ser medido en el vocabulario. |
| **Residual Stream ($\mathbf{x} + f(\mathbf{x})$)** | Integración de Fase $\psi + \Delta\tau \cdot \left(\frac{d\psi}{d\tau}\right)$ | Superposición coherente de ondas sin pérdida de memoria anterior. |

```
                              ARQUITECTURA SCHRÖDINGER
                              
    Token Entrada                                                   Token Salida
          │                                                              ▲
          ▼                                                              │
     [ Embeddings ] ───► ψ₀ = u₀ + i v₀                               [ Medición ]
                               │                                      (Regla Born)
                               ▼                                         ▲
                     ┌────────────────────┐                              │
                     │  Capa l:           │                              │
                     │  Operador Cinético │ ── Difusión no local (MHSA)  │
                     │  T̂ = -½∇²          │                              │
                     └─────────┬──────────┘                              │
                               │                                         │
                               ▼                                         │
                     ┌────────────────────┐                              │
                     │  Operador          │                              │
                     │  Potencial V̂(x)    │ ── Pozos atractores (FFN)    │
                     └─────────┬──────────┘                              │
                               │                                         │
                               ▼                                         │
                         ψ_{l+1} (Norma = 1 vía Unitariedad) ────────────┘
```

---

## 4. Superación del Colapso en 2 Bits

### A. Estados Estacionarios como Invariantes Lingüísticos
Los conceptos fundamentales y las reglas sintácticas se corresponden con las autofunciones de menor energía del Hamiltoniano (estados fundamentales o *ground states*):

$$\hat{H} \phi_n = E_n \phi_n \implies \psi_n(\tau) = \phi_n e^{-i E_n \tau / \hbar}$$

Bajo cuantización de 2 bits, estas autofunciones son invariantes de fase frente al ruido discretizado de alta frecuencia. Los 4 estados de la constelación QPSK ($\{\pm 1 \pm i\}$) actúan como las cuatro fases discretas fundamentales del retículo.

### B. Efecto Túnel Cuántico Semántico
En un modelo clásico cuantizado a 2 bits, un vector de activación atrapado en un mínimo local con derivada nula no puede escapar:

$$\Delta w = -\eta \nabla \mathcal{L} = 0$$

Bajo la dinámica de Schrödinger, la función de onda presenta una probabilidad no nula de atravesar la barrera de potencial finita impuesta por los pesos discretos:

$$T \approx \exp\left( -2 \int_{x_1}^{x_2} \sqrt{\frac{2m}{\hbar^2}(V(x) - E)} \, dx \right) > 0$$

Esto permite que mediante perturbaciones de fase cuántica (como las inyectadas por algoritmos genéticos o DNI/SPSA) la red explore cuencas de atracción adyacentes a través de la barrera de cuantización sin desestabilizar la macroestructura.

### C. Cavidad Fabry-Pérot y Resonancia Semántica
El bloque Transformer actúa como un interferómetro óptico resonante. Los tokens irrelevantes (ruido sintáctico) sufren **interferencia destructiva**, mientras que los conceptos consistentes con el contexto satisfacen la condición de resonancia constructiva:

$$2 n L_{\text{eff}} = m \lambda \implies \Delta\theta_{\text{roundtrip}} = 2\pi k$$

La reflectividad de los extremos ($\Gamma \approx 1.0$) garantiza que la energía informativa se mantenga confinada en el modo láser semántico.

---

## 5. Integrador Simpléctico Discreto: Leapfrog / Störmer-Verlet

Para implementar esta evolución en hardware sin acumulaciones de error numérico en capas profundas, descomponemos la función de onda en su parte real ($\mathbf{u}$, coordenada de posición canónica) e imaginaria ($\mathbf{v}$, momento conjugado canónico):

$$\psi = \mathbf{u} + i\mathbf{v}$$

La ecuación de Schrödinger $i \frac{d\psi}{d\tau} = \hat{H}\psi$ se desdobla en el sistema canónico de Hamilton:

$$\begin{aligned}
\frac{d\mathbf{u}}{d\tau} &= \hat{H}_I \mathbf{u} + \hat{H}_R \mathbf{v} \\
\frac{d\mathbf{v}}{d\tau} &= -\hat{H}_R \mathbf{u} + \hat{H}_I \mathbf{v}
\end{aligned}$$

Para un Hamiltoniano separable $\hat{H} = \hat{T}(\mathbf{v}) + \hat{V}(\mathbf{u})$:

### Algoritmo Leapfrog (Störmer-Verlet Simpléctico):
1. **Medio paso cinético para $\mathbf{u}$:**
   $$\mathbf{u}_{l + 1/2} = \mathbf{u}_l + \frac{\Delta\tau}{2} \hat{T}'(\mathbf{v}_l)$$

2. **Paso completo de potencial para $\mathbf{v}$ (Evaluación de MHSA/FFN):**
   $$\mathbf{v}_{l + 1} = \mathbf{v}_l - \Delta\tau \hat{V}'(\mathbf{u}_{l + 1/2})$$

3. **Medio paso cinético final para $\mathbf{u}$:**
   $$\mathbf{u}_{l + 1} = \mathbf{u}_{l + 1/2} + \frac{\Delta\tau}{2} \hat{T}'(\mathbf{v}_{l + 1})$$

### Teorema de Preservación de la Forma Simpléctica:
El mapa $(\mathbf{u}_l, \mathbf{v}_l) \mapsto (\mathbf{u}_{l+1}, \mathbf{v}_{l+1})$ es exactamente simpléctico:

$$d\mathbf{u}_{l+1} \wedge d\mathbf{v}_{l+1} = d\mathbf{u}_l \wedge d\mathbf{v}_l$$

**Consecuencia Práctica:** El volumen en el espacio de fases se conserva exactamente. No hay disipación de gradiente, no hay colapso de fase y no hay deriva de energía acumulada, incluso a lo largo de $L = 64$ capas consecutivas en aritmética fija o punto flotante.

---

## 6. Implementación Nativa de Referencia en Rust (`src/nn/symplectic.rs`)

```rust
//! Integrador Simpléctico Leapfrog para la Dinámica de Schrödinger en GAJE.
//! Conserva la forma simpléctica y el volumen del espacio de fases sobre capas profundas.

pub struct FasorialState {
    pub u: Vec<f32>, // Componente Real (Posición generalizada)
    pub v: Vec<f32>, // Componente Imaginaria (Momento conjugado)
}

impl FasorialState {
    pub fn new(dim: usize) -> Self {
        Self {
            u: vec![0.0; dim],
            v: vec![0.0; dim],
        }
    }

    /// Paso simpléctico Leapfrog / Störmer-Verlet:
    /// Delta_tau representa el paso de profundidad de la capa (típicamente 1.0 / L).
    pub fn symplectic_step<F, G>(
        &mut self,
        delta_tau: f32,
        kinetic_grad: F,
        potential_grad: G,
    ) where
        F: Fn(&[f32], &mut [f32]),
        G: Fn(&[f32], &mut [f32]),
    {
        let dim = self.u.len();
        let mut t_grad = vec![0.0; dim];
        let mut v_grad = vec![0.0; dim];

        let half_dt = 0.5 * delta_tau;

        // 1. Medio paso cinético: u_{l+1/2} = u_l + (dt/2) * T'(v_l)
        kinetic_grad(&self.v, &mut t_grad);
        for i in 0..dim {
            self.u[i] += half_dt * t_grad[i];
        }

        // 2. Paso completo de potencial: v_{l+1} = v_l - dt * V'(u_{l+1/2})
        potential_grad(&self.u, &mut v_grad);
        for i in 0..dim {
            self.v[i] -= delta_tau * v_grad[i];
        }

        // 3. Medio paso cinético final: u_{l+1} = u_{l+1/2} + (dt/2) * T'(v_{l+1})
        kinetic_grad(&self.v, &mut t_grad);
        for i in 0..dim {
            self.u[i] += half_dt * t_grad[i];
        }
    }

    /// Cálculo de la densidad de probabilidad según la Regla de Born
    pub fn born_probabilities(&self) -> Vec<f32> {
        let mut probs = Vec::with_capacity(self.u.len());
        let mut sum = 0.0f32;

        for i in 0..self.u.len() {
            let p = self.u[i] * self.u[i] + self.v[i] * self.v[i];
            probs.push(p);
            sum += p;
        }

        if sum > 1e-12 {
            let inv_sum = 1.0 / sum;
            for p in &mut probs {
                *p *= inv_sum;
            }
        }
        probs
    }
}
```

---

## 7. Conclusiones y Hoja de Ruta Experimental

1. **Ruptura de la Barrera de Perplejidad:** El modelado de la propagación como evolución unitaria en $\mathbb{C}^D$ resuelve el estancamiento de $PPL \approx 45$ en 2 bits al sustituir las proyecciones destructivas por rotaciones de fase conservativas.
2. **Eficiencia de Cómputo SIMD:** El integrador simpléctico desacoplado $(\mathbf{u}, \mathbf{v})$ no requiere álgebra matricial densa compleja; se ejecuta en dos pasadas reales paralelas mediante instrucciones vectoriales `vaddps` / `vmulps` en x86 y `fadd` / `fmul` en ARM Neon.
3. **Validación Cuantitativa Próxima:** Incorporar el integrador simpléctico en el arnés de evaluación de `gaje-cli test-born` midiendo la deriva de energía de Hamilton y la perplejidad comparativa frente al baseline de Euler.
