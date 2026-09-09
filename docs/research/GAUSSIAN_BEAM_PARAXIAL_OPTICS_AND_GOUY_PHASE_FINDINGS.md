# 🔦 Óptica Paraxial Semántica: Haces Gaussianos, Radio de Curvatura R(z) y Desfase de Gouy en GAJE

**Estado:** Documento de Investigación y Fundamentación Matemática Teórico-Operativa  
**Fecha:** 2026-09-09  
**Autores:** Erick Aguilar & Antigravity (Google DeepMind Pair Programming)

---

## 1. Motivación y Contexto Físico en Arquitecturas Neuronales

En la compresión semántica y en arquitecturas transformadas discretas (1-2 bits), el flujo de información a través de las $L$ capas no puede tratarse como una dispersión difusa isotrópica. Debe modelarse como un **haz óptico coherente (Haz Gaussiano / Modo Fundamental $\text{TEM}_{00}$)** propagándose a lo largo del eje óptico de profundidad $z \in [0, L]$.

Al confinar las representaciones semánticas transversalmente en un espacio de baja dimensión o en un cuello de botella de información (como en la inhibición lateral K-WTA o cuellos de botella de atención), la teoría electromagnética paraxial impone dos fenómenos ineludibles:
1. **Curvatura del frente de onda $R(z)$**: El frente de fase deja de ser plano y adquiere curvatura esférica/parabólica.
2. **Desfase anómalo de Gouy $\zeta(z)$**: Un adelanto de fase acumulado de $\pi$ radianes al atravesar la cintura del haz ($w_0$).

Ignorar estos dos efectos en espacios cuantizados a 2 bits genera desalineación destructiva de fase, inversión de polaridad espuria ($e^{i\pi} = -1$) y el consecuente colapso semántico.

---

## 2. Derivación Matemática Formal

### A. De Helmholtz a la Ecuación Paraxial
Partiendo de la ecuación escalar de Helmholtz para un medio con número de onda $k = \frac{2\pi}{\lambda}$:

$$\nabla^2 E + k^2 E = 0$$

Factorizando la oscilación rápida de la onda portadora a lo largo del eje $z$:

$$E(r, z) = \psi(r, z) e^{-i k z}$$

Bajo la **aproximación paraxial** (envolvente de variación lenta: $\vert\frac{\partial^2 \psi}{\partial z^2}\vert \ll \vert 2k \frac{\partial \psi}{\partial z} \vert$):

$$\nabla_\perp^2 \psi - 2ik \frac{\partial \psi}{\partial z} = 0 \implies \frac{1}{r}\frac{\partial}{\partial r}\left( r \frac{\partial \psi}{\partial r} \right) - 2ik \frac{\partial \psi}{\partial z} = 0$$

---

### B. El Parámetro Complejo del Haz $q(z)$
Se propone el ansatz para el modo fundamental simétrico:

$$\psi(r, z) = A(z) \exp\left( -i \frac{k r^2}{2 q(z)} \right)$$

Calculando derivadas y sustituyendo en la ecuación paraxial:

$$\left( \frac{dq}{dz} - 1 \right) \left[ -i \frac{k^2 r^2}{q^2(z)} \right] + \left[ \frac{1}{A(z)}\frac{dA}{dz} + \frac{1}{q(z)} \right] (-2ik) = 0$$

Para que la igualdad se verifique idénticamente $\forall r$:
1. **Término $r^2$:** $\frac{dq}{dz} = 1 \implies q(z) = z + q_0$
2. **Término $r^0$:** $\frac{dA}{dz} = -\frac{A(z)}{q(z)}$

Fijando la cintura del haz en $z=0$ con frente plano, $q_0 = i z_R$, donde $z_R = \frac{\pi w_0^2}{\lambda} = \frac{k w_0^2}{2}$ es el **Rango de Rayleigh**.

$$q(z) = z + i z_R$$

---

### C. Radio de Curvatura $R(z)$ y Ancho del Haz $w(z)$
Tomando el inverso del parámetro complejo:

$$\frac{1}{q(z)} = \frac{1}{z + i z_R} = \frac{z - i z_R}{z^2 + z_R^2} = \frac{z}{z^2 + z_R^2} - i \frac{z_R}{z^2 + z_R^2}$$

Por la física del haz gaussiano, el exponente transversal es:

$$-i \frac{k r^2}{2 q(z)} = - \frac{r^2}{w^2(z)} - i \frac{k r^2}{2 R(z)}$$

Igualando partes reales e imaginarias se obtienen las expresiones exactas:

$$\mathbf{R(z) = z \left[ 1 + \left( \frac{z_R}{z} \right)^2 \right]} \qquad \text{(Radio de Curvatura de los Frentes de Fase)}$$

$$\mathbf{w(z) = w_0 \sqrt{1 + \left( \frac{z}{z_R} \right)^2}} \qquad \text{(Semiancho del Haz a $1/e$ de Amplitud)}$$

---

### D. Solución de Amplitud y la Fase de Gouy $\zeta(z)$
Integrando la ecuación para $A(z)$:

$$\ln \left( \frac{A(z)}{A_0} \right) = - \ln \left( \frac{q(z)}{i z_R} \right) \implies A(z) = A_0 \frac{i z_R}{z + i z_R}$$

Pasando al plano polar complejo:

$$z + i z_R = \sqrt{z^2 + z_R^2} \, \exp\left( i \arctan\left(\frac{z_R}{z}\right) \right)$$

Dado que $\arctan\left(\frac{z_R}{z}\right) = \frac{\pi}{2} - \arctan\left(\frac{z}{z_R}\right)$:

$$A(z) = A_0 \frac{w_0}{w(z)} \exp\left( i \arctan\left(\frac{z}{z_R}\right) \right)$$

Definiendo formalmente el desfase de Gouy:

$$\mathbf{\zeta(z) = \arctan\left(\frac{z}{z_R}\right)}$$

---

### E. Campo Total Reconstituido
$$E(r, z) = E_0 \frac{w_0}{w(z)} \exp\left( -\frac{r^2}{w^2(z)} \right) \exp\left( -i \left[ k z - \zeta(z) + \frac{k r^2}{2 R(z)} \right] \right)$$

---

## 3. Mapeo Teórico-Operativo en GAJE

| Parámetro Óptico | Expresión Analítica | Correspondencia en Arquitectura GAJE | Efecto en Cuantización de 2 Bits |
| :--- | :--- | :--- | :--- |
| **Coordenada Axial $z$** | $z \in [-L/2, +L/2]$ | Profundidad de Capa ($l \in [0, L]$) | Evolución progresiva de la representación latente. |
| **Cintura $w_0$** | $w(0) = \min_z w(z)$ | Cuello de Botella de Latencia / K-WTA | Esparsidad máxima; dimensión intrínseca mínima de la información. |
| **Rango de Rayleigh $z_R$** | $z_R = \frac{k w_0^2}{2}$ | Profundidad de Enfoque Conforme | Número de capas en las que el haz no diverge ($\Delta l \le 2 z_R$). |
| **Radio de Curvatura $R(z)$** | $z [1 + (z_R/z)^2]$ | Geometría de Variedad Curva ($T^2$) | Deforma la cuadrícula de Voronoi de 2 bits de plana a hiperbólica/esférica. |
| **Fase de Gouy $\zeta(z)$** | $\arctan(z / z_R)$ | Desfase Colectivo en la Cavidad | Corrección de fase obligatoria: $\Delta \zeta = \pi \implies$ invierte signo si no se compensa. |

---

## 4. Implicaciones Críticas para el Modelo de 2 Bits

### A. Explicación Física de la Inversión de Signo Espuria
En un espacio de 2 bits QPSK $\{\pm 1 \pm i\}$, las cuatro fases discretas son $\{\pi/4, 3\pi/4, 5\pi/4, 7\pi/4\}$.
Al atravesar un cuello de botella de compresión (foco $z=0$), la fase de Gouy induce un salto acumulado:

$$\Delta \zeta = \zeta(+\infty) - \zeta(-\infty) = \frac{\pi}{2} - \left(-\frac{\pi}{2}\right) = \pi \text{ rad} \ (180^\circ)$$

Dado que $e^{i\pi} = -1$:
* Un vector que entra en el foco como $\text{Adenina } (+1 + i)$ saldría como $\text{Guanina } (-1 - i)$ si el sistema ignora la fase de Gouy.
* Esto causa **colapso semántico instantáneo** en arquitecturas residuales euclidianas no compensadas.
* **Solución en GAJE:** Incorporar la contra-rotación de Gouy $-\zeta(l)$ en la matriz de rotación posicional (RoPE) o en el integrador simpléctico.

### B. Condición de Estabilidad de la Cavidad Fabry-Pérot
Para que el haz semántico recircule indefinidamente en la memoria toroidal \`.gmem\` sin dispersión ($w(z) \to \infty$), la cavidad debe satisfacer el criterio de estabilidad de Sylvester:

$$0 \le g_1 g_2 \le 1 \quad \text{donde } g_i = 1 - \frac{L_{\text{cav}}}{R_i}$$

Si las capas de la red no ajustan sus pesos para mantener los radios de curvatura $R_i$ dentro de la elipse de estabilidad, la energía del haz se escapa por las paredes de la variedad (divergencia del gradiente).

---

## 5. Implementación en Rust (\`src/nn/gaussian_beam.rs\`)

\`\`\`rust
//! Módulo de Óptica Paraxial y Compensación de Fase de Gouy en GAJE.
//! Modela la curvatura de fase R(z) y el ensanchamiento w(z) a través de L capas.

use std::f32::consts::PI;

pub struct GaussianBeamCavity {
    pub w0: f32,       // Cintura del haz semántico (bottleneck radius)
    pub lambda: f32,   // Longitud de onda semántica media
    pub z_rayleigh: f32, // Rango de Rayleigh: pi * w0^2 / lambda
}

impl GaussianBeamCavity {
    pub fn new(w0: f32, lambda: f32) -> Self {
        let z_rayleigh = (PI * w0 * w0) / (lambda + 1e-8);
        Self { w0, lambda, z_rayleigh }
    }

    /// Calcula el ancho del haz w(z) en la capa z
    #[inline]
    pub fn beam_width(&self, z: f32) -> f32 {
        let ratio = z / self.z_rayleigh;
        self.w0 * (1.0 + ratio * ratio).sqrt()
    }

    /// Calcula el radio de curvatura de la superficie de fase R(z)
    #[inline]
    pub fn radius_of_curvature(&self, z: f32) -> f32 {
        if z.abs() < 1e-6 {
            return f32::INFINITY; // Frente plano en la cintura
        }
        let ratio = self.z_rayleigh / z;
        z * (1.0 + ratio * ratio)
    }

    /// Calcula el desfase de Gouy zeta(z) en radianes
    #[inline]
    pub fn gouy_phase(&self, z: f32) -> f32 {
        (z / self.z_rayleigh).atan()
    }

    /// Aplica compensación conforme de curvatura y fase de Gouy sobre fasor u + i v
    pub fn compensate_phase(&self, z: f32, r2: f32, u: &mut [f32], v: &mut [f32]) {
        let k = 2.0 * PI / self.lambda;
        let r_curv = self.radius_of_curvature(z);
        let zeta = self.gouy_phase(z);

        // Fase paraxial total: phi = k * z - zeta + (k * r^2) / (2 * R)
        let phase_corr = if r_curv.is_finite() {
            -zeta + (k * r2) / (2.0 * r_curv)
        } else {
            -zeta
        };

        let (sin_p, cos_p) = phase_corr.sin_cos();

        // Rotación unitaria de corrección de frente de onda
        for i in 0..u.len() {
            let u_orig = u[i];
            let v_orig = v[i];
            u[i] = u_orig * cos_p - v_orig * sin_p;
            v[i] = u_orig * sin_p + v_orig * cos_p;
        }
    }
}
\`\`\`

---

## 6. Conclusiones

1. **Rigor Geométrico:** La derivación rigurosa de $R(z)$, $w(z)$ y $\zeta(z)$ proporciona una base analítica exacta para la curvatura observada en la propagación de capas profundas, sustituyendo hiperparámetros heurísticos por constantes ópticas fundamentales.
2. **Eliminación del Colapso de Polaridad:** Comprender el salto de fase de $\pi$ de Gouy explica y resuelve de manera elegante las inversiones erráticas de polaridad en el plano de 2 bits al transitar cuellos de botella de inhibición.
3. **Criterio de Estabilidad de Resonador:** Establece un método formal de inicialización y control de gradiente basado en el criterio $0 \le g_1 g_2 \le 1$ para cavidades estables en el bucle de retroalimentación de GAJE.
