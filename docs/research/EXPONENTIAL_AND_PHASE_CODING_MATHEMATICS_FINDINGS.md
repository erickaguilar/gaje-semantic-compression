# 🧬 Hallazgos de Investigación: Codificación Exponencial, LNS y Dominios de Fase Compleja en Modelos de Lenguaje

> **Fecha:** 2 de Septiembre de 2026  
> **Versión del Motor:** `GAJE Helix v1.8.0 / Math Core`  
> **Ámbitos:** Sistemas Numéricos Logarítmicos (LNS) · Codificación en Fase Compleja ($\mathbb{C}$) · Aritmética sin Multiplicadores · Microscaling Tensorial  
> **Módulos Afectados:** `src/compute/math.rs`, `src/compute/bf2.rs`, `src/compute/gpu/shaders/`

---

## 1. 🎯 Fundamento Teórico

Almacenar o codificar un número de punto flotante en forma exponencial consiste en mapear el valor real $x$ al exponente de una base matemática fija (generalmente la base natural $e$ o la base binaria $2$).

Dependiendo de si se busca compresión extrema de memoria, aceleración aritmética sin multiplicadores o representación espectral/holográfica, existen **tres técnicas fundamentales**:

```
                       MAPEOS EXPONENCIALES DEL FLOTANTE x
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        ▼                               ▼                               ▼
  1. Sistema LNS                 2. Fasores / Fase               3. Microscaling
  (x → log2|x|)                  (x → e^{i θ(x)})                (x = s · 2^E)
  • Mults. pasan a sumas         • Continuidad en C              • Bloques compartidos
  • Inferencia ultra-ligera      • RoPE / Holografía             • Formatos MXFP / E8M0
```

---

## 2. 🧮 1. Sistema Numérico Logarítmico (LNS - Logarithmic Number System)

En lugar de almacenar mantisa y exponente por separado (como prescribe el estándar IEEE 754), el número se representa puramente a través de su logaritmo binario:

* **Almacenamiento:** Se almacena el signo de $x$ (1 bit) y el exponente $E = \log_2(|x|)$ como un entero con signo o un flotante de escala calibrada.
* **Propiedad Matemática Clave:** Las multiplicaciones y divisiones se transforman en **sumas y restas lineales**:

$$\log_2(A \cdot B) = \log_2(A) + \log_2(B) \implies A \cdot B = 2^{E_A + E_B}$$
$$\log_2\left(\frac{A}{B}\right) = \log_2(A) - \log_2(B) \implies \frac{A}{B} = 2^{E_A - E_B}$$

* **Impacto en Hardware:** Elimina la necesidad de unidades multiplicadoras (ALU/FMA complejas en silicio), reduciendo el consumo energético y permitiendo inferencia masiva en dispositivos móviles y microcontroladores.

---

## 3. 🌀 2. Codificación en Fase Compleja (Fasores y Dominio Holográfico)

Si el objetivo es mapear un escalar continuo $x$ a un espacio de fase periódica unitaria (como en neuronas holográficas o en Rotary Position Embeddings - RoPE), se codifica en el exponente complejo mediante la fórmula de Euler:

$$\mathbf{z}(x) = e^{i \cdot \theta(x)} = \cos(\theta(x)) + i \cdot \sin(\theta(x))$$

Donde $\theta(x) = 2\pi \cdot \text{norm}(x)$ mapea el rango del flotante a un ángulo de rotación en $[-\pi, \pi]$.

* **Propiedad Clave:** El número no se almacena como una magnitud estática en la recta real, sino como una **rotación de fase pura en el círculo unidad de $\mathbb{C}$**.
* **Interferencia y Distancia:** La atención del transformador mide diferencias angulares relativas ($\theta_q - \theta_k$) mediante interferencia constructiva/destructiva, preservando la continuidad de la "montaña semántica".

---

## 4. 🦀 3. Implementación Canónica en Rust: Cuantización Logarítmica (`LogFloat`)

Estructura para comprimir valores `f32` preservando resolución fina en valores cercanos a cero mediante un entero de 8 bits con escala sub-entera:

```rust
//! Representación logarítmica de punto flotante en 8-bits (LNS)
pub struct LogFloat {
    pub sign: bool,
    pub log_val: i8, // Exponente cuantizado en base 2 con escala
}

impl LogFloat {
    /// Escala de cuantización sub-entera: 16 pasos por cada potencia de 2 (resolución = 0.0625)
    pub const SCALE: f32 = 16.0;

    /// Comprime un f32 continuo al dominio logarítmico i8
    pub fn from_f32(val: f32) -> Self {
        if val == 0.0 {
            // Valor centinela para representar el cero absoluto (log2(0) = -inf)
            return Self { sign: false, log_val: i8::MIN };
        }
        let sign = val < 0.0;
        let abs_val = val.abs();

        // E = log2(|x|) * SCALE
        let raw_exp = abs_val.log2() * Self::SCALE;
        let clamped = raw_exp.clamp((i8::MIN + 1) as f32, i8::MAX as f32);

        Self {
            sign,
            log_val: clamped.round() as i8,
        }
    }

    /// Reconstruye el valor aproximado en f32
    pub fn to_f32(&self) -> f32 {
        if self.log_val == i8::MIN {
            return 0.0;
        }
        let exp = self.log_val as f32 / Self::SCALE;
        let mag = 2.0_f32.powf(exp);
        if self.sign { -mag } else { mag }
    }

    /// Multiplicación LNS exacta O(1) mediante suma de enteros
    #[inline(always)]
    pub fn multiply_lns(&self, other: &Self) -> Self {
        if self.log_val == i8::MIN || other.log_val == i8::MIN {
            return Self { sign: false, log_val: i8::MIN };
        }
        let sign = self.sign ^ other.sign;
        let sum_exp = (self.log_val as i16) + (other.log_val as i16);
        let clamped = sum_exp.clamp((i8::MIN + 1) as i16, i8::MAX as i16) as i8;

        Self { sign, log_val: clamped }
    }
}
```

---

## 5. 📊 Matriz Comparativa de Métodos

| Técnica | Fórmula | Espacio Requerido | Complejidad Aritmética | Caso de Uso Ideal |
| :--- | :---: | :---: | :---: | :--- |
| **LNS (Log Number System)** | $x \to \log_2(x)$ | **8 a 16 bits** | $\mathcal{O}(1)$ Sumas enteras | Compresión de pesos y GEMV sin multiplicadores en CPU/Edge. |
| **Exponente Complejo / Fasor** | $x \to e^{i \omega x}$ | **Vector en $\mathbb{C}$** | Rotaciones angulares / Bit-shifts | RoPE, coherencia de fase en 2-bits y capas holográficas. |
| **Microscaling (MXFP / E8M0)** | $x = s \cdot 2^{E}$ | **8 bits por bloque** | Shift escalar masivo | Escala de amplitud macro en bloques de pesos Q4_0 y BF2. |

---

## 6. 🛠️ Aplicación en la Arquitectura GAJE Helix

1. **Cuerpo del Transformador:** Uso de LNS / Microscaling para representar factores de escala por bloque (`scale` en FP16/E8M0), reduciendo la huella de metadatos.
2. **Atención y Proyecciones en 2-Bits (BF2-Complex):** Codificación de fase fasorial pura ($e^{i\theta}$) sobre el retículo $A, C, G, T$, permitiendo que las multiplicaciones se ejecuten como rotaciones de $90^\circ$ resueltas con `swap` y `XOR` a nivel de bit.
