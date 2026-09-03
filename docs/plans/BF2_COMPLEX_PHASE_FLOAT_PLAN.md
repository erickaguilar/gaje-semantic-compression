# 🧬 Plan Arquitectónico: BF2-Complex (Punto Flotante de Fase en $\mathbb{C}$ para Inferencia de 2-Bits sin Multiplicadores)

> **Fecha:** 2 de Septiembre de 2026  
> **Versión:** `GAJE Helix v1.8.0 / BF2 Phase Engine`  
> **Estado:** 📝 `PROPUESTA DE INVESTIGACIÓN Y ESPECIFICACIÓN DE INGENIERÍA`  
> **Ámbitos:** Aritmética en $\mathbb{C}$ · Punto Flotante de Fase (BF2) · Kernels SIMD/WGSL Zero-Multiplier · Desplazamiento de Bits  
> **Módulos Directos:** `src/compute/math.rs`, `src/compute/gpu/shaders/`, `src/io/flat_header.rs`, `models/born/max_laser.gaje`

---

## 1. 🎯 Diagnóstico y Fundamento Matemático

En la recta real ($\mathbb{R}$), un formato flotante tradicional de 2 bits `BF2 E1M0` ($\text{Signo} + \text{Exponente}$) solo produce 4 escalones discretos $\{-2.0, -1.0, +1.0, +2.0\}$, carece de cero y sufre de colapso de precisión acumulativa en transformers profundos.

### 💡 La Solución: BF2-Complex (Punto Flotante de Fase en $\mathbb{C}$)
Al proyectar los 2 bits sobre el círculo unitario en el plano complejo:
* **Bit 0 ($b_0$):** Componente Real ($\text{Signo } \pm 1$).
* **Bit 1 ($b_1$):** Componente Imaginario ($\text{Signo } \pm i$).

$$\mathbf{w}_{\text{BF2}} = \frac{1}{\sqrt{2}} \left( (-1)^{b_0} + i \cdot (-1)^{b_1} \right) = e^{i \left( \frac{\pi}{4} + k \frac{\pi}{2} \right)} \quad \text{donde } k \in \{0, 1, 2, 3\}$$

```
                           Im (Eje Imaginario)
                                    │
                       01 (-1 + i)  │  00 (+1 + i)   [Fase θ = 45°]
                       (Cian / C)   │  (Verde / A)
                                    │
                      ──────────────┼────────────── Re (Eje Real)
                                    │
                       11 (-1 - i)  │  10 (+1 - i)
                       (Rojo / G)   │  (Amarillo / T)
                                    │
```

---

## 2. ⚡ Propiedades y Ventajas de BF2-Complex

1. **Continuidad por Interferencia Constructiva:**
   Aunque cada peso almacena solo 2 bits, la suma vectorial de $N$ pesos complejos genera un continuo suave con infinitos decimales:
   $$\mathbf{y}_{\text{acumulado}} = \sum_{j=1}^{N} \mathbf{w}_j \cdot \mathbf{x}_j \in \mathbb{C} \implies \text{Topología de Montaña Continua}$$
2. **Cero Silicio Multiplicador (Zero-Multiplier):**
   La multiplicación compleja por un peso BF2 se reduce a **inversiones de signo (`XOR`) y permutación de componentes (`swap`)**, ejecutadas en 1 ciclo de reloj SIMD.
3. **Escala Micro-Block ($\alpha$):**
   Cada bloque de 32 pesos incluye un factor de escala flotante $\alpha \in \text{FP16}$, permitiendo modular la energía global a **2.125 bits reales por peso**.

---

## 3. 🏛️ Arquitectura del Pipeline de Cómputo BF2

```
      [ Activación Compleja: (x_r, x_i) ]       [ Peso BF2: (b0, b1) ]
                        │                                  │
                        └──────────────┬───────────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │     BF2 Bit-Shift Kernel          │
                     │  (Swap Registros + Flip de Signo) │
                     └─────────────────┬─────────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │    Acumulación Vectorial SIMD     │
                     │    Σ (y_r + i·y_i) en Registros   │
                     └─────────────────┬─────────────────┘
                                       │
                                       ▼  (Escalar Macro α)
                     ┌───────────────────────────────────┐
                     │    Proyección de Amplitud FP16    │
                     │    y_final = α · ||y_acumulado||  │
                     └─────────────────┬─────────────────┘
                                       │
                                       ▼
                        [ Activación de Capa Siguiente ]
```

---

## 4. 🔬 Especificación de las 4 Rotaciones Canónicas

| Bits ($b_0 b_1$) | Base ADN | Fase Compleja ($e^{i\theta}$) | Operación sobre $(x_r, x_i)$ | Instrucción de Bajo Nivel |
| :---: | :---: | :---: | :---: | :--- |
| **`00`** | **A** | $\frac{+1 + i}{\sqrt{2}}$ | $(+x_r - x_i, \,\, +x_r + x_i)$ | Suma / Resta cruzada directa |
| **`01`** | **C** | $\frac{-1 + i}{\sqrt{2}}$ | $(-x_r - x_i, \,\, +x_r - x_i)$ | Negación $x_r$ + Suma cruzada |
| **`11`** | **G** | $\frac{-1 - i}{\sqrt{2}}$ | $(-x_r + x_i, \,\, -x_r - x_i)$ | Inversión total de signo |
| **`10`** | **T** | $\frac{+1 - i}{\sqrt{2}}$ | $(+x_r + x_i, \,\, -x_r + x_i)$ | Negación $x_i$ + Suma cruzada |

---

## 5. 📅 Fases del Plan de Implementación

| Fase | Duración | Entregables | Criterio de Éxito |
| :--- | :---: | :--- | :--- |
| **Fase 1: Motor Aritmético en Rust** | 2 días | `src/compute/bf2.rs` con kernels AVX2/NEON sin multiplicadores | Throughput $>120\text{ tok/s}$ en un solo hilo |
| **Fase 2: Shader GPU WGSL** | 3 días | `src/compute/gpu/shaders/batched_gemv_bf2.wgsl` | Paridad bit a bit con el kernel nativo en Rust |
| **Fase 3: Formato Binario Flat** | 2 días | Extensión en `FlatHeaderV2` para identificar tensores `QuantFormat::BF2Complex` | Carga zero-copy mmap $<0.75\text{ ms}$ |
| **Fase 4: Validación en `max_laser`** | 3 días | Forward pass y benchmark sobre `models/born/max_laser.gaje` | $\text{CosSim} > 0.94$ y PPL $< 8.0$ |

---

## 6. 🧪 Escenarios BDD de Certificación

```gherkin
Característica: Motor de Inferencia BF2-Complex
  Como subsistema de cómputo en baja precisión de GAJE
  Quiero ejecutar multiplicaciones tensoriales usando fases complejas de 2-bits
  Para obtener throughput ultra-alto sin degradación de precisión

  Escenario: Rotación de fase sin multiplicadores de punto flotante
    Dado un par de activación compleja (x_r, x_i)
    Y un bloque de pesos BF2 empaquetados en 2-bits
    Cuando se despacha el kernel "gemv_bf2_simd"
    Entonces el cálculo se realiza únicamente con swaps, adiciones e inversiones de signo
    Y el resultado coincide con la rotación analítica en C con error relativo < 1e-6

  Escenario: Carga y ejecución zero-copy en max_laser
    Dado el modelo "models/born/max_laser.gaje" exportado con tensores BF2-Complex
    Cuando se carga en memoria con el mmap flat reader
    Entonces el cold-start es menor a 0.75 ms
    Y la inferencia autoregresiva genera texto continuo sin NaNs ni divergencia de norma
```
