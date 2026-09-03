# 🏔️ Plan Estratégico: Recuperación de la Montaña Semántica (Topología Continua en $\mathbb{C}$ mediante Desplazamiento de Bits en 2-Bits)

> **Fecha:** 2 de Septiembre de 2026  
> **Versión:** `GAJE Helix v1.8.0-research`  
> **Estado:** 📝 `PROPUESTA DE INVESTIGACIÓN Y DISEÑO ARQUITECTÓNICO`  
> **Ámbitos:** Redes Complejas en $\mathbb{C}$ · Álgebra de Clifford / Rotaciones QPSK · Kernels SIMD por Desplazamiento de Bits · Interferencia Constructiva  
> **Módulos Afectados:** `src/compute/math.rs`, `src/compute/gpu/shaders/`, `src/nn/spiking/`, `src/io/flat_header.rs`

---

## 1. 🎯 La Metáfora y el Fundamento Físico: "Recuperar la Montaña"

Los modelos de frontera (FP32/FP16) navegan por **montañas y valles continuos** de topología diferenciable suave. La cuantización escalar directa aplasta ese paisaje en una **escalera rígida de 4 peldaños (2-bits)**, provocando una pérdida acumulativa del 98% de resolución tras 120 capas.

### 💡 El Principio de Recuperación Topológica:
Un punto flotante arbitrario no necesita almacenarse como un número monolítico; puede sintetizarse como la **superposición armónica de fases ortogonales en el plano complejo $\mathbb{C}$**.

$$\mathcal{W}_{\text{continuo}}(z) = \sum_{k=1}^{K} \alpha_k \cdot e^{i \theta_k} \quad \text{donde } \theta_k \in \left\{ 0, \, \frac{\pi}{2}, \, \pi, \, \frac{3\pi}{2} \right\} \equiv \{00, 01, 11, 10\}$$

Al desacoplar el peso en **1-bit Real ($\pm 1$) + 1-bit Imaginario ($\pm i$)**, cada proyección se evalúa mediante **desplazamientos de bits y permutaciones de signo en registros SIMD/GPU**, recuperando las curvaturas suaves (la montaña) mediante interferencia constructiva sin incurrir en multiplicaciones flotantes pesadas.

---

## 2. 🏛️ Arquitectura del Sistema: Transformación de Escalera a Onda Continua

```
 [ Peso 2-Bits: 01 (i) ] ──┐
 [ Peso 2-Bits: 00 (+1) ] ─┼──► [ Bit-Shift Kernel (SIMD/WGSL) ] ──► [ Onda Compleja z(t) ]
 [ Peso 2-Bits: 10 (-i) ] ─┘            (Swap + Bit-Flip)                     │
                                                                              ▼
                                                                [ Interferencia Constructiva ]
                                                                (Superposición de Fases)
                                                                              │
                                                                              ▼
                                                                  🏔️ MONTAÑA SEMÁNTICA
                                                                 (Topología Suave Reconstruida)
```

---

## 3. 🔬 Los 4 Pilares Técnicos del Plan

### A. Álgebra de Proyección por Desplazamiento de Bits (Zero-Multiplier GEMV)
En lugar de multiplicar matrices densas $W \cdot x$:
* Representamos la activación de entrada como un par complejo $(x_r, x_i)$.
* El kernel evalúa las 4 rotaciones canónicas en 1 ciclo de reloj:

| Código 2-Bit | Estado Complejo | Operación Aritmética | Transformación a Nivel de Bit |
| :---: | :---: | :---: | :--- |
| **`00`** (A) | $+1$ | $(+x_r, +x_i)$ | Identidad (Paso directo) |
| **`01`** (C) | $+i$ | $(-x_i, +x_r)$ | Permutación cruzada + bit-flip en $x_i$ |
| **`11`** (G) | $-1$ | $(-x_r, -x_i)$ | Inversión de signo en ambos componentes |
| **`10`** (T) | $-i$ | $(+x_i, -x_r)$ | Permutación cruzada + bit-flip en $x_r$ |

### B. Cavidad de Resonancia Fabry-Pérot (Acumulador Toroidal)
A través de las 12 capas de `max_laser` ($D=384$), las fases no colapsan porque el potencial de activación se acumula en un toroide de fase:

$$\Psi_{l+1} = \text{RoPE}(\Psi_l) \odot \exp\left( i \cdot \mathbf{W}_{\text{QPSK}}^{(l)} \right)$$

Esto preserva la información angular ($\cos \theta$) en lugar de la magnitud absoluta, cancelando el decaimiento exponencial de la norma.

### C. Gating Conforme y Calibración de Escala Macro ($\alpha$)
Cada bloque de 32 pesos retiene un único escalar $\alpha \in \mathbb{R}^+$ que modula la amplitud global de la montaña:

$$\mathbf{y} = \alpha \cdot \left[ \sum_{j} \mathbf{Rot}(w_j, x_j) \right]$$

Esto reduce el consumo a **2.125 bits reales por peso** (2-bits de fase + 4 bytes de escala cada 32 pesos), manteniendo la huella del modelo en $\approx 20\text{ MB}$.

### D. Shader GPU Acelerado (`complex_bitshift_gemv.wgsl`)
Implementación en WebGPU/Vulkan de despacho masivo con empaquetado de 16 fases por `u32` (8 pesos complejos simultáneos por hilo de ejecución).

---

## 4. 📅 Cronograma y Fases de Implementación

| Fase | Duración | Entregables | Criterio de Éxito |
| :--- | :---: | :--- | :--- |
| **Fase 1: Kernels de Fase SIMD** | 3 días | `src/compute/complex_simd.rs` con operaciones AVX2/NEON por bit-shift | $0\text{ multiplicaciones FP32}$, throughput $>100\text{ tok/s}$ |
| **Fase 2: Shader WGSL Complejo** | 3 días | `src/compute/gpu/shaders/complex_bitshift_gemv.wgsl` | Paridad exacta con kernel Rust en GPU |
| **Fase 3: Integración en `max_laser`** | 4 días | Adaptación de `src/nn/linear/` para inferencia con fases QPSK | $\text{CosSim} > 0.92$ sostenida en 12 capas |
| **Fase 4: Certificación de Paridad** | 2 días | Reporte comparativo de generación y perplejidad vs baseline | PPL reducida de $45.0 \to < 8.0$ en 2-bits |

---

## 5. 🧪 Escenarios BDD de Verificación

```gherkin
Característica: Recuperación de la Montaña Semántica en 2-Bits
  Como motor de inferencia nativo GAJE
  Quiero reconstruir variedades semánticas continuas usando rotaciones complejas por bit-shift
  Para evitar el colapso de la escalera de 2-bits sin incurrir en consumo de punto flotante

  Escenario: Rotación pura sin multiplicadores en SIMD
    Dado un vector de activación complejo (x_r, x_i)
    Y un vector de pesos empaquetados en fases QPSK (00, 01, 11, 10)
    Cuando ejecuto el kernel "complex_bitshift_gemv"
    Entonces el cómputo se realiza exclusivamente con shifts, XOR y adiciones
    Y la similitud angular con respecto al cálculo analítico es 1.000000

  Escenario: Preservación de la coherencia en 12 capas
    Dado el organismo colimado "max_laser.gaje" (D=384, L=12)
    Cuando realizo el forward pass complejo a través de las 12 capas
    Entonces la similitud de fase final se mantiene superior a 0.90
    Y la generación no sufre de colapso de vocabulario ni bucles caóticos
```
