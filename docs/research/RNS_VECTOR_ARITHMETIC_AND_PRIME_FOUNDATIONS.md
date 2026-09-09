# 🧬 Especificación de Investigación: Aritmética Vectorial RNS (Residue Number System), Frecuencias Primas en RoPE y Dispersión Cuántica en GAJE

> **Fecha:** 9 de Septiembre de 2026  
> **Versión:** `GAJE Helix v1.8.0-research / Prime & RNS Engine`  
> **Estado:** 📝 `FORMALIZADO Y ESPECIFICADO`  
> **Ámbitos:** Sistema de Residuos Numéricos (RNS) · Aritmética Carry-Free · RoPE Primo · Secuencias de Halton · Hashing Universal LSH / K-WTA  
> **Módulos Directos:** `src/compute/kernels/`, `src/compute/math.rs`, `src/nn/attention.rs`, `src/compute/island.rs`, `src/nn/linear/`

---

## 1. 🎯 Visión General y Justificación en Silicio

Los modelos de lenguaje y compresión semántica convencionales ejecutan millones de multiplicaciones matriciales mediante **aritmética de punto flotante tradicional ($\text{FP32} / \text{FP16}$)** sobre una recta euclidiana real.

En dispositivos de borde y arquitecturas móviles (ARM Cortex / Qualcomm Snapdragon en Termux), este enfoque impone dos restricciones físicas fundamentales:
1. **Propagación del Bit de Acarreo (*Carry Propagation*):** En sumadores y multiplicadores ALU convencionales, el bit de acarreo debe propagarse secuencialmente a través de la palabra de 32 o 64 bits, imponiendo una barrera física en la frecuencia del reloj y disipando energía en forma de calor.
2. **Aliasing Armónico en Atención:** El uso de frecuencias geométricas estándar en RoPE produce armónicos espurios en secuencias largas, generando el conocido *Context Drift*.

Esta especificación formaliza las **cuatro vías clave donde la Teoría de Números y los Números Primos resuelven estos límites en `gaje-core`**.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                 Los 4 Pilares de los Números Primos en GAJE                 │
├──────────────────────────┬──────────────────────────┬───────────────────────┤
│ 1. RoPE Primo            │ 2. Aritmética RNS        │ 3. Cuasi-Ortogonalidad│
│    (Fases sin Aliasing)  │    (Carry-Free SIMD)     │    (Secuencias Halton)│
├──────────────────────────┼──────────────────────────┼───────────────────────┤
│ • Periodos coprimos      │ • Base de primos {p_k}   │ • Sin Gram-Schmidt    │
│ • Cero colisión angular  │ • Operaciones paralelas  │ • Baja discrepancia   │
│ • Longitud infinita CRT  │ • Registros u8/u16       │ • Cobertura uniforme  │
├──────────────────────────┴──────────────────────────┴───────────────────────┤
│ 4. Hashing Universal & Poda Cuántica (LSH sobre F_p para K-WTA y .gmem)     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 🏛️ Pilar 1: Codificación Posicional Libre de Colisiones (RoPE Primo)

En arquitecturas Transformer con Rotary Position Embedding (RoPE), el vector de activación de cada cabeza de atención se rota en el plano complejo mediante frecuencias angulares fijas:

$$\mathbf{x}_{\text{rotado}} = \mathbf{x} \cdot e^{i m \theta}$$

### Formulación Tradicional vs Formulación Prima:
* **Tradicional:** $\theta_j = \text{base}^{-2j/d}$. Como las frecuencias son potencias continuas, en secuencias de más de 500 tokens dos posiciones relativas distintas $m$ y $n$ producen configuraciones fasoriales que entran en armónicos periódicos simples ($\Delta \theta \approx 2\pi k$), degradando la capacidad del modelo para distinguir el orden temporal.
* **Solución RoPE Primo:** Fijar las frecuencias angulares utilizando **números primos mutuamente coprimos**:

$$\theta_j = \frac{2\pi}{p_j}, \quad p_j \in \mathbb{P} \quad (p_1 < p_2 < \dots < p_{d/2})$$

### Teorema y Propiedades Físicas:
1. **Ortogonalidad de Fase:** Las trayectorias de rotación de cada par de dimensiones $(x_{2j}, x_{2j+1})$ son mutuamente aperiódicas.
2. **Invarianza de Posición vía Teorema Chino del Resto (CRT):** Dos posiciones relativas $m \ne n$ jamás producen la misma firma de fase en todas las dimensiones simultáneamente antes de alcanzar el periodo global:

$$T_{\text{global}} = \prod_{j=1}^{d/2} p_j \gg 10^{14} \text{ tokens}$$

* **Resultado:** Cero colisiones posicionales y erradicación del *Context Drift* en secuencias extensas.

---

## 3. ⚡ Pilar 2: Sistema de Residuos Numéricos (RNS) — Producto Punto Carry-Free en Paralelo

El **Sistema de Residuos Numéricos (RNS)** es el mecanismo más potente para lograr cálculo vectorial ultra-rápido en registros enteros pequeños (`u8` / `u16`) sin desbordamiento ni propagación de acarreo.

### Fundamento Matemático (Teorema Chino del Resto):
Sea una base de módulos primos mutuamente coprimos:

$$\mathcal{B} = \{p_1, p_2, \dots, p_k\}, \quad \text{MCD}(p_i, p_j) = 1 \ \forall \ i \ne j$$

El rango dinámico unívoco del sistema es el producto de los módulos:

$$M = \prod_{i=1}^{k} p_i$$

Cualquier entero o coordenada $X \in [0, M-1]$ se proyecta de forma unívoca en su tupla de residuos:

$$X \xrightarrow{\text{RNS}} (r_1, r_2, \dots, r_k), \quad \text{donde } r_i = X \pmod{p_i}$$

```text
               CÁLCULO VECTORIAL RNS DESACOPLADO (CARRY-FREE)

                 Vector U ──► [ RNS Canal p₁ ] ──► (u · v) mod p₁ ──┐
                              [ RNS Canal p₂ ] ──► (u · v) mod p₂ ──┼──► Reconstrucción
                              [ RNS Canal p₃ ] ──► (u · v) mod p₃ ──┤    Exacta CRT
                 Vector V ──► [ RNS Canal p₄ ] ──► (u · v) mod p₄ ──┘
                              (Paralelo en Registros SIMD u8 / u16)
```

### Cálculo Vectorial Desacoplado:
El producto escalar entre dos vectores $\mathbf{u} \cdot \mathbf{v} = \sum_{j=1}^{D} u_j v_j$ se evalúa **canal por canal en paralelo sin interacción cruzada**:

$$\langle \mathbf{u}, \mathbf{v} \rangle_i = \left( \sum_{j=1}^{D} u_{j, i} \cdot v_{j, i} \right) \bmod p_i$$

### Ventajas Radicales para el Núcleo Nativo en Rust (`src/compute/kernels/`):
1. **Aritmética Libre de Acarreo (*Carry-Free Arithmetic*):** No existe propagación de acarreo entre los canales primos. Cada canal opera de forma completamente aislada.
2. **Vectorización SIMD Masiva en `u8`:** Si elegimos primos menores a 256 (por ejemplo: $\{251, 241, 239, 233, 229, 227, 199, 197\}$), un registro ARM NEON de 128 bits procesa **16 operaciones modulares simultáneas por ciclo de reloj**.
3. **Rango Dinámico Gigantesco:** El producto de esos 8 primos de 8 bits produce un rango dinámico de $M \approx 1.2 \times 10^{19}$ (equivalente a **64 bits enteros de precisión perfecta sin usar ALUs de 64 bits**).
4. **Cero Errores de Redondeo:** El producto punto es **matemáticamente exacto**, eliminando las derivas numéricas causadas por la mantisa de punto flotante en cuantizaciones profundas.

---

## 4. 🎲 Pilar 3: Secuencias de Baja Discrepancia (Vectores Cuasi-Ortogonales con Halton)

En modelos comprimidos a 2 bits o 4 bits, la inicialización de matrices de pesos, codebooks genómicos y perturbaciones SPSA requiere vectores homogéneamente distribuidos y cuasi-ortogonales.

### La Falacia del Muestreo Pseudoaleatorio (PRNG):
Los generadores estándar generan cúmulos (*clustering*) en ciertas regiones del espacio y dejan vacíos dimensionales. Ortogonalizarlos mediante Gram-Schmidt exige $O(D^3)$ operaciones de punto flotante.

### Generación Cuasi-Ortogonal vía Secuencias de Halton:
Para generar la coordenada $j$-ésima en un espacio de dimensión $D$, se utiliza el $j$-ésimo número primo $p_j$ como base de inversión radical:

$$\mathbf{x}_n = \left( \phi_{p_1}(n), \phi_{p_2}(n), \phi_{p_3}(n), \dots, \phi_{p_D}(n) \right)$$

Donde:

$$\phi_p(n) = \sum_{k=0}^{L} b_k(n) \cdot p^{-(k+1)}, \quad \text{con } n = \sum_{k=0}^{L} b_k(n) \cdot p^k$$

### Propiedades:
* **Cuasi-Ortogonalidad Emergente:** Dos vectores generados en dimensiones primas distintas son naturalmente ortogonales en alta dimensión:
  $$\lim_{D \to \infty} \cos(\mathbf{x}^{(a)}, \mathbf{x}^{(b)}) \to 0$$
* **Cero Costo de Gram-Schmidt:** No requiere normalizaciones costosas; las coordenadas cubren el hiperespacio de activación con **discrepancia mínima garantizada ($O(\log^D N / N)$)**.

---

## 5. 🕸️ Pilar 4: Tablas Hash Vectoriales y Poda Cuántica (LSH sobre $\mathbb{F}_p$ para K-WTA y `.gmem`)

Para acelerar la búsqueda de vecinos más cercanos en la memoria asociativa (`.gmem`) y realizar la reducción dimensional dispersa en la inhibición lateral **K-WTA (K-Winners-Take-All)**:

### Familia de Funciones Hash Universales Lineales Congruenciales:

$$h(\mathbf{v}) = \left( \sum_{j=1}^{D} p_j \cdot v_j \right) \bmod P$$

Donde:
* $p_j \in \mathbb{P}$ es una secuencia precalculada de números primos asignados a cada dimensión latente.
* $P$ es un número primo grande que define el tamaño del buffer o tabla de activación (ej. $P = 65537$, primo de Fermat).

### Impacto en la Inhibición Lateral K-WTA:
1. **Minimización de Colisiones Destructivas:** La probabilidad de que dos características semánticas no relacionadas colisionen en el mismo canal de activación es estrictamente:
   $$P_{\text{colisión}}(\mathbf{u}, \mathbf{v}) \le \frac{1}{P}$$
2. **Poda Cuántica Eficiente:** Permite seleccionar los $K$ canales dominantes en $O(D)$ operaciones de enteros sin ordenar arrays completos en memoria, acelerando la propagación de capas en CPU móvil.

---

## 6. 📊 Matriz de Impacto en Silicio: Álgebra Lineal Clásica vs Sistema Primo RNS

| Dimensión de Ingeniería | Álgebra Convencional (Float / $2^n$) | Arquitectura RNS / Primos (GAJE) | Ganancia en Silicio |
|---|---|---|---|
| **Propagación de Acarreo** | Presente en cada suma/multiplicación | **Cero (Carry-Free Arithmetic)** | Mayor frecuencia de reloj y menor consumo térmico |
| **Ancho de Registros SIMD** | ALUs pesadas de 32/64 bits | Registros empaquetados `u8` / `u16` | **$4\times$ más operaciones por ciclo SIMD** |
| **Precisión de Acumulación** | Error de redondeo flotante acumulado | **Resultado exacto vía CRT** | Cero deriva numérica en transformers profundos |
| **Resonancia en RoPE** | Aliasing periódico cíclico | Aperiodismo coprimo ($\prod p_k$) | **Cero Context Drift en secuencias largas** |
| **Generación de Vectores** | Gram-Schmidt $O(D^3)$ | Halton de base prima $O(D)$ | **Inicialización instantánea sin cómputo matricial** |
| **Búsqueda y Poda K-WTA** | Hashing con colisiones armónicas | Familias lineales sobre campo $\mathbb{F}_P$ | **Dispersión óptima con garantía teórica $\le 1/P$** |

---

## 7. 🛠️ Hoja de Ruta de Implementación en Código (`gaje-core`)

1. **Módulo RNS (`src/compute/rns.rs`):**
   * Definición de la estructura `RnsChannel<const P: u8>` en Rust.
   * Implementación del producto punto SIMD empaquetado para NEON (`vmlal_u8`) y AVX2.
2. **Tabla de Frecuencias Primas para RoPE (`src/nn/attention.rs`):**
   * Generación en tiempo de compilación (`const fn`) de la tabla de primos coprimos para los cabezales de atención.
3. **Generador Cuasi-Monte Carlo (`src/compute/halton.rs`):**
   * Generador de inversión radical para inicialización de anclas en modelos de 2 bits y muestreo SPSA.
4. **Validación Formal en Harness BDD:**
   * Comparación de paridad entre producto punto FP32 y producto punto exacto RNS reconstruido por CRT.
