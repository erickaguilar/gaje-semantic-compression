# 🧬 Dinámica del Gradiente en Modelos Cuantizados y Modulación del Flujo de Información

> **Versión:** v1.6.0-alpha (Silver Adult)  
> **Fecha:** 20 de agosto de 2026  
> **Ubicación:** `docs/reports/GRADIENT_DYNAMICS_AND_FLOW_MODULATION.md`  
> **Componente:** Teoría de Optimización, Straight-Through Estimator (STE) y Arquitectura de Flujo  

---

## 1. 🎯 El Problema Fundamental: La Escalera de Peldaños Rígidos

En el entrenamiento tradicional de redes neuronales (FP32), la superficie de pérdida (*loss surface*) es una **rampa continua y suave**. El gradiente $\nabla L = \frac{\partial \text{Loss}}{\partial W}$ permite dar micro-pasos infinitesimales ($\Delta W = -lr \cdot \nabla L$) que conducen al fondo del valle de error.

En modelos comprimidos a **4-bits (Q4_0)** o **2-bits**, la superficie de pérdida se transforma en una **escalera discreta con peldaños rígidos**:

```
RAMPA SUAVE (FP32 / Continuo)         ESCALERA DE PELDAÑOS (GAJE Cuantizado)
          ╲                                     ┌───┐ (Peldaño 3)
           ╲                                ┌───┘   └─── (Salto discreto)
            ╲  ◄── Micro-ajuste             └───┐ (Peldaño 2)
             ╲     suave y exacto               └───┐
              ╲                                     └───┐ (Peldaño 1)
               ▼ (Fondo del Valle)                      ▼ (El valle cayó en medio)
```

### ⚠️ Los 3 Obstáculos de Optimizar en Escalones:
1. **Valles Inalcanzables:** La solución óptima suele ubicarse en valores intermedios (ej. $2.5$). Al tener solo enteros ($2$ o $3$), el peso no puede ubicarse en el punto exacto sin saltar bruscamente.
2. **Oscilación y Saltos Violentos:** Si el gradiente acumulado supera el umbral del centroide, millones de pesos saltan de escalón al mismo tiempo, desestabilizando la red.
3. **Fractura de la Variedad Geométrica (*Manifold Fracture*):** El transformer opera como un engranaje de 120 matrices correlacionadas. Forzar saltos discretos masivos destruye la coherencia de atención, provocando bucles repetitivos y balbuceo.

---

## 2. 🌊 La Solución: Modulación del Flujo de Información

Para permitir que un modelo cuantizado aprenda conocimiento nuevo (ej. destilación de español) sin destruir su base comprimida, la estrategia óptima no es mover los 500 millones de pesos cuantizados, sino **calibrar el flujo de información por donde viaja la señal**.

```
                            ENTRADA (x)
                                 │
                ┌────────────────┴────────────────┐
                ▼                                 ▼
       ┌─────────────────┐               ┌─────────────────┐
       │   CUERPO Q4_0   │ (Roca sólida) │  CANAL CONTINUO │ (FP16 / LoRA)
       │  (16 escalones) │ [CONGELADO]   │   (Micro-flujo) │ [APRENDE SUAVE]
       │   500M pesos    │               │  < 1M pesos     │
       └────────┬────────┘               └────────┬────────┘
                │                                 │
                └────────────────┬────────────────┘
                                 ▼ (Suma de flujos)
                             SALIDA (y)
```

---

## 3. 🛠️ Las 3 Vías de Calibración de Flujo para GAJE

### A. Vía 1: Desvío Continuo de Rango Bajo (*LoRA / Bypass Adapters*)
* **Mecanismo:** Mantener la matriz $W_{\text{Q4}}$ 100% congelada y sumar una proyección lineal continua de bajo rango:
  $$y = W_{\text{Q4}} \cdot x + (B \cdot A) \cdot x \quad \text{donde } A \in \mathbb{R}^{r \times d}, B \in \mathbb{R}^{d \times r}, r \ll d$$
* **Ventaja:** Toda la micro-precisión del gradiente se absorbe en $A$ y $B$ (FP16/FP32), evitando saltos discretos en el cuerpo cuantizado.

### B. Vía 2: Válvulas Residuales (*LayerScale / RMSNorm Gains*)
* **Mecanismo:** Entrenar únicamente un vector multiplicador $\mathbf{\alpha}$ de dimensión $d_{\text{model}}$ (896 floats) a la salida de cada bloque residual:
  $$h_{l+1} = h_l + \mathbf{\alpha}_l \odot \text{Bloque}_l(h_l)$$
* **Ventaja:** Ajusta la ganancia de cada bloque con solo $\approx 20,000$ parámetros flotantes, cerrando el paso a bloques que generen ruido y amplificando los bloques informativos.

### C. Vía 3: Modulación Epigenética por Entropía (Válvulas RNA)
* **Mecanismo:** Utilizar el subsistema bio-inspirado de GAJE (`src/compute/math.rs` y `src/core/topology.rs`):
  * **Baja Entropía (Tokens comunes):** El flujo pasa directo sin perturbación.
  * **Alta Entropía (Incertidumbre / Razonamiento):** Se activa la resonancia de fase y confinamiento K-WTA para expandir la capacidad expresiva de la capa en vuelo.

---

## 4. 📋 Conclusiones y Reglas de Producción

1. **El cuerpo Q4_0 debe mantenerse congelado o con adaptaciones ultra-cortas:** El modelo campeón `smollm2_4bit_quality` demostró que solo 1,520 tokens con $lr=2\times 10^{-4}$ y `lm_head` congelado es el punto dulce de adaptación.
2. **Para adaptación profunda a nuevas tareas:** Implementar canales continuos (*Bypass/LoRA*) o multiplicadores de capa (*LayerScale*) antes de intentar mover centroides de 4-bits masivamente.
