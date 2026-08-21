# Plan: Entrenamiento Nativo de Orden Cero — SPSA Discreto sobre Centroides

> Rama: `test/experimental` · Estado: **PROPUESTO** · Fecha: 2026-08-20
> Complementa a `docs/plans/MEMORY_EPOCHS_PLAN.md`, `docs/research/FROZEN_BODY_CAUSAL_AB_PROTOCOL.md`
> y a la visión de nacimiento nativo (organismos criados DENTRO del paisaje discreto Q4_0).
> **Tesis**: los métodos de orden cero (Monte Carlo) estiman la dirección de mejora con solo
> forward passes — el punto fuerte del motor GAJE (19–28 tok/s CPU) — eliminando por completo
> el backprop a través de la escalera discreta y su STE.

---

## 1. Contexto y motivación

### 1.1 El problema del gradiente en el hábitat discreto

En Q4_0 los pesos viven en 16 centroides por bloque. El backprop (STE) finge que la
escalera es una rampa: matemáticamente correcto (gradient check rel_err ≈ 0.06) pero
sospechoso de destruir la generación al modificar cuerpos preentrenados
(`BODY_QAT_06B_PROTOCOL.md`). Para un organismo **nacido nativo**, la pregunta es si
acaso necesita ese gradiente fingido en absoluto.

### 1.2 Precedentes que convergen

| Hecho | Fuente | Implicación |
|:---|:---|:---|
| MeZO (Malladi et al., 2023) fine-tunea LLMs hasta 66B con **solo forwards** vía SPSA | Literatura | El orden cero es práctico a escala real |
| ES ≈ estimador Monte Carlo del gradiente | Salimans et al., 2017 | La evolución y los gradientes son extremos de un mismo espectro |
| `evolution_bitwise` murió por coste (421 s/gen, OOM), no por idea | `EMPIRICAL_TRUTH_STATE.md` §4 | Un ES eficiente (2 evaluaciones/paso) es la versión viable de lo ya intentado |
| Forward nativo a 19–28 tok/s; backward era el cuello de botella | Benchmarks internos | GAJE está estructuralmente alineado con el orden cero |
| Mutación dirigida `apply_targeted_mutation_v2` existe pero evalúa ciega | `src/core/dni/evolution.rs` | El operador de perturbación discreta ya tiene base de código |

### 1.3 La idea central: SPSA discreto

SPSA clásico perturba pesos ±δε y compara dos pérdidas. La versión nativa para Q4_0
perturba **asignaciones de centroide**:

```text
1. Sortear k pesos y reasignarlos estocásticamente a un centroide vecino (c7→c8, etc.)
2. Evaluar pérdida con la perturbación aplicada      ← forward
3. Evaluar pérdida con la perturbación antitética    ← forward
4. Si L⁺ < L⁻: consolidar dirección +; si L⁻ < L⁺: consolidar −;
   desempate por fitness histórico del bloque
```

Dos forwards por paso. Sin grafo, sin activaciones, sin STE. La mutación deja de ser
ciega: pasa de "mutar y rezar" (poblaciones completas) a **comparación dirigida de pares**.

---

## 2. Objetivo e hipótesis

> **H1 (viabilidad)** — El SPSA discreto mejora el fitness de un organismo nativo usando
> exclusivamente forward passes, a coste por paso ~N/2× menor que el ES de población que
> fue refutado (N = tamaño de población típico 20–100).
>
> **H2 (estabilidad)** — Con reducción de varianza (pares antitéticos, baseline histórico,
> semillas parametrizadas), la curva de fitness no diverge en 10⁴ pasos.
>
> **H3 (híbrido)** — En nacimiento nativo, el régimen híbrido (crecimiento con reglas
> locales `centroid_counts` + refinamiento SPSA) supera a cada método puro.

**Hipótesis nula**: si el SPSA discreto no supera a la mutación aleatoria simple en el
micro-benchmark de Fase 0, se documenta el veredicto negativo y el frente se congela
(patrón Q2_0).

---

## 3. Diseño

### 3.1 Operador de perturbación discreta

- Perturbación = vector disperso de reasignaciones `(tensor_idx, weight_idx, Δcentroide)`
  con Δ ∈ {−1, +1} (vecinos en el codebook).
- k por paso adaptativo: empezar k≈64, ajustar según ratio de aceptación.
- Antitesis exacta: mismos (tensor, peso) con Δ invertido.

### 3.2 Semillas como parametrización (ADN, no blob)

La perturbación ε se regenera desde una seed (filosofía del formato genómico: el genoma
es la semilla). Coste de memoria por paso: O(1). Permite reproducibilidad bit a bit y
auditoría de cualquier paso del entrenamiento.

### 3.3 Evaluación de fitness

- Señal primaria: pérdida generativa en mini-batch fijo (no CE media de stream — lección
  de `CE_VS_GENERATION.md`).
- Verificación periódica (cada E pasos): harness generativo completo; nunca promocionar
  checkpoints sin degeneración 0%.

### 3.4 Integración con código existente

| Componente | Reutilización |
|:---|:---|
| `src/core/dni/evolution.rs::evaluate_mutant` | Base del evaluador de par (L⁺, L⁻) |
| `apply_targeted_mutation_v2` | Operador de reasignación (extender Δcentroide a vecinos) |
| Kernels forward (`genomic_dot_product_q4_0*`) | Único cómputo requerido |
| `GenomicTrainerCore` | Punto de enganche del loop alternativo `--train-zero-order` |
| Curva T_g de `dni/python.rs` | Schedule de temperatura para exploración→explotación |

### 3.5 Currículo híbrido (H3)

```text
Etapa 1 (crecimiento): reglas locales centroid_counts (gradientes nativos suaves)
Etapa 2 (refinamiento): SPSA discreto puro sobre el organismo crecido
Etapa 3 (consolidación): épocas de memoria sellan el resultado (MEMORY_EPOCHS_PLAN)
```

---

## 4. Fases con umbrales de decisión

### Fase 0 — Micro-benchmark decisivo (2–3 días)
Tarea `micro_organism` (patrón conocido): comparar 3 brazos con presupuesto idéntico de
forwards: (a) mutación aleatoria simple, (b) SPSA discreto, (c) reglas locales actuales.
**Gate**: (b) alcanza Loss objetivo ≥ 2× más rápido que (a); si no, hipótesis nula ⇒ cierre.

### Fase 1 — Módulo Rust `train-zero-order` (1 semana)
Loop forward-only integrado a `gaje-cli --train --zero-order`. **Gate**: throughput de
pasos ≥ 5× al ES refutado; memoria adicional < 50 MB sobre inferencia.

### Fase 2 — Escalera de nacimiento nativo (2–4 semanas)
Aplicar el currículo §3.5 en la escalera `micro → 5M → 32M` con destilación-como-nutrición.
**Gate por peldaño**: generación coherente (0% degeneradas) antes de crecer. Este experimento
es también LA prueba de la visión de nacimiento nativo: si el nativo llega a 32M, queda
validado como programa; si colapsa sistemáticamente, se identifica el muro real.

### Fase 3 — Especialización de organismos adultos congelados (opcional)
SPSA sobre `.gmem`-anclas y adaptadores ligeros (no cuerpo). **Gate**: mejora needle-recall
sin incumplir jamás el gate generativo.

---

## 5. Riesgos y mitigaciones

| Riesgo | Prob. | Mitigación |
|:---|:---:|:---|
| Varianza alta en 135M+ params (maldición dimensional) | Alta | Pares antitéticos + perturbación de bajo rango + baseline histórico; si persiste, confinar a ≤32M |
| Orden cero lento para aprendizaje de novo (vs adaptación) | Alta | Currículo híbrido §3.5: crecimiento con reglas locales, MC solo para refinar |
| Ratio señal/ruido insuficiente en mini-batches pequeños | Media | Batches mayores solo para el par antitético; promediado EMA de fitness |
| Duplicación de infraestructura de entrenamiento | Baja | Reutilizar evaluadores y kernels existentes (§3.4); loop nuevo mínimo |

---

## 6. Métricas de éxito

| Métrica | Umbral mínimo | Objetivo |
|:---|:---:|:---:|
| Speedup SPSA vs mutación aleatoria (Fase 0) | ≥ 2× | ≥ 5× |
| Forwards por paso | 2 | 2 |
| Memoria adicional sobre inferencia | < 50 MB | ~0 MB |
| Estabilidad (10⁴ pasos sin divergencia) | Sí | Sí |
| Peldaño 32M con 0% degeneradas | Alcanzado | + PPL < 50 |
| Regresión suite nativa | 0 fallos | 0 fallos |

---

## 7. Referencias

- MeZO: Malladi et al., *Fine-Tuning Language Models with Just Forward Passes*, 2023
- ES como estimador MC del gradiente: Salimans et al., *Evolution Strategies as a Scalable Alternative to RL*, 2017
- SPSA: Spall, 1992 · REINFORCE: Williams, 1992
- Motor: `src/compute/kernels/genomic.rs` (forwards), `src/core/dni/evolution.rs` (operadores)
- Ley del cuerpo congelado: `docs/research/BODY_QAT_06B_PROTOCOL.md`
- Métrica de éxito: `docs/research/CE_VS_GENERATION.md` · Refutación previa del ES crudo: `EMPIRICAL_TRUTH_STATE.md` §4

---
*Plan de Orden Cero Nativo v1 (Agosto 2026) — Donde la escalera es el mapa, se camina probando peldaños, no calculando rampas.*
