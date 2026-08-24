# Protocolo A/B Causal: ¿Por qué se congela el cuerpo? (Herencia vs Régimen vs Discretización Q4_0)

> **Estado**: PROPUESTO (pendiente de ejecución) · Fecha: 2026-08-20
> Extiende a `docs/research/BODY_QAT_06B_PROTOCOL.md` y `docs/research/CE_VS_GENERATION.md`.
> Pregunta origen: *el congelamiento del cuerpo, ¿es heredado de los maestros/alumno o es
> propiedad de la arquitectura GAJE?*

---

## 1. Contexto y evidencia previa

La regla operativa actual ("el cuerpo cuantizado se congela") está respaldada por
múltiples cierres negativos, pero su **causa** no está aislada. Evidencia disponible:

| Observación | Evidencia | Qué descarta/apoya |
|:---|:---|:---|
| El alumno base ya cuantizado genera perfecto (0% degeneradas, d1=0.764) | `BODY_QAT_06B_PROTOCOL.md`, fila Base | **Descarta herencia**: si la línea genética fuera la causa, el base estaría roto |
| Dos familias no emparentadas colapsan igual (SmolLM2 y Qwen2-0.5B) | Ambos protocolos 2026-08 | **Descarta linaje específico**: el fallo es asimétrico ni depende del maestro |
| Los maestros solo generan corpus por inferencia (nunca se entrenan) | Pipeline de destilación | **Descarta transmisión**: nada del maestro "congela" al alumno |
| El backprop Q4_0 es numéricamente correcto (gradient check f64, rel_err ≈ 0.06) | `CE_VS_GENERATION.md` §Cierre | **Descarta bug de implementación**: es propiedad del paisaje, no un error de código |
| CosSim 0.997 (1 capa) → 0.733 (120 capas): el error de cuantización se amplifica en cascada | `Q2_0_SPATIAL_2BIT_EXPERIMENT.md` §3 | **Apoya discretización**: la perturbación por capa compone |
| Todos los entrenamientos usaron corpus minúsculos (16 pares, 2274 tok) con lr 2e-4 | `BODY_QAT_06B_PROTOCOL.md` §Corpus | **Variable no controlada**: ese régimen dañaría también a un LLM FP32 (olvido catastrófico universal) |

**Conclusión del análisis**: el congelamiento NO es heredado; es una interacción entre el
régimen de entrenamiento y la discretización Q4_0. Este protocolo aísla cuál domina.

---

## 2. Hipótesis en competencia

> **H1 (Régimen)** — La causa dominante es el régimen de entrenamiento (corpus diminuto +
> lr alto). Un fine-tune FP32 con el protocolo idéntico también colapsa.
>
> **H2 (Discretización)** — La causa dominante es el paisaje discreto Q4_0 (gradientes STE
> que cruzan centroides, amplificados por profundidad). El FP32 idéntico sobrevive.
>
> **H3 (Mixta)** — El FP32 degrada parcialmente (pierde factualidad held-out) sin colapsar
> a degeneración: ambas causas contribuyen.

---

## 3. Diseño experimental

### 3.1 Brazos

| Brazo | Precisión | Método | Corpus | LR | Estado |
|:---|:---|:---|:---|:---:|:---|
| **C (control)** | Q4_0 | IQAT GAJE (`refine_with_grads_core`) | `train_06b_15.jsonl` | 2e-4 | ✅ Ya ejecutado (100%/95% degeneradas) |
| **A** | FP32 | PyTorch HF, full fine-tune | ídem (16 pares) | 2e-4 | Pendiente |
| **B** | FP32 | PyTorch HF, full fine-tune | ídem | 1e-5 | Pendiente |
| **D (opcional)** | FP32 | PyTorch HF, LoRA r=16 sobre q/k/v/o | ídem | 2e-4 | Pendiente |

Estudiante en todos los brazos: **Qwen2-0.5B-Instruct** (mismo checkpoint que el control).
Mismo tokenizador, mismos pasos efectivos, mismo seed.

### 3.2 Evaluación (idéntica para todos los brazos)

- Harness generativo fijado: `scripts/eval_generation.py --prompts-file data/distill/heldout_06b.json`
  (20 prompts held-out, temp 0.4 + greedy).
- Métricas: distinct-1/2, tasa de repetición de bigramas, % degeneradas, longitud media.
- Regla del Mandato de Verdad Empírica: la generación es la métrica de éxito; la CE es auxiliar.

### 3.3 Extensión clave: reordenamiento del pipeline

Si algún brazo FP32 sobrevive, exportarlo con `export_gaje_flat.py` (Q4_0 híbrido v2) y
re-evaluar el `.flat` resultante con el mismo harness. Esto valida el flujo alternativo
**especializar-primero, cuantizar-después** frente al flujo actual **cuantizar-primero,
adaptar-después** (IQAT).

---

## 4. Matriz de decisión

| Resultado | Veredicto | Implicación de producto |
|:---|:---|:---|
| A colapsa (≥50% degeneradas) | **H1**: régimen dominante | El cuerpo puede descongelarse con corpus grandes/lr bajo; el límite era el dato, no el formato |
| A sobrevive y B colapsa parcialmente | **H2**: discretización dominante | Ley reforzada: especialización SOLO pre-cuantización; IQAT post-export contraindicado |
| A sobrevive completo | **H2 fuerte** | Reordenar pipeline: train FP32 → export Q4_0 → cuerpo congelado definitivo |
| D sobrevive donde A colapsa | Adaptadores como vía ligera | LoRA pre-export como régimen de especialización recomendado |

En todos los escenarios, el flujo **especializar-primero, cuantizar-después** queda como
recomendación por defecto: elimina la variable STE sin costo conocido.

---

## 5. Coste estimado

- Infraestructura: 100% existente (corpus, held-out, harness, exporter).
- Cómputo: 4 brazos × ~10 min de fine-tune en 0.5B (GPU o CPU lenta) + evaluaciones.
- Riesgo de ejecución: bajo; ningún componente nuevo por construir.

---

## 6. Referencias

- `docs/research/BODY_QAT_06B_PROTOCOL.md` — cierre negativo que motiva el A/B
- `docs/research/CE_VS_GENERATION.md` — lecciones metodológicas (harness generativo obligatorio)
- `docs/meta/EMPIRICAL_TRUTH_STATE.md` — mandato de registro de veredictos
- `data/distill/train_06b_15.jsonl`, `data/distill/heldout_06b.json` — insumos
- `scripts/eval_generation.py` — evaluador oficial

---
*Protocolo A/B Causal v1 (Agosto 2026) — Aislar la causa antes de legislar la ley.*
