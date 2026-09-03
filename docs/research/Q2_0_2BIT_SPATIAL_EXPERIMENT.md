# Experimento Q2_0 (Cuantización Espacial 2-bit) — Veredicto

> Rama: `develop` · Fecha: 2026-08-20
> Ejecuta `docs/plans/Q2_0_SPATIAL_2BIT_EXPERIMENT.md`. Resultado: **NEGATIVO — H2 rechazada**.

> [!WARNING]
> **HIPÓTESIS REFUTADA EMPÍRICAMENTE (RESULTADO NEGATIVO)**  
> La cuantización pura de 2-bits degrada la coherencia semántica en modelos densos más allá del umbral admisible.  
> **Ruta Certificada:** Cuantización híbrida de producción **$Q4\_0$ (cuerpo) + $FP32$ (embeddings)** en formato `.gaje.flat` v2.

---

## 1. Resumen ejecutivo

Se implementó Q2_0 (2 bits/peso + scale/min por bloque de 32) como formato nativo en
GAJE, se exportaron modelos reales y se evaluaron frente a Q4_0. **El formato funciona
end-to-end y ahorra memoria, pero destruye la generación semántica por completo.** Q4_0
se mantiene como representación mínima viable.

| Hipótesis | Resultado |
|:---|:---|
| **H1** — Ahorro de memoria real (~40% cuerpo) | ✅ **Confirmada** |
| **H2** — Degradación tolerable en el modelo completo | ❌ **Rechazada** |
| **H3** — No apto como base de fine-tune | ✅ Confirmada (por precedente 0.6B) |

---

## 2. Infraestructura entregada

- `Q2_0Block` (32 pesos → 12 bytes: scale f16 + min f16 + 8 bytes de códigos 2-bit),
  `QuantFormat::Q2_0` (id 3), `WeightDatabase::GenomicQ2_0`, kernel escalar + AVX2.
- `quantize_q2_0_native` (PyO3) y `scripts/build_q2_0.py` (export `.gaje.flat`).
- Fase 1: `cargo test` 41 passed / 0 failed.

---

## 3. Fase 2 — Construcción y memoria (H1 confirmada)

Modelos exportados desde el mismo GGUF fuente FP16 que el Q4_0 de referencia:

| Modelo | Embd | Tamaño | Cuerpo |
|:---|:---:|:---:|:---|
| `qwen2_0_5b_q4_0_q8_0_embd.gaje.flat` (ref) | Q8_0 | 498 MB | ~226 MB (5 bits/peso) |
| `qwen2_0_5b_q2_0_q8_0_embd.gaje.flat` | Q8_0 | **413 MB** | **~141 MB (3 bits/peso)** |
| `qwen2_0_5b_q2_0.gaje.flat` | FP32 | 1175 MB | ~141 MB |

- Cuerpo Q2_0 ≈ **62%** del Q4_0 → **ahorro ~38%** (≈3 vs 5 bits/peso). ✅ H1.
- A nivel de archivo completo el ahorro se diluye (83%) porque las embeddings Q8_0
  dominan. La reducción real es del **cuerpo**, no del modelo entero.

---

## 4. Fase 3 — Evaluación generativa (H2 rechazada)

Harness `scripts/eval_generation.py`, greedy (`--temp 0.0`), 20 prompts held-out OOD
(`data/distill/heldout_06b.json`).

| Modelo | d1 | d2 | rep | deg% | Lenguaje coherente |
|:---|:---:|:---:|:---:|:---:|:---|
| Q4_0 (ref) | 0.765 | 0.882 | 0.118 | 0% | **Sí** (20/20) |
| Q2_0 | **0.997** | 1.000 | 0.000 | **0%** | **No — sopa de tokens (0/20)** |

Ejemplos representativos:
- **Q4_0** (prompt "capital de Noruega"): *"La capital de Noruega es Oslo. Es la capital
  del país."* — coherente.
- **Q2_0** (mismo prompt): *"atoiace Sous SIell讯雾PLACEConventioncomes意见rasında..."* —
  incoherente, sin significado.

### Hallazgo metodológico: la métrica de n-gramas es ciega al colapso semántico

El harness reporta **deg=0%** para Q2_0 porque su heurística (`d1 < 0.25 o rep > 0.7`) no
distingue "texto diverso y coherente" de "sopa de tokens aleatoria" (que tiene d1 alto).
Lección, en la línea de `CE_VS_GENERATION.md`: **las métricas agregadas de diversidad no
validan calidad semántica; hay que inspeccionar la salida.** Para veredictos de formato,
la inspección cualitativa + conteo de respuestas sensatas es el criterio decisivo.

---

## 5. Análisis de causa raíz

El prototipo de 120 capas (`docs/research/temporal_4bit_fase1_test.py`) ya lo predijo:
CosSim de Q2_0 en cascada = **0.146** vs **0.733** de Q4_0. Con solo 4 niveles por bloque
y un `scale/min` por bloque de 32, el error de cuantización por peso es ~4× el de Q4_0; el
ruido se amplifica a través de las 24 capas (×24 proyecciones), y la activación resultante
proyectada por el lm_head (FP32/Q8_0) ya no cae en la región semántica válida → el
`argmax` escoge tokens arbitrarios.

**La dimensión temporal no aporta** (repetir un peso estático no añade bits); la dimensión
espacial (scale/min compartido) sí hace viable el 2-bit como representación, pero a costa
de una degradación irreparable para generación.

---

## 6. Decisión final

- **Q2_0 queda descartado** como representación de producción para el cuerpo.
- **Q4_0 + embeddings FP32/Q8_0** se mantiene como el mínimo viable (regla del proyecto).
- La calidad se gana por **corpus/destilación**, no por reducir bits del cuerpo
  (confirmado también por el protocolo 0.6B: el QAT del cuerpo Q4_0 ya degrada).
- Si en el futuro se busca 2-bit, la vía sería **mixed-precision** (outliers a 4-bit +
  resto a 2-bit), no el Q2_0 uniforme — quedaría como extensión hipotética, sin
  expectativa de paridad con Q4_0.

---

## 7. Artefactos

- Código: `src/io/header/blocks.rs`, `src/nn/linear/{database,init,forward}.rs`,
  `src/compute/kernels/genomic.rs`, `src/compute/quantize/python.rs`,
  `scripts/build_q2_0.py`.
- Plan: `docs/plans/Q2_0_SPATIAL_2BIT_EXPERIMENT.md`.
- Prototipo refutación: `docs/research/temporal_4bit_fase1_test.py`.
- Modelos (gitignored): `models/production/qwen2_0_5b_q2_0*.gaje.flat`.