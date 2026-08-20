# Plan de Experimento: Q2_0 — Cuantización Espacial por Bloque (2 bits/peso) vs Q4_0

> Rama: `develop` · Estado: **propuesto** · Fecha: 2026-08-20
> Complementa a `docs/plans/QUALITY_EVAL_PROTOCOL.md` y `docs/plans/QUALITY_EXPORT_PLAN.md`.
> Refuta y reemplaza la vía de `docs/plans/TEMPORAL_4BIT_EMULATION_DESIGN.md`.

---

## 1. Contexto y motivación

El esquema de **emulación temporal 2-bit→4-bit** fue refutado numéricamente en
`docs/research/temporal_4bit_fase1_test.py` (commit `82685fd`):

- **Temporal no aporta**: repetir un peso estático K veces no añade bits. El RMSE del
  dithering no converge con K (≈6.9 constante); la información descartada no se recupera.
- **La "emulación temporal" real almacena 4 bits** (msb 2 + lsb 2) y es bit a bit idéntica
  a Q4_0: no ahorra memoria, solo añade latencia.

La vía que **sí** permite acercarse a 2 bits/peso es la **dimensión espacial**: compartir
el `scale`/`min` por bloque de pesos. El prototipo Q2_0 (2 bits por peso + scale/min del
bloque) mostró en cascada de 120 capas CosSim **0.146** vs **0.005** del 2-bit puro global
(no colapsa) — pero aún degradado frente a Q4_0 (0.733).

Este plan convierte ese prototipo en un **formato real en GAJE** y decide, con métricas
objetivas en el **modelo completo**, si el ahorro de ~40% de RAM justifica la degradación.

---

## 2. Objetivo e hipótesis

**Objetivo**: implementar `Q2_0` (2 bits/peso + scale/min por bloque) como formato nativo
en GAJE y medir su impacto frente a Q4_0 en generación y preservación de conocimiento.

**Hipótesis central**:
> H1 — El ahorro de memoria de Q2_0 es real y sustancial: ~40% frente a Q4_0
> (~3 bits/peso vs ~5 bits/peso).
>
> H2 — La degradación de Q2_0 frente a Q4_0 es **tolerable** en el modelo completo:
> no colapsa la generación (0% degeneración) y preserva la capacidad factual, aunque
> con peor perplejidad que Q4_0.
>
> H3 (clave) — Q2_0 **no** es apto como base de fine-tune del cuerpo: el QAT del cuerpo
> Q4_0 ya destruye la generación (protocolo 0.6B); a 2 bits la fragilidad será mayor.

**Hipótesis nula (lo que desecharía Q2_0)**:
> Si Q2_0 muestra >5% de secuencias degeneradas o colapso semántico en el harness, se
> descarta como representación de producción y Q4_0 queda como mínimo viable.

---

## 3. Antecedentes empíricos (prototipo, no producto)

Resultados de `docs/research/temporal_4bit_fase1_test.py` (matriz 896×896, 120 capas
lineales en cascada, normalización por capa):

| Esquema | bits/peso | CosSim 1 capa | CosSim 120 capas |
|:---|:---:|:---:|:---:|
| Q4_0 (scale+min por bloque) | 4 | 0.997 | 0.733 |
| 2-bit puro (4 centroides globales) | 2 | 0.935 | **0.005** (colapso) |
| 2-bit real (solo MSB) | 2 | 0.932 | 0.005 (colapso) |
| **Q2_0 (2-bit + scale/min por bloque)** | **~2.1** | **0.932** | **0.146** (no colapsa) |
| Emulación temporal | 4 (no ahorra) | 0.997 | 0.733 (≡ Q4_0) |

Lección: la **estructura espacial del bloque** (scale/min compartido) es lo que hace
viable el 2-bit. El prototipo es a capa suelta; este plan lo lleva al modelo completo.

---

## 4. Diseño del formato Q2_0

### 4.1 Estructura del bloque (análoga a `Q4_0Block`)

Formato actual `Q4_0Block` (`src/io/header/blocks.rs`): 32 pesos → 20 bytes
(scale f16 + min f16 + 16 bytes de códigos 4-bit). 5 bits/peso.

Formato propuesto `Q2_0Block`:

```
#[repr(C, packed)]
pub struct Q2_0Block {
    pub scale: half::f16,   // scale del bloque (compartido, 32 pesos)
    pub min:   half::f16,   // min del bloque (compartido, 32 pesos)
    pub qs:    [u8; 8],     // 8 bytes -> 4 códigos de 2 bits por byte = 32 pesos
}
```

- 32 pesos → 12 bytes (2+2+8) = **3 bits/peso**.
- Ahorro frente a Q4_0 (20 B/bloque): **40% menos memoria**.
- Layout de bits dentro de cada byte `b`: `c0 = b & 0b11`, `c1 = (b>>2)&0b11`,
  `c2 = (b>>4)&0b11`, `c3 = (b>>6)&0b11`.
- Dequant: `valor = q_value(idx) as f32 * scale + min` (misma fórmula que Q4_0, 4 niveles).

### 4.2 Cuantizador

Algoritmo por bloque de 32 pesos (estilo Variant B, igual que Q4_0):
1. `min = min(bloque)`, `max = max(bloque)`.
2. `scale = (max - min) / 3.0` (4 niveles → paso de 2 bits).
3. `code = clamp(round((w - min)/scale), 0, 3)`.
4. Dequant = `code * scale + min`.

Opcional de calibración (posterior, no en la Fase 1): **Q2_0 + outlier** — mantener un
pequeño subconjunto de pesos de alto impacto (outliers) a 4-bit y el resto a 2-bit, para
recuperar el CosSim en cascada. Se evalúa solo si el Q2_0 puro no pasa el umbral.

---

## 5. Fases del experimento

### Fase 1 — Infraestructura Rust: `Q2_0Block`
**Deliverable**: bloque nativo + dequant + serialización en `.gaje.flat`.

- `src/io/header/blocks.rs`: añadir `Q2_0Block` con `q_value()`/`dequantize_weight()`.
- `src/io/header/types.rs`: nuevo `QuantFormat::Q2_0`.
- `src/io/header/flat.rs` / `flat_writer.rs`: leer/escribir bloques Q2_0 y los fusionados
  `attn_qkv`/`ffn_gate_up` (reutilizar la ramificación existente).
- `src/nn/linear/database.rs`: variante `WeightDatabase::GenomicQ2_0(Arc<Vec<Q2_0Block>>)`
  y su `bit_depth()` (2) y `size()`.
- `src/nn/linear/forward.rs`: forward de Q2_0 (mismo patrón que GenomicQ4_0, dequant en
  bloques contiguos con rayon).
- Tests unitarios: roundtrip, error ≤ paso/2, forward equivale a dequant+matmul.

**Criterio de salida**: `cargo test` verde (los 38 existentes + nuevos).

### Fase 2 — Cuantizador y export de un modelo Q2_0
**Deliverable**: primer artefacto `.gaje.flat` Q2_0 completo.

- `src/compute/quantize/python.rs`: `quantize_q2_0_native` (wrapper PyO3).
- Script `scripts/build_q2_0.py`: cargar FP32 (reutilizar `build_fp32_qwen2.py` /
  `build_smollm2.py`), cuantizar cada tensor a Q2_0, exportar `.gaje.flat`.
- Verificación: `scripts/check_flat_parity.py` adaptado a Q2_0; RSS/memoria medida.

**Criterio de salida**: modelo Q2_0 carga y genera sin panic; memoria ≈ 60% del Q4_0.

### Fase 3 — Harness de evaluación idéntico a Q4_0
**Deliverable**: números comparables Q4_0 vs Q2_0 con `scripts/eval_generation.py`.

- Evaluar ambos sobre **los mismos prompts** y temperatura.
- Para veredictos deterministas usar `--temp 0.0` (greedy); `--temp 0.4` solo como barrido.
- Set held-out: reutilizar `data/distill/heldout_06b.json` + prompts factuales de
  `QUALITY_EVAL_PROTOCOL.md` (capitales, fechas, aritmética, traducción, pregunta OOD).

**Criterio de salida**: tabla con métricas de ambas representaciones.

### Fase 4 — Decisión
**Deliverable**: documento de veredicto en `docs/research/` + entrada en `docs/INDEX.md`.

---

## 6. Métricas y criterios de éxito

| Métrica | Cómo | Q4_0 (ref) | Q2_0 objetivo | Umbral de aceptación |
|:---|:---|:---:|:---:|:---|
| Memoria (bytes/peso) | `size()` / RSS | ~5 bits/peso | ~3 bits/peso | ≤ 65% del Q4_0 |
| CosSim capa aislada | dequant vs FP32 | 0.997 | ≥ 0.90 | — |
| CE/PPL held-out | eval_generation | base | ≤ ~2× base | informe, no bloqueante |
| % secuencias degeneradas | greedy, N prompts | 0% | 0% | **≤ 5%** (bloqueante) |
| Diversidad | greedy, N prompts | ref | ≥ 0.5× ref | informe |
| Facts correctos | set factual | ref | ≥ 0.5× ref | informe |
| KL(P_base‖P_q2) | distribución | 0 | < K | informe |

**Regla de oro** (de `CE_VS_GENERATION.md`): la CE agregada NO predice calidad de
generación. El veredicto se toma con % degeneración + diversidad + facts, **no** con CE.

---

## 7. Riesgos y mitigación

| Riesgo | Mitigación |
|:---|:---|
| Q2_0 colapsa en el modelo completo (como en capas) | El umbral bloqueante de >5% degeneración lo descarta; Q4_0 queda como mínimo viable |
| El cuantizador base (min/max por bloque) es sensible a outliers | Fase post: Q2_0 + outliers a 4-bit |
| El cuerpo a 2-bit es inutilizable para fine-tune | No se entrena el cuerpo Q2_0; se documenta H3 como límite del formato |
| Sesgo de selección (tokens de entrenamiento en prompts) | Prompts estrictamente OOD + greedy |

---

## 8. Artefactos y comandos

- Prototipo refutación: `docs/research/temporal_4bit_fase1_test.py`
- Bloque actual: `src/io/header/blocks.rs` · formatos: `src/io/header/types.rs`
- Base de pesos: `src/nn/linear/database.rs` · forward: `src/nn/linear/forward.rs`
- Cuantizador Q4_0 (patrón): `src/compute/quantize/python.rs`
- Harness: `scripts/eval_generation.py` (`--models`, `--temp`, `--prompts-file`)
- Protocolo de métricas: `docs/plans/QUALITY_EVAL_PROTOCOL.md`
- Held-out: `data/distill/heldout_06b.json`

Corridas clave:
```
cargo test --release            # Fase 1
python scripts/build_q2_0.py    # Fase 2
python scripts/eval_generation.py --models qwen2_0_5b_q4_0.gaje.flat,qwen2_0_5b_q2_0.gaje.flat --temp 0.0 --prompts-file data/distill/heldout_06b.json  # Fase 3
```

---

## 9. Veredicto esperado

- **H1 (ahorro de memoria)**: se confirma (40%).
- **H2 (degradación tolerable)**: **dudosa** — el prototipo en capas sugiere que la
  degradación acumulada en 120 capas (~0.146) se traducirá en generación degradada;
  el harness decidirá.
- **H3 (no apto para fine-tune)**: se confirma por precedente (protocolo 0.6B).

Si H2 falla, el resultado neto del experimento es una **caracterización negativa de
calidad**: Q4_0 se mantiene como representación de producción y se documenta el límite
del 2-bit espacial. Si H2 pasa, se explora Q2_0 + outliers como formato de menor RAM.
