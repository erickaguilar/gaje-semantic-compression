# 🕸️ Paso 2 — Spike de Implementación HNSW: Hallazgos y Decisión

> Fecha: 2026-08-23 · Estado: **COMPLETADO — decisión tomada con datos**
> Precedido por: `benchmarks/logs/hnsw_gate_step1_results.json` (gate lineal falla desde ~5k entradas)
> Escenario de prueba: vectores f32 normalizados, dim=768, k=10, Ryzen 7 5800H

---

## 1. Candidatos evaluados

| Crate | Versión | Build | Búsqueda p50 @N | Recall@10 | Veredicto |
|-------|---------|-------|-----------------|-----------|-----------|
| `instant-distance` | 0.6.1 | 147 inser/s @20k | 3.4 ms (@ef64) / 9.1 ms (@ef256) | 39% / 82.5% | ❌ Descalificado |
| `hnsw_rs` (paralelo) | 0.2.1 | 541 inser/s @100k | 10.6 ms @100k (ef128) | **22%** | ❌ Descalificado |
| `hnsw_rs` (secuencial) | 0.2.1 | **81 inser/s @50k** | 8.7 ms @50k | **34%** | ❌ Descalificado |
| Harness sanity (N=1k) | — | 10,877 inser/s | 788 µs | **100%** | ✅ Harness correcto |

### Detalle clave

1. **`instant-distance`**: API sólida pero rendimiento patológico (build 147/s vs ~10k/s esperable;
   búsqueda 15× sobre lo razonable). Recall aceptable solo con ef muy alto, que agrava latencia.
2. **`hnsw_rs`**: el recall colapsa a escala (22–34%) **incluso con inserción secuencial** —
   descartado que sea defecto del `parallel_insert`. El build además se degrada con N
   (10.8k/s @1k → 81/s @50k). Problema estructural del crate en dim alta.
3. El harness está validado: N=1000 produce recall 100% contra brute-force exacto.

## 2. Insight estratégico que revela el spike

El costo dominante NO es la navegación del grafo sino **el costo por distancia**: cada
comparación coseno 768-d escalar cuesta ~200 ns (medido en el paso 1: 20.73 ms/100k).
Un HNSW con ef=128 evalúa ~4k distancias por query → ~0.8 ms *en teoría ideal*. Los crates
fallan porque multiplican ese costo base por mal layout/SIMD, y su grafo degrada el recall.

Esto abre una **tercera vía** que los crates no ofrecen: atacar el costo por distancia
con la maquinaria que GAJE ya tiene certificada.

## 3. Decisión: camino nativo sin grafo HNSW

**Opción elegida: SIMD kernel + particionado grueso (IVF-lite) sobre la maquinaria ADC existente.**

| Componente | Reutiliza | Ganancia esperada |
|------------|-----------|-------------------|
| Kernel SIMD L2/coseno para `.gmem` | Patrón de `genomic_dot_product_*` AVX2/FMA ya certificado | 20.7 ms → **~2-4 ms** @100k (×5-10 por vectorización) |
| Particionado IVF-lite (k-means 256 clústeres, sondeo top-8) | `quantize_embedding` + umbrales existentes | ~4 ms → **<1 ms** @100k (÷8 por sondeo parcial) |
| Formato binario | Byte `index_type=1` ya reservado en cabecera `.gmem` v2 | Compatibilidad total hacia atrás |

**Por qué no un grafo HNSW propio**: el spike demuestra que implementar HNSW correcto es
justamente lo que ambos crates fallan en hacer (recall colapsado a escala). Reproducir ese
riesgo internamente para luego competir contra hnswlib C++ no pasa el análisis costo/beneficio.

**Por qué IVF-lite sí**: sin grafo, sin punteros entre nodos, recall controlable por diseño
(sondear más clústeres = más recall, determinista), y cada pieza usa primitivas ya validadas
del codebase (Recall@10 85.3% certificado en búsqueda ADC).

## 4. Gates propuestos para la implementación (paso 3)

| Métrica | Umbral |
|---------|--------|
| Latencia p50 @100k entradas | < 1 ms |
| Recall@10 vs brute-force | ≥ 85% (par con el certificado ADC) |
| Tiempo de indexación @100k | < 30 s |
| Regresión formato | `.gmem` v1/v2 existentes cargan intactos |

## 5. Artefactos

- Script del paso 1: `benchmarks/research/hnsw_gate_step1_linear_scan.py`
- Datos del paso 1: `benchmarks/logs/hnsw_gate_step1_results.json`
- Proyecto spike (aislado): `/tmp/opencode/hnsw_spike` — bins `hnsw_spike` (instant-distance)
  y `spike_hnswrs`, parametrizables por CLI (`N dim [seq]`, `N ef`)

---
*Micro-plan HNSW Paso 2 — Donde medir antes de construir ahorró dos crates y una semana.*
