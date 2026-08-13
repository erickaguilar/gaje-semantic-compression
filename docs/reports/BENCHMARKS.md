# 📊 BENCHMARKS — GAJE Helix v1.6.0-alpha

> **Benchmark oficial del motor GAJE.** Medición reproducible del formato `.gaje.flat` v2 (`Q4_0` cuerpo + `Q8_0`/`FP32` embeddings) sobre CPU x86_64.

---

## 1. Entorno

| | |
| :--- | :--- |
| **CPU** | AMD Ryzen 7 5800H (Zen 3, 8C/16T) |
| **RAM** | 12 GB |
| **SO** | Linux (Fedora 43) |
| **Runtime** | Python 3.14 · PyO3 (extensión **release**) |
| **Versión** | GAJE Helix **v1.6.0-alpha** |
| **Modelos** | `models/production/*.gaje.flat` |

## 2. Metodología (reproducible)

- **Compresión:** `os.path.getsize` sobre el archivo plano + parámetros nominales públicos del checkpoint HF.
- **Velocidad/Latencia:** ruta de producción `rust_llm.generate_native_py(tokens, 48, temp=0.2, rep_penalty=1.1, eos)` — la misma config de la Web UI (`server.py`).
  - **TTFT:** llamada de 1 token sobre caché fría (incluye prefill).
  - **Throughput:** generación de 48 tokens, `tokens/segundo`.
- **Memoria:** `/proc/self/status` (VmRSS/VmSize).
- **PPL:** log-verosimilitud desplazada, softmax sobre vocabulario completo, corpus ES de dominio filtrado.

Scripts: `scripts/benchmarks/engine_benchmark.py`, `scripts/benchmarks/ppl_suite.py`, `scripts/benchmarks/ppl_parity_fp16.py`.

---

## 3. Compresión (tamaño real vs FP16/FP32)

| Modelo | Archivo | b/peso | bits/peso | vs FP16 | vs FP32 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen2-0.5B (`q8_0_embd`) | 498.5 MB | 1.06 | **8.5** | −49.5% | −74.8% |
| Qwen2.5-1.5B (`q4_0`, emb FP32) | 2571.0 MB | 1.75 | 14.0 | −16.5% | −58.3% |
| Qwen2.5-1.5B (`q8_0_embd`) | 1263.5 MB | 0.86 | **6.9** | −59.0% | −79.5% |
| Qwen2.5-3B (`q8_0_embd`) | 2294.3 MB | 0.78 | **6.2** | −62.9% | −81.4% |
| SmolLM2-135M (`4bit`) | 471.4 MB | 3.66 | 29.3 | **+74.6%** ⚠️ | −12.7% |

> **Hallazgo honesto:** la compresión **efectiva es ~6–8 bits/peso**, no 4. El nombre `Q4_0` se refiere a los bloques del cuerpo; al sumar embeddings en `Q8_0`/`FP32` y overhead, el promedio sube. Un modelo pequeño con embeddings FP32 puede incluso **superar a su versión FP16** (SmolLM2-135M: +74.6%).

---

## 4. Velocidad y latencia (ruta de producción)

| Modelo | Carga (mmap) | TTFT (1er token) | 48 tok | tok/s |
| :--- | :---: | :---: | :---: | :---: |
| Qwen2-0.5B (`q8_0_embd`) | 0.92 s | 880 ms | 2.33 s | **20.63** |
| Qwen2.5-1.5B (`q4_0`, emb FP32) | 11.60 s | 2678 ms | 6.65 s | 7.22 |
| Qwen2.5-1.5B (`q8_0_embd`) | 6.87 s | 2347 ms | 5.10 s | **9.41** |
| Qwen2.5-3B (`q8_0_embd`) | 10.25 s | 3828 ms | 9.10 s | **5.28** |
| SmolLM2-135M (`4bit`) | 1.84 s | 1348 ms | 2.58 s | 18.62 |

> El throughput varía con el prompt y la longitud; estos son valores en generación de 48 tokens con sampler estable (T=0.2, pen=1.1). **La variante `q8_0_embd` del 1.5B es ~30% más rápida y la mitad de tamaño que la `q4_0` (emb FP32).**

---

## 5. Memoria (RSS, con advertencia)

| Modelo | VMS tras carga | RSS tras carga | RSS tras generar |
| :--- | :---: | :---: | :---: |
| Qwen2-0.5B | 5.49 GB | 1.36 GB | 1.36 GB |
| Qwen2.5-1.5B (`q8_0_embd`) | 8.22 GB | 2.88 GB | 2.88 GB |
| Qwen2.5-3B | 9.40 GB | 3.72 GB | 3.87 GB |

> ⚠️ **Advertencia metodológica:** estos RSS se midieron en un **mismo proceso que cargó los modelos secuencialmente**, por lo que las páginas mmap de modelos previos se acumulan. El RSS aislado por modelo sería menor. Medición limpia por proceso independiente: **pendiente**.

---

## 6. Calidad (PPL)

| Modelo | PPL GAJE | PPL FP16 (ref) | Ratio |
| :--- | :---: | :---: | :---: |
| Qwen2-0.5B | 403.6 | 176.9 | 2.28 |
| Qwen2.5-1.5B (`q4_0`, emb FP32) | 130.7 | — | — |
| Qwen2.5-1.5B (`q8_0_embd`) | 130.4 | — | — |
| Qwen2.5-3B | 1213 (no fiable) | — | — |
| SmolLM2-135M | 461.2 | — | — |

> **Advertencias:**
> - La PPL **absoluta es inestable** sobre corpus de dominio corto (varía mucho con muestra/longitud). No es una métrica de calidad estable; **la métrica robusta es la correlación de ranking**.
> - **Correlación GAJE vs FP16 = 0.87–0.93** (Qwen2-0.5B, corpus mayor): el motor conserva el *orden* de probabilidades aunque no los valores absolutos. Es decir, conserva **decisiones** (argmax), no valoraciones.
> - **Hallazgo:** las variantes 1.5B con embeddings **Q8_0 vs FP32 dan PPL casi idéntica** (130.4 vs 130.7) con la mitad de tamaño → el diseño `q8_0_embd` es estrictamente mejor en tamaño/velocidad sin pérdida de calidad en este corpus.

---

## 7. Hallazgos honestos y pendientes

### ✅ Confirmado
1. Compresión efectiva **~6–8 bits/peso** (no 4).
2. **Rendimiento release real:** 5–20 tok/s según modelo (3B ≈ 5.3 tok/s).
3. **Paridad de decisión** (correlación de ranking 0.87–0.93), no paridad de PPL absoluta.
4. **Q8_0 embeddings ≈ FP32** en calidad, con mitad de tamaño → `q8_0_embd` es el formato recomendado.

### 🔜 Pendiente
- **Columna 4-bit total** (llama.cpp Q4_0 puro) para aislar el efecto de la preservación de embeddings.
- **PPL en corpus limpio held-out** (subconjunto Wikitext/El País ES), ≥100 muestras.
- **RSS aislado por modelo** (proceso independiente por modelo).
- **Paridad FP16 para 1.5B y 3B** (RAM limitada: 3B FP16 ≈ 6 GB no cupo).

---

## 8. Reproducción

```bash
# Compresión / memoria / velocidad (todos los modelos)
python scripts/benchmarks/engine_benchmark.py --gen_tokens 48

# PPL GAJE (por modelo, para limitar RAM) + paridad FP16
python scripts/benchmarks/ppl_suite.py --only qwen2_0_5b --samples 30 --max_len 48
```
