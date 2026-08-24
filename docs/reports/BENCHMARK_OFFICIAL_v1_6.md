# 📊 GAJE Helix — Reporte Oficial de Benchmarks Científicos

**Fecha:** 2026-08-22 03:05:54  
**Versión del Motor:** GAJE v1.6.0-alpha (Rust SIMD AVX2 + PyO3 / Python 3.14.6)  
**Hardware de Referencia:** AMD Ryzen 7 5800H (16 hilos) - x86_64  

---

## 🏆 1. Resumen Comparativo de Modelos

| Modelo | Arquitectura | Tamaño Disco | Cold-Start | Peak RSS | Gen Speed | Diversidad ($d_1/d_2$) | Recall Semántico | Degeneración | Compresión |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`deepseek_r1_1_5b.flat`** | DeepSeek-R1 | 1263.5 MB | 2084.7 ms | 1987.0 MB | **7.0 tok/s** | 0.73 / 0.91 | 14.8% | 24.0% | **8.0x (87.5%)** |
| **`qwen2_0_5b.flat`** | Qwen2 | 498.5 MB | 1078.0 ms | 1198.6 MB | **17.8 tok/s** | 0.68 / 0.91 | 46.9% | 8.0% | **8.0x (87.5%)** |
| **`qwen2_5_3b.flat`** | Qwen2.5 | 2294.3 MB | 7099.3 ms | 3099.4 MB | **4.6 tok/s** | 0.73 / 0.93 | 59.7% | 8.0% | **8.0x (87.5%)** |
| **`smollm2_135m.flat`** | SmolLM2 | 473.2 MB | 1761.2 ms | 1380.0 MB | **17.0 tok/s** | 0.79 / 0.99 | 7.0% | 0.0% | **8.0x (87.5%)** |
| **`feto_genomico_v1.gaje`** | GAJE-Born | 2082.5 MB | 16459.2 ms | 1722.5 MB | **32.5 tok/s** | 1.00 / 1.00 | 3.2% | 0.0% | **8.0x (87.5%)** |

---

## 🔬 2. Definición de Métricas Evaluadas

* **Cold-Start (ms):** Tiempo para mapear el archivo binario a memoria vía Zero-Copy Mmap.
* **Peak RSS (MB):** Memoria física residente máxima alcanzada en el sistema.
* **Gen Speed (tok/s):** Throughput sostenido de generación autoregresiva token a token.
* **Diversidad ($d_1 / d_2$):** Fracción de unigramas y bigramas únicos generados (métrica Distinct).
* **Recall Semántico (%):** Porcentaje de conceptos clave esperados respondidos con éxito.
* **Degeneración (%):** Tasa de respuestas que caen en bucles repetitivos (objetivo: 0.0%).
* **Compresión:** Ratio de ahorro de memoria vs FP32 equivalente.

---
*Reporte generado automáticamente por `scripts/gaje_benchmark.py`.*