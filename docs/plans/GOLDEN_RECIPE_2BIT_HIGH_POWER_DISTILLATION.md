# 🧬 Plan Maestro: La Receta Dorada para un LLM Coherente en 2-Bits (Entrenamiento y Destilación en Alta Potencia)

**Estado:** Especificación de Ingeniería y Protocolo de Reproducción en GPU  
**Versión:** 1.0  
**Fecha:** Septiembre 2026  
**Ámbito:** Escalado en Cluster / GPU (RTX / Colab / Kaggle) · Destilación DNI Multi-Maestro · Arquitectura Híbrida Q2_0/FP32

---

## 1. 🎯 Diagnóstico Científico: La Trampa de la "Fuerza Bruta"

La evidencia empírica acumulada en el ecosistema GAJE (reportes `BORN_Q2_0_FAILURE_FINDINGS.md` y experimentos en CPU/móvil) demostró de forma concluyente que:
* **Entrenar por más horas una arquitectura 2-bit pura ($Q2\_0$ total) NO produce coherencia.**
* Tras 20 épocas continuas (8,494s) en una estación de trabajo, el modelo alcanzó una meseta en $\text{Loss} \approx 3.80$ y $PPL \approx 45$. El modelo memorizó fragmentos léxicos aislados (*"Tierra"*, *"Lógica"*, *"cuerpo"*), pero fue incapaz de encadenar gramática sintáctica.
* **Causa Física:** Con $D=384$ y $V=49,152$, una matriz de proyección final (`lm_head`) cuantizada a solo 2 bits/peso sufre de un hacinamiento vectorial extremo ($\rho = V/D > 100$). Miles de palabras colisionan en la misma región del hiperplano, impidiendo que el optimizador discrimine el token correcto.

---

## 2. 🏛️ La Receta Dorada: Arquitectura Híbrida Desacoplada

Para romper la barrera de coherencia en una GPU o cluster de alta potencia, se debe abandonar la cuantización homogénea y aplicar la **tríada desacoplada**:

```
                 ARQUITECTURA DE LA RECETA DORADA (~24 MB TOTAL)
                 
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 1. CUERPO DEL TRANSFORMER (12 Capas · D=384 · SwiGLU)                 │
  │    • Cuantización: Q2_0_CONFORMAL (2.0 bits/peso en C)                 │
  │    • Tamaño en disco: ~19.72 MB                                        │
  │    • Reside íntegro en memoria caché L3 del CPU                        │
  └──────────────────────────────────┬─────────────────────────────────────┘
                                     │ Activaciones D=384
                                     ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 2. CABEZA DE PROYECCIÓN (lm_head) DESACOPLADA                          │
  │    • Cuantización: FP32 o Q8_0 (8.0 bits/peso)                         │
  │    • Vocabulario Calibrado: 4,096 tokens esenciales (GTOK 4K)          │
  │    • Presión Vectorial: rho = 4096 / 384 = 10.6 (Optima)               │
  │    • Tamaño adicional: ~4.1 MB                                         │
  └──────────────────────────────────┬─────────────────────────────────────┘
                                     │ Distribución de Probabilidades
                                     ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 3. HIPOCAMPO PERSISTENTE CONGÉNITO (.gmem)                             │
  │    • Zero-Training RAG Cache (< 0.12 ms mmap)                          │
  │    • Almacena hechos fácticos, fechas y conocimiento enciclopédico     │
  └────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 🎓 Protocolo de Destilación por Consenso (`CouncilOfTeachers`)

En el equipo potente con GPU (ej. Google Colab T4 / RTX 3090 / A100), el alumno no se entrenará como un LLM ciego sobre texto plano. Se utilizará la señal *dark knowledge* de dos maestros complementarios:

1. **Maestro de Razonamiento (CoT):**  
   * **Modelo:** `deepseek-r1-1.5b` o `deepseek-r1-7b`.
   * **Objetivo:** Enseñar al micro-modelo a formular monólogos internos antes de responder (`<think>...</think>`).
2. **Maestro de Sintaxis y Fluidez en Español:**  
   * **Modelo:** `qwen2.5-3b-instruct`.
   * **Objetivo:** Proporcionar la distribución de probabilidad de lenguaje natural coherente.
3. **Pérdida por Divergencia KL Ponderada:**
   $$\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{CE}}(y_{\text{gold}}, \hat{y}) + (1 - \alpha) \cdot \tau^2 \cdot \mathcal{D}_{\text{KL}}\left( \sigma\left(\frac{z_{\text{maestro}}}{\tau}\right) \parallel \sigma\left(\frac{z_{\text{alumno}}}{\tau}\right) \right)$$
   * Temperatura de suavizado: $\tau = 2.0$
   * Balance de destilación: $\alpha = 0.3$ (70% del peso en la imitación del maestro).

---

## 4. 🚀 Pipeline de Reproducción Paso a Paso para la GPU

### Paso 1: Clonar e Instalar en la Máquina Potente
```bash
git clone https://github.com/erickaguilar/gaje-semantic-compression.git
cd gaje-semantic-compression
cargo build --release --bin gaje-cli --bin distill_run
```

### Paso 2: Generar el Corpus Curado con Razonamiento CoT (2,000 Pares)
```bash
python scripts/data_processing/generate_distill_corpus.py \
  --teacher-model models/production/qwen2_5_3b.flat \
  --output data/distill/curated_2k_high_coherence.jsonl \
  --samples 2000 \
  --include-cot
```

### Paso 3: Dar a Luz al Organismo Base con Cabecera Desacoplada
```bash
./target/release/gaje-cli birth \
  --name max_gold_2bit \
  --dim 384 \
  --layers 12 \
  --heads 6 \
  --ffn-dim 1024 \
  --vocab-size 4096 \
  --tokenizer data/gtok_human_4k.bin \
  --with-memory \
  --output models/born/max_gold_2bit.gaje
```

### Paso 4: Ejecutar la Destilación Acelerada por GPU
```bash
# En GPU (Vulkan / CUDA / ROCm), 50 épocas toman ~2 horas en total
./target/release/distill_run \
  --student models/born/max_gold_2bit.gaje \
  --teacher models/production/gaje_prime_3b.flat \
  --dataset data/distill/curated_2k_high_coherence.jsonl \
  --epochs 50 \
  --lr 0.003 \
  --batch-size 64 \
  --output models/production/max_gold_2bit_certified.gaje
```

### Paso 5: Auditar y Validar Criterios de Éxito (*Definition of Done*)
```bash
# 1. Chequeo de integridad tensorial
./target/release/gaje-cli audit models/production/max_gold_2bit_certified.gaje

# 2. Evaluación objetiva de perplejidad y fluidez
./target/release/gaje-cli benchmark \
  --model models/production/max_gold_2bit_certified.gaje \
  --tokens 64 \
  --suite full
```

---

## 5. 📊 Tabla de Métricas Objetivo para Certificación

| Métrica | Estado Previo (2-Bit Puro Móvil) | Objetivo en GPU con la Receta Dorada | Veredicto Requerido |
| :--- | :---: | :---: | :---: |
| **Tamaño Total del Archivo** | 19.72 MB | **~23.8 MB** (Cuerpo Q2_0 + Head FP32) | 🟢 Cabe en Caché L3 |
| **Perplejidad Held-Out (PPL)** | $45.0$ (Meseta plana) | **$< 12.0$** | 🟢 Gramática Preservada |
| **Correlación de Ranking ($r$)** | $0.42$ | **$> 0.88$** | 🟢 Decisiones Alineadas |
| **Tasa de Degeneración** | $0.0\%$ (Emisión de 1 token) | **$0.0\%$** (Secuencias fluidas de 64 tokens) | 🟢 Coherencia E2E |
| **Throughput en ARM Móvil** | ~20 tok/s | **`35 - 45 tok/s`** (Con kernel NEON optimizado) | 🟢 Ultra-Baja Latencia |

---

## 6. Conclusión Estratégica

La barrera del LLM en 2-bits no se supera con tiempo bruto sobre una arquitectura colapsada; se supera **eliminando el cuello de botella del vocabulario y nutriendo al modelo con la distribución de un consejo de maestros**. Este documento queda fijado como la guía maestra para el momento en que se disponga del entorno GPU para dar a luz al primer micro-LLM conversacional sub-25MB de GAJE.
