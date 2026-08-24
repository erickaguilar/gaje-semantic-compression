# 🧪 Plan Maestro de Pruebas y Verificación Integral — GAJE Helix Engine

**Fecha de Creación:** 22 de Agosto de 2026  
**Área:** Aseguramiento de Calidad (QA), Benchmarking & Validación de Inferencia Nativa  
**Estado:** [PLANNING / READY FOR EXECUTION] (Local / No Sincronizado)  
**Versión del Motor:** GAJE v0.9.7 (Rust SIMD AVX2 + PyO3 / Python 3.14.6)  

---

## 🎯 1. Objetivo del Plan

Garantizar la estabilidad, precisión semántica, eficiencia de memoria y rendimiento en tiempo real de todos los componentes del ecosistema **GAJE Helix**, abarcando:
1. El motor de inferencia nativo en Rust con aceleración SIMD AVX2 / FMA.
2. Los modelos transmutados (`.flat`) y nacidos de cero por GAJE (`.gaje`).
3. El subsistema de memoria episódica *Island Model* (`.gmem`).
4. La interfaz de usuario Web UI (Streaming SSE, Telemetría HUD y Exportación).
5. El prototipo de Tokenización Cuántico-Genómica.

---

## 📋 2. Matriz de Suites de Prueba

```
                                  ┌────────────────────────────────────────────────────────┐
                                  │           MATRIZ DE PRUEBAS INTEGRAL GAJE              │
                                  └────────────────────────────────────────────────────────┘
                                                               │
             ┌───────────────────────┬─────────────────────────┼─────────────────────────┬────────────────────────┐
             ▼                       ▼                         ▼                         ▼                        ▼
┌─────────────────────────┐ ┌─────────────────────────┐ ┌─────────────────────────┐ ┌─────────────────────────┐ ┌────────────────────────┐
│  SUITE 1: Inferencia    │ │   SUITE 2: Memoria      │ │   SUITE 3: Web UI &     │ │   SUITE 4: Benchmark    │ │  SUITE 5: Tokenizador  │
│  Zero-Copy Mmap         │ │   Purga Total & Leak    │ │   Telemetría en Vivo    │ │   Científico (PPL/TPS)  │ │  Cuántico-Genómico     │
│  Validación de Pesos    │ │   malloc_trim(0)        │ │   SSE / Badges / Logs   │ │   gaje benchmark        │ │  Superposición ρ(4x4)  │
└─────────────────────────┘ └─────────────────────────┘ └─────────────────────────┘ └─────────────────────────┘ └────────────────────────┘
```

---

## 🔬 3. Detalle de Suites de Prueba

### Suite 1: Validación de Inferencia Nativa & Modelos Registrados
* **Objetivo:** Verificar que todos los modelos binarios carguen vía `mmap` y generen texto coherente sin bucles infinitos ni excepciones.
* **Modelos a Evaluar:**
  - `models/production/qwen2_5_3b.flat` (2.24 GB)
  - `models/production/deepseek_r1_1_5b.flat` (1.23 GB)
  - `models/born/feto_genomico_v1.gaje` (2.08 GB — Modelo nacido por GAJE)
  - `models/production/qwen2_0_5b.flat` (499 MB)
  - `models/production/smollm2_135m.flat` (474 MB)
* **Casos de Prueba:**
  1. **TC-1.1:** Carga instantánea Mmap (< 6.0s en frío, < 0.1s en caliente).
  2. **TC-1.2:** Detección automática de arquitectura desde la cabecera binaria (Qwen2.5, DeepSeek-R1, SmolLM2).
  3. **TC-1.3:** Inferencia en español, inglés, portugués y japonés con parada limpia en tokens de detención (`<|im_end|>`, `<end_of_turn>`).

---

### Suite 2: Gestión de Memoria y Purga Agresiva (`unload_model`)
* **Objetivo:** Asegurar que la liberación de modelos devuelva el 100% de la memoria residente (RSS) al kernel de Linux mediante `malloc_trim(0)` sin fugas de memoria (*memory leaks*).
* **Casos de Prueba:**
  1. **TC-2.1:** Medir RSS base del servidor (esperado: ~45 MB).
  2. **TC-2.2:** Cargar `qwen2_5_3b.flat` (RSS sube a ~2.58 GB).
  3. **TC-2.3:** Invocar botón o endpoint de descarga (`/api/unload_model`).
  4. **TC-2.4:** Verificar que RSS retorne al nivel base (~45-60 MB) inmediatamente.
  5. **TC-2.5 (Estrés):** Ejecutar 10 ciclos consecutivos de carga/descarga alternando entre modelos grandes y pequeños.

---

### Suite 3: Web UI, Streaming SSE y Telemetría en Vivo
* **Objetivo:** Validar la experiencia de usuario completa y el cálculo exacto de tokens y compresión.
* **Casos de Prueba:**
  1. **TC-3.1 (Streaming SSE):** Emisión fluida de tokens por segundo con renderizado Markdown progresivo.
  2. **TC-3.2 (Cálculo de Tokens):** Verificación de que los badges muestren el desglose exacto:
     - `🔢 Total tokens (p prompt + g generados)`
     - `🧬 Cuantización Q4_0 (8.0x · 87.5% ahorro)`
     - `⚡ Velocidad tok/s y Latencia total`
  3. **TC-3.3 (Modal HUD):** Abrir el modal desde el nuevo botón en `chat-toolbar` y conmutar entre pestañas (*Métricas Genómicas*, *Island Model*, *Hardware*, *Alertas*).
  4. **TC-3.4 (Exportar Bitácora):** Descarga del archivo `.md` con marcas de tiempo por turno, especificaciones de hardware y log de alertas.

---

### Suite 4: Suite de Benchmarks Automatizada (`gaje benchmark`)
* **Objetivo:** Medir perplejidad, ratio de compresión genómica y latencia científica.
* **Métricas a Registrar:**
  - **Perplejidad (PPL):** Evaluación en WikiText-2 / C4.
  - **TTFT (Time-To-First-Token):** Tiempo hasta emitir el primer token (< 80 ms).
  - **Throughput de Inferencia:** Tokens por segundo sostenidos con vectorización SIMD AVX2.
  - **Latencia de Retrieval Island Model:** Búsqueda asociativa en `.gmem` (< 1.0 ms).

---

### Suite 5: Prototipo de Tokenización Cuántico-Genómica
* **Objetivo:** Validar los principios descritos en `docs/research/QUANTUM_GENOMIC_TOKENIZATION_FINDINGS.md`.
* **Casos de Prueba:**
  1. **TC-5.1:** Mapeo de bases genómicas a vectores de estado $|00\rangle, |01\rangle, |10\rangle, |11\rangle$.
  2. **TC-5.2:** Construcción de matriz de densidad $\\rho \\in \\mathbb{C}^{4 \\times 4}$ para palabras polisémicas.
  3. **TC-5.3:** Colapso contextual proyectivo con vector del *Island Model* en $< 1\ \\mu\\text{s}$.
  4. **TC-5.4:** Medición de reducción de vocabulario (compresión de tabla de embeddings de 600 MB a < 4.5 MB).

---

## 🛠️ 4. Cronograma y Pasos de Ejecución

| Paso | Actividad | Herramienta / Comando | Responsable |
| :---: | :--- | :--- | :--- |
| **Fase A** | Ejecución de Suite 1 & 2 (Inferencia y Purga de RAM) | Script automatizado Python/Rust | GAJE Core |
| **Fase B** | Validación de Suite 3 (Web UI, Telemetría y Tokens) | Test E2E / Curl & Browser | Frontend / UI |
| **Fase C** | Ejecución de Suite 4 (Benchmarks PPL y Throughput) | CLI \`gaje benchmark\` | QA / Benchmarking |
| **Fase D** | Validación de Suite 5 (Prototipo Cuántico-Genómico) | Test unitario en Python/Rust | Research Team |

---
*Fin del plan maestro de pruebas. Archivo registrado localmente en \`docs/plans/TESTING_AND_VERIFICATION_PLAN.md\`.*
