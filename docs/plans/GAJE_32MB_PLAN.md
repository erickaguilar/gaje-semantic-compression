# 🧬 Plan de Implementación: Organismo Silver Adult 32MB (`silver_adult_32m`)

Este documento detalla la hoja de ruta y especificaciones técnicas para la compilación, destilación y estabilización del organismo genómico **Silver Adult 32MB** (`silver_adult_32m`), definido recientemente como un puente crítico hacia la madurez del ecosistema GAJE.

---

## 1. 📐 Dimensionamiento y Arquitectura del Preset

El preset `silver_adult_32m` ha sido diseñado para equilibrar la fidelidad semántica en dispositivos de borde con restricciones de memoria estrictas (<32MB en disco/RAM).

*   **Identificador en CLI:** `silver_adult_32m`
*   **Parámetros Clave (SoA):**
    *   `n_embd` (Dimensión Oculta): `512`
    *   `n_blocks` (Capas Transformer): `8`
    *   `n_head` (Cabezas de Atención): `8` (GQA balanceado)
    *   `vocab_size`: `32768` (Vocabulario Silver Optimizado)
*   **Volumen de Parámetros:** ~67 Millones
*   **Presupuesto en Disco (Comprimido):** ~32 MB bajo almacenamiento binario `.gaje`

---

## 2. 🏛️ Pilares de Estabilidad de 32MB

Para lograr un rendimiento óptimo de esta configuración sin comprometer la homeostasis del modelo, se implementan tres pilares de estabilidad:

### A. Anclas de Estabilidad Adaptativas (Stability Anchors - F16)
*   **Densidad de Anclas:** Configurada en un `0.10` basal (con posibilidad de elevación adaptativa en capas lógicas/FFN).
*   **Mapeo:** Las anclas de precisión F16 se colocan estratégicamente en los pesos de proyección QKV y en las compuertas lógicas de SwiGLU.
*   **Prevención de Deriva:** Evitan que el error acumulado de la cuantización agresiva de 2 bits desvíe las activaciones.

### B. Confinamiento Toroidal $\mathbb{Q}(\zeta_{16})$
*   **Comportamiento:** En lugar de truncar valores extremos (que genera NaNs en contextos largos), la señal semántica recircula sobre las 16 fases discretas del toroide.
*   **Implementación:** Utilización de kernels nativos de fase compleja en `src/compute/math.rs` para realizar la rotación y el decaimiento.

### C. Inhibición Lateral K-WTA (K-Winner-Take-All)
*   **Mapeo Temporal:** Filtrado de ruido de fase mediante competencia local de activaciones. Solo los "canales de alta energía" se propagan, lo que no solo incrementa la coherencia sino que reduce la huella de cómputo en CPUs ARM.

---

## 3. 🛠️ Protocolo de Ejecución y Validación

El ciclo de vida del preset sigue el estándar del protocolo **GAJE-Flow (SDD -> BDD -> TDD)**.

### Paso 1: Inicialización Nativa (Nacimiento del Organismo)
Generación del andamiaje algebraico básico sin inicialización caótica aleatoria.
```bash
cargo run --release --bin gaje-cli -- --init models/silver_adult_32m_born.gaje --preset silver_adult_32m
```

### Paso 2: Diagnóstico de Estructura y Tamaños
Validación inmediata del empaquetamiento del archivo `.gaje` con el binario de diagnóstico recién incorporado.
```bash
# Copiar/enlace del archivo al target de diagnóstico e inspección
cargo run --release --bin inspect_sizes
```
> [!IMPORTANT]
> El tamaño total reportado por `inspect_sizes` no debe superar los 33,554,432 bytes (32 MB) para asegurar el cumplimiento del presupuesto del Edge SDK.

### Paso 3: Destilación por Consenso de Maestros (Breeding)
Entrenamiento nativo guiado mediante `Rayon` (Zero-GIL) usando mapas de activación.
*   **Maestro Primario:** SmolLM2-135M / Qwen2.5-0.5B.
*   **Dataset de Refinamiento:** Curaduría reducida en español e instrucciones lógicas.

---

## 4. 📈 Criterios de Aceptación (KPIs)

*   **PPL (Perplejidad Basal):** < 250 tras el ciclo inicial de crianza.
*   **Latencia en ARM (Edge-Core):** < 80ms para el primer token, > 100 tps sostenidos en arquitecturas multicore.
*   **Estabilidad en Contexto:** Cero NaNs tras secuencias continuas de 1024 tokens.

---

*Plan de trabajo vinculante bajo el protocolo de estabilidad Silver Adult v1.0.0-alpha.*
