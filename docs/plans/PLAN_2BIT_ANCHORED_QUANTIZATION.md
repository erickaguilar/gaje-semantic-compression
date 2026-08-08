# 🧬 Plan de Implementación y Certificación: Modelo GAJE 2-Bit Anclado (v0.9.8-alpha)

> **Objetivo**: Reducir el consumo de memoria RAM viva de **448 MB (4-bit)** a **~224 MB (2-bit con 5% Stability Anchors)**, logrando una compresión del **93.75% (16.0x)** preservando la paridad factual y la paridad de Token IDs BPE.

---

## 📋 Fases del Plan de Ejecución

### 1. Fase 1: Exportador a 2-Bit con Anclajes de Estabilidad (Python/PyTorch)
- **Archivo**: `scripts/export_qwen2_2bit_anchored.py`
- **Mecánica**:
  - Cuantizar el 95% de los tensores FFN y Atención a **2-bit (4 centroides: A=00, C=01, G=11, T=10)**.
  - Empaquetar 4 pesos neuronales por cada Byte binario (`shift 6, 4, 2, 0`).
  - Preservar el **5% de los pesos con mayor gradiente/magnitud en FP16** (*Stability Anchors*).
- **Entregable**: `models/production/qwen2_0_5b_2bit_anchored.gaje.flat` (~920 MB en disco).

---

### 2. Fase 2: Soporte en Kernels Nativo Rust (Kernel AVX2 2-Bit)
- **Archivos**: `src/compute/kernels.rs` y `src/nn/layer.rs`
- **Mecánica**:
  - Optimizar el des-empaquetado de 2-bit en el bucle desenrollado `Unrolled 4-Way FMA` usando LUTs de 4 centroides en L1 Cache.
  - Ejecutar verificación de límites y validación de paridad de tensores.
- **Entregable**: Tests unitarios nativos pasando (`cargo test --release`).

---

### 3. Fase 3: Suite de Pruebas de Certificación Empírica
- **Prueba 3.1: Medición de RAM**: Confirmar consumo en RAM viva $\le 224\text{ MB}$.
- **Prueba 3.2: Velocidad Cold Start**: Validar carga `mmap` $< 1.5\text{ s}$.
- **Prueba 3.3: Auditoría Factual Multilingüe A/B**:
  - Español: *"¿Cuál es la capital de Francia?"*
  - Chino: *"太阳系中最大的行星是哪一颗？"* (Júpiter = 木星)
  - Conteo numérico: *"Count from 1 to 5"* (`1 2 3 4 5`).

---

### 4. Fase 4: Integración Web UI y Despliegue en Ramas Git
- **Archivos**: `examples/ui/web_ui/index.html`, `script.js` y `server.py`
- **Mecánica**:
  - Agregar la opción `🧬 QWEN2 0.5B 2-BIT ANCHORED (224 MB RAM / 16.0x)` al desplegable de modelos.
  - Validar ruff, pre-commit y tests de integración.
- **Entregable**: Commit y push a `origin/develop` y `origin/linux`.

---

## 🎯 Criterios de Certificación y Éxito
| Métrica | Umbral Requerido | Metodología de Validación |
| :--- | :---: | :--- |
| **Consumo de RAM Viva** | **$\le 224\text{ MB}$** | Profiling de memoria de proceso (`/proc/self/status` / `psutil`). |
| **Ratio de Compresión** | **`16.0x` (`93.75%`)** | Comparación directa contra modelo FP16 original. |
| **Precisión Factual** | **Paridad con FP32** | Test A/B multilingüe (Chino: 木星, Español: París). |
| **Cold Start mmap** | **$< 1.5\text{ segundos}$** | Benchmark de mapeo plano a disco `load_genomic`. |
