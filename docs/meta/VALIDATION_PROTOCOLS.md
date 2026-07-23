# 🛡️ Protocolos de Validación Empírica (v1.0)

Este documento define los procedimientos técnicos y métricas de éxito para validar las capacidades disruptivas del proyecto **DNA Semantic Compression**. Ninguna innovación se considera "activa" hasta que supere con éxito estos protocolos.

---

## 1. Validación de Compresión y Fidelidad (Resonancia Semántica)
*   **Objetivo:** Garantizar que la reducción a 2 bits mantiene la lógica del modelo maestro.
*   **Procedimiento:**
    1.  Medir la **Perplejidad (PPL)** de un modelo maestro (FP16) sobre un dataset de referencia (ej: WikiText-2).
    2.  Convertir el modelo al protocolo GAJE (2-bits + Stability Anchors).
    3.  Medir la PPL del modelo convertido sobre el mismo dataset.
*   **Métrica de Éxito:** La degradación de PPL debe ser **≤ 4.0% relativo** frente al modelo maestro.
*   **Herramienta:** `pytest tests/metrics/test_perplexity.py`.

## 2. Validación de Ingestión Neuronal Directa (DNI)
*   **Objetivo:** Verificar la capacidad de aprendizaje instantáneo sin olvido catastrófico.
*   **Procedimiento:**
    1.  **Baseline:** Solicitar al modelo un dato específico que NO posee (ej: una clave de sesión efímera).
    2.  **Acción:** Ejecutar `gaje-cli ingest --text "Dato Específico"`.
    3.  **Verificación:** Solicitar nuevamente el dato al modelo.
*   **Métrica de Éxito:** 100% de acierto en la recuperación del dato inyectado y **ΔPPL < 1%** en el conocimiento general base.
*   **Herramienta:** `python examples/core_demos/chat_soberano.py`.

## 3. Validación de Motor Neuromórfico (Eficiencia Energética)
*   **Objetivo:** Confirmar el bajo consumo y la activación por eventos (Sparsity).
*   **Procedimiento:**
    1.  Medir el Wattaje consumido durante 1 minuto de inferencia continua en un dispositivo ARM (Android).
    2.  Registrar el ratio de activación de neuronas (spikes disparados vs neuronas totales).
*   **Métrica de Éxito:** Consumo energético **< 0.5W** y Sparsity temporal **> 80%** (8 de cada 10 neuronas en reposo).
*   **Herramienta:** `python benchmarks/performance/latencies_and_throughput.py`.

## 4. Validación de Topología Circular (Estabilidad de Contexto)
*   **Objetivo:** Asegurar que la memoria no se degrada en contextos masivos.
*   **Procedimiento:**
    1.  **Prueba "Needle in a Haystack":** Insertar un dato único en la posición 0 de un contexto de 128k tokens.
    2.  Recuperar el dato al final de la secuencia.
*   **Métrica de Éxito:** **100% Accuracy** (Recall perfecto) independientemente de la profundidad de la "aguja" en el contexto.
*   **Herramienta:** `python scripts/benchmarks/needle_haystack.py`.

## 5. Validación de Soberanía Nativa (Independencia de Frameworks)
*   **Objetivo:** Garantizar el funcionamiento autónomo del núcleo en Rust.
*   **Procedimiento:**
    1.  Compilar el binario único: `cargo build --release --bin gaje-cli`.
    2.  Ejecutar inferencia en un entorno con un intérprete de Python desinstalado o bloqueado.
*   **Métrica de Éxito:** Ejecución exitosa de inferencia y evolución con un binario de **< 20MB** sin dependencias dinámicas externas.
*   **Herramienta:** Binario ejecutable en `./target/release/gaje-cli`.

---
*Este marco de validación es vinculante para certificar la madurez del organismo computacional en su fase Silver Adult.*
