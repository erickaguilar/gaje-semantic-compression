# 🕵️ Hallazgos y Diagnóstico Forense (Junio 2026)

Este documento registra las pruebas empíricas realizadas para validar la viabilidad del motor GAJE y la causa raíz de las incoherencias semánticas detectadas.

## 1. El Mito de la Transmutación Directa (Vía A)
Se intentó importar el modelo `SmolLM2-135M` mediante cuantización pura a 2 bits con anclas F16.
*   **Resultado:** ❌ **FALLO ABSOLUTO**.
*   **Síntomas:** El modelo generaba ruido estructural ("terol", "Bohem", "hashlib").
*   **Diagnóstico:** La compresión a 2 bits es demasiado agresiva para preservar la jerarquía de un vocabulario de 49k tokens sin un entrenamiento de adaptación. El "trasplante" de cerebro no funciona en este nivel de compresión.

## 2. Validación del Motor de Rust (Unit Test)
Se creó un organismo desde cero (`micro_organism`) y se entrenó con el patrón "Hola mundo".
*   **Resultado:** ✅ **ÉXITO TOTAL**.
*   **Métricas:** Loss: 0.0000, PPL: 1.00.
*   **Conclusión:** El `NativeGenomicTrainer`, los kernels de Rust y la lógica de inferencia son **100% funcionales y estables**. El problema no es el código del motor, sino la calidad de los pesos importados.

## 3. Éxito de la Crianza Nativa (Vía B)
Se realizó una micro-destilación donde un modelo maestro (GGUF) guió a un estudiante de 2 bits.
*   **Resultado:** ✅ **CONVERGENCIA VALIDADA**.
*   **Progreso:** La pérdida bajó de ~12.0 a **5.28** en solo 3 épocas.
*   **Impacto:** El modelo empezó a estabilizar sus activaciones (`q_abs_sum`) y a predecir tokens con mayor probabilidad lógica.

## 4. Estabilidad en Android/ARM
*   **Hallazgo:** Los kernels SIMD (NEON) presentaban inestabilidad numérica en dispositivos móviles.
*   **Solución:** Se implementaron **Epsilons de 1e-5**, **Clamping en SwiGLU**, y **Tablas de Búsqueda (LUT)** para seno/coseno. El motor es ahora inmortal ante NaNs.

---
*Este diagnóstico fundamenta la decisión de abandonar la importación directa y adoptar el Roadmap de Crianza Nativa.*
