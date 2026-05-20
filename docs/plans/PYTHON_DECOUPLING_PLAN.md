# 🧬 Plan de Desacoplamiento de Python: Hacia la Natividad Total

Este documento define la estrategia para reducir y eventualmente eliminar la dependencia de Python en el ecosistema **GAJE-Flow**, permitiendo su despliegue como una librería nativa pura en Android/iOS y dispositivos embebidos.

---

## Nivel 1: Autonomía de Adaptación (Balancer Nativo)
**Objetivo:** Portar el `SignalToNoiseBalancer` de Python a Rust.

*   **Estado Actual:** Python analiza la entropía de los tensores y genera máscaras de precisión para decidir qué pesos necesitan protección de anclas.
*   **Acción:** Implementar el algoritmo de análisis de señal/ruido en `src/compute/math.rs`.
*   **Beneficio:** El motor podrá auto-configurarse durante la carga o evolución sin intervención de scripts externos, reduciendo la latencia de inicialización.

## Nivel 2: Soberanía de Tokenización (Native Tokenizers)
**Objetivo:** Eliminar la dependencia de la librería `transformers` de Hugging Face.

*   **Estado Actual:** El proyecto depende de Python para descargar y cargar `tokenizer.json`, así como para aplicar plantillas de chat (Chat Templates).
*   **Acción:** Integrar profundamente la crate `tokenizers` de Rust en `src/io/loader.rs` y `src/bin/gaje-cli.rs`.
*   **Beneficio:** Eliminación de dependencias pesadas de Python (PyTorch/Transformers). El binario de Rust podrá procesar texto crudo directamente, permitiendo apps de chat 100% nativas y ligeras.

## Nivel 3: Entrenamiento Genómico Puro (Native Loss & Auto-Grad)
**Objetivo:** Mover el bucle de entrenamiento y cálculo de pérdida al núcleo de Rust.

*   **Estado Actual:** El cálculo de `Softmax`, `CrossEntropy` y la orquestación del gradiente ocurre en Python, enviando datos de ida y vuelta a Rust.
*   **Acción:** Implementar funciones de pérdida (`loss`) y el orquestador del paso de entrenamiento (`train_step`) dentro de `RustGenomicLLM`.
*   **Beneficio:** Incremento de velocidad de entrenamiento estimado en 10x-20x. Permite el "Aprendizaje Continuo en el Dispositivo" sin el consumo de memoria y CPU que requiere mantener un intérprete de Python activo en segundo plano.

---

## Hoja de Ruta Sugerida
1.  **Fase 1:** Implementar **Nivel 1** en la rama actual para estabilizar la auto-genomización.
2.  **Fase 2:** Migrar la carga de modelos 100% a Rust (**Nivel 2**).
3.  **Fase 3:** Consolidar el entrenamiento nativo (**Nivel 3**) para habilitar la evolución local masiva.

*Documento generado como guía técnica para la transición a GAJE-Native v0.8+.*
