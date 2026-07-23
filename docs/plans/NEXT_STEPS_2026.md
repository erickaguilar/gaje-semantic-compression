# 🚀 Próximos Pasos: Operación Rescate (Q3 2026)

Tras la homologación empírica del proyecto, se ha definido una hoja de ruta crítica para rescatar el valor del experimento. Nos enfocaremos en una **estrategia de doble vía**: certificar la capacidad de almacenamiento (L1) mientras intentamos arreglar la coherencia lingüística (L2).

---

## 🏔️ Vía A: Certificación de Resonancia (Nivel 1)
**Objetivo:** Demostrar que el modelo es un contenedor de datos de ultra-alta densidad, independientemente de su fluidez al hablar.

1.  **Ejecutar Needle In A Haystack (128k):**
    *   **Acción:** Usar `scripts/benchmarks/needle_test.py` para inyectar una "aguja" (un dato específico) en un contexto de 128,000 tokens.
    *   **Meta:** 100% de precisión en la recuperación del dato mediante resonancia de fase compleja.
    *   **Impacto:** Si tiene éxito, certificamos el **Nivel 1** y validamos el modelo como un sistema de almacenamiento revolucionario de 10MB.

2.  **Auditoría de Ingesta DNI (Nivel 3):**
    *   **Acción:** Probar el comando `gaje-cli --dni-ingest` con datos nuevos.
    *   **Meta:** Validar que el modelo puede aprender información específica sin que su PPL empeore (Recall Delta < 1%).

---

## 🧠 Vía B: Rescate de la Fidelidad Genómica (Nivel 2)
**Objetivo:** Bajar la perplejidad de 572 a < 15.0 y eliminar la "Obsesión Técnica".

1.  **Auditoría de Gradientes y Loss:**
    *   **Acción:** Revisar `src/nn/trainer.rs` para verificar si el cálculo de `cross_entropy` es compatible con el espacio toroidal.
    *   **Sospecha:** Los gradientes podrían estar colapsando o explotando debido a una mala escala en la topología circular.

2.  **Entrenamiento con Mosaic Dataset (420MB):**
    *   **Acción:** Ejecutar `micro-distiller.rs` usando el dataset balanceado (Cultura General 40%, Diálogo 30%, Técnico 30%).
    *   **Meta:** Diluir la obsesión técnica y restaurar la gramática básica del español.

3.  **Afinación de Anclas (Rigidez):**
    *   **Acción:** Experimentar con un `anchor_threshold` más dinámico. 
    *   **Meta:** Permitir que los pesos de 2 bits se muevan con más libertad durante el aprendizaje de lenguaje natural.

---

## 🛠️ Tareas de Infraestructura Inmediatas

*   **Restauración de SIMD en Android:** Investigar por qué los kernels NEON generaban NaNs. El objetivo es volver a tener velocidad máxima sin sacrificar estabilidad.
*   **Limpieza de Vaporware:** Eliminar o marcar como "Aspiracional" las carpetas y reportes que hablan de RAG o Island Model hasta que L1 y L2 estén certificados.

---
*Este plan será actualizado semanalmente según los resultados de los logs en `benchmarks/logs/`.*
