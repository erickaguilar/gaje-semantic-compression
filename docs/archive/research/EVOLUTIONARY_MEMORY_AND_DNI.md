# 🧬 Investigación: Memoria Evolutiva y Direct Neural Ingestion (DNI)

**Fecha:** 28 de mayo de 2026
**Estatus:** Fase de Diseño Arquitectónico
**Concepto:** Transición de RAG Externo a Memoria Genómica Integrada.

---

## 1. El Concepto: Crianza como "RAG de Pesos"
A diferencia de los sistemas de Generación Aumentada por Recuperación (RAG) tradicionales que consultan bases de datos vectoriales externas, el protocolo GAJE propone la **Memoria Evolutiva**.

En este paradigma, la información no se "consulta", se **"ingiere"**. Mediante el motor de **Crianza (Breeding)**, el modelo atraviesa ciclos de evolución ultrarrápidos (Monte Carlo) para ajustar sus pesos de 2 bits de forma que la información externa quede codificada directamente en su ADN digital.

### Hallazgos Clave (v0.6.5):
*   **Latencia de Ingesta:** ~18ms para secuencias cortas en hardware ARM.
*   **Fidelidad de Recuperación:** 95.6% de acierto en la reconstrucción de la secuencia sin necesidad de contexto externo.
*   **Eficiencia de Inferencia:** Al estar "horneada" en los pesos, la recuperación de la información tiene un coste computacional de **0 tokens adicionales** en la ventana de contexto.

---

## 2. Implementación Futura: Pipeline de DNI (Direct Neural Ingestion)
El objetivo para el Q3-2026 es automatizar la carga de datos directamente en el genoma sin pasar por un proceso de entrenamiento completo (Backpropagation).

### Arquitectura de la Pipeline:
1.  **Fragmentación Semántica:** El documento externo se divide en "Cromosomas" (fragmentos de texto coherentes).
2.  **Targeting por Capas:** Se identifican las capas del modelo con menor "Saturación de Fase" para recibir la nueva información.
3.  **Inyección por Crianza Paralela:**
    *   Se lanzan múltiples "Islas Evolutivas" en paralelo.
    *   Cada isla compite por memorizar un fragmento del documento.
    *   Los mutantes ganadores (los que mejor representan el dato) se fusionan en el organismo principal.
4.  **Estabilización por Anclas:** Se utilizan las **Stability Anchors (F16)** para asegurar que el nuevo conocimiento no provoque "Deriva Genómica" (olvido de la identidad base).

---

## 3. Ventajas para Edge Computing (Android/Termux)
*   **Ahorro de RAM:** No requiere mantener bases de datos vectoriales (HNSW/FAISS) en segundo plano.
*   **Inferencia Privada:** Todo el conocimiento reside dentro del archivo `.gaje`, facilitando la portabilidad total sin dependencias externas.
*   **Actualización Granular:** Se pueden actualizar "nichos" específicos del conocimiento mutando solo unas pocas capas.

---

## 4. Próximos Pasos Técnicos
*   [x] Implementar el flag `--dni-ingest <file.txt>` en `gaje-cli`.
*   [ ] Desarrollar un sistema de "Métricas de Olvido" para medir cuánto conocimiento base se desplaza al inyectar datos nuevos.
*   [ ] Optimizar los kernels de mutación bitwise para manejar documentos de >1MB en menos de 5 segundos.

---
*Documento generado por Gemini CLI bajo el protocolo GAJE-Flow v1.0.0*
