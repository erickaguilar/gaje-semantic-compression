# 🧪 Plan de Viabilidad: Topología de Centroides (Fase 4.0)

Este documento describe el experimento controlado para validar si el uso de **Grafos de Relación entre Centroides** mejora la coherencia y retención del Micro-Genoma.

---

## 🏔️ Objetivo del Experimento
Demostrar que inyectar una **Topología Relacional** (extraída de un modelo maestro) en un organismo de 2 bits (Gold Embryo) reduce la perplejidad (PPL) y mejora la recuperación de información en contextos largos, incluso antes del entrenamiento masivo.

---

## 🛠️ Fases del Experimento de Viabilidad

### Paso 1: Extracción del Mapa Topológico (The Map Maker)
*   **Acción:** Crear `scripts/research/extract_centroid_topology.py`.
*   **Procedimiento:**
    1.  Pasar un corpus de 10,000 tokens por un modelo maestro (SmolLM2-F32).
    2.  Registrar las secuencias de activación de los centroides por capa.
    3.  Generar una **Matriz de Adyacencia de Centroides (CAM)** que represente las probabilidades de transición.
*   **Resultado:** Un archivo `topology_map.json` con la "huella digital" del pensamiento del maestro.

### Paso 2: Inyección y Motor Híbrido (The Bridge)
*   **Acción:** Modificar el motor de inferencia en Rust (`src/nn/block.rs`).
*   **Procedimiento:**
    1.  Implementar un "Relational Bias": Durante el forward, los logits resultantes de los pesos de 2 bits se multiplican/suman por el peso de relación del grafo.
    2.  Fórmula: `Logit_final = Logit_DNA + (Alpha * Graph_Relational_Strength)`.
*   **Meta:** El grafo debe actuar como un "corrector semántico" en tiempo real.

### Paso 3: Prueba Comparativa (The Showdown)
*   **Escenario A:** Gold Embryo (Línea Base - Solo ADN).
*   **Escenario B:** Gold Embryo + Topología de Grafo (Fase 4.0).
*   **Métricas de Éxito:**
    1.  **Reducción de PPL:** ¿Baja la PPL de 53k a < 5k solo con el grafo?
    2.  **Needle in a Haystack:** ¿Es capaz de encontrar la "aguja" en 256 tokens gracias a la guía relacional del grafo?

---

## 📈 KPIs de Viabilidad
| Métrica | Meta (Éxito) | Impacto Esperado |
| :--- | :--- | :--- |
| **Aumento de Coherencia** | > 40% | Reducción drástica de tokens repetitivos (ej. "Optical"). |
| **Fidelidad al Maestro** | > 70% Overlap | Los tokens más probables deben alinearse con el grafo del maestro. |
| **Latencia adicional** | < 5ms | El paso de grafo en Rust debe ser casi instantáneo. |

---

## 📅 Próximos Pasos Inmediatos
1.  Desarrollar el script de extracción de topología.
2.  Implementar el soporte de `CentroidGraph` en la capa de datos de Rust.
3.  Ejecutar la validación cruzada sobre `data/datasets/dataset_es.txt`.

---
*Este plan define el criterio de "Go/No-Go" para la implementación definitiva de la Fase 4.0.*
