# 🧬 Plan de Mejora: Direct Neural Ingestion (DNI) v1.0

**Fecha:** 1 de junio de 2026  
**Estatus:** Plan Propuesto  
**Objetivo:** Evolucionar el motor DNI de prototipo a componente industrial para Edge AI.

---

## 1. Fase 1: Accesibilidad y Herramientas (Tooling)
Actualmente, el motor de ingesta reside en el núcleo de Rust pero carece de una interfaz de usuario fluida.

*   **Integración CLI:** Implementar el subcomando `gaje-cli ingest` para permitir la ingesta de archivos externos de forma directa.
*   **Parámetros de Control:** Exponer `--intensity` (tasa de mutación), `--generations` (ciclos de evolución) y `--pop-size` para permitir al usuario calibrar el equilibrio entre velocidad y precisión.
*   **Retroalimentación Visual:** Integrar barras de progreso (indicatif) que muestren la evolución del *Fitness* (coherencia semántica) en tiempo real.

## 2. Fase 2: Mutación Quirúrgica y Protección de Identidad
Transición de una mutación estocástica general a una intervención precisa sobre el genoma.

*   **Blindaje de Anclas (Stability Anchors):** Asegurar que los pesos en F16 sean inmutables durante el proceso de DNI para evitar la pérdida de la estructura base del modelo.
*   **Targeting por Activación:** Implementar un sistema de detección de neuronas "silenciosas" o subutilizadas. El DNI priorizará mutar estas neuronas para minimizar el impacto en las capacidades existentes.
*   **Localización de Capas:** Refinar la heurística de selección de capas para enfocar la ingesta en bloques específicos de la FFN, preservando los heads de atención.

## 3. Fase 3: Escalabilidad y Procesamiento por Islas
Optimización para documentos de gran tamaño (>1MB) en hardware móvil (ARM).

*   **Fragmentación Semántica (Cromosomas):** Pre-procesar documentos para dividirlos en fragmentos coherentes que puedan ser ingeridos individualmente.
*   **Island Model Integration:** Utilizar la arquitectura de Islas para procesar diferentes fragmentos del documento en paralelo, aprovechando todos los núcleos de la CPU.
*   **Fusión de Mutantes:** Desarrollar algoritmos de mezcla para integrar el conocimiento de diferentes islas sin introducir ruido destructivo.

## 4. Fase 4: Métricas de Deriva Genómica (Anti-Olvido)
Garantizar que el aprendizaje de nueva información no degrade el conocimiento previo.

*   **Puntos de Control de Identidad:** Establecer un set de validación base que el modelo debe seguir prediciendo correctamente durante el proceso de ingesta.
*   **Función de Fitness Multiobjetivo:** La evolución premiará la memorización del nuevo dato pero penalizará severamente el aumento de la perplejidad en el conocimiento base.
*   **Métricas de Deriva:** Informar al usuario sobre el porcentaje de "desplazamiento genómico" resultante tras la operación de DNI.

---
*Este plan sigue el protocolo GAJE-Flow v1.0.0 y busca la soberanía total del conocimiento en el borde.*
