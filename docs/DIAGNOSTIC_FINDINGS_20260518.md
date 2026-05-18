# 🔍 Diagnóstico Técnico de Estabilidad y Rendimiento (18 de Mayo, 2026)

Este documento resume los hallazgos actuales sobre el estado del código, bugs detectados y áreas de mejora crítica tras la transición a la versión 0.7.0.

---

## 1. Discrepancias de Integridad y Versiones
- **Conflicto de Versión:**
    - `Cargo.toml`: v0.7.0 (Soberanía Nativa).
    - `pyproject.toml`: v0.6.5 (Desincronizado).
    - *Impacto:* Fallos potenciales en el despliegue de paquetes y gestión de dependencias.
- **Cambios no Confirmados:**
    - `src/compute/math.rs`: Implementación experimental de **Pre-shaping estadístico** en la generación de ADN. No ha sido validada formalmente contra métricas de perplejidad (PPL).

---

## 2. Errores Críticos de Sistema (Bugs)
- **Bloqueos de Base de Datos (Redb):**
    - Error detectado: `Database already open. Cannot acquire lock.`
    - *Causa probable:* Mal manejo de las transacciones de escritura o persistencia de punteros `Arc` sobre la base de datos en entornos multihilo.
- **Exposición de Atributos (Rust-Python):**
    - Persisten reportes de `AttributeError` en la integración con Python.
    - *Hallazgo:* Los campos que utilizan `Arc` (como `database`, `epi_strands`, etc.) requieren getters manuales consistentes para ser accesibles desde el ecosistema Python.

---

## 3. Cuellos de Botella de Rendimiento
- **Inferencia Escalar en Activaciones:**
    - El bloque SwiGLU en `src/nn/block.rs` opera en bucles escalares.
    - *Acción requerida:* Migrar a una implementación vectorizada con `Rayon` o kernels SIMD.
- **Overhead en Conversión f16 -> f32:**
    - El kernel `genomic_dot_product` realiza conversiones de anclas `f16` en tiempo de ejecución mediante bucles manuales.
    - *Acción requerida:* Optimizar la de-cuantización de anclas utilizando instrucciones intrínsecas de NEON (vcvt) para ARM.

---

## 4. Estabilidad Algorítmica (Fidelidad)
- **Deriva Semántica (Semantic Drift):**
    - Se ha identificado que en modelos profundos (24+ bloques), el error de cuantización de 2 bits se magnifica exponencialmente tras pasar por SwiGLU.
    - *Impacto:* Degradación de la coherencia en respuestas largas (ej. "Capital de México" -> Alucinación).
- **Alineación GQA:**
    - El plan de estabilización indica dudas sobre la correcta proyección de cabezas de Query vs Key/Value en el kernel de atención.

---

## 5. Archivos Huérfanos (Untracked)
- `dataset_entrenamiento.txt`
- `tokenizer.json`
- *Riesgo:* Estos archivos son vitales para la reproducibilidad de la "crianza" del modelo y deben ser gestionados o ignorados explícitamente en `.gitignore`.

---
**Próximos Pasos Recomendados:**
1. Sincronizar versiones a 0.7.0.
2. Corregir el sistema de bloqueos de `Redb`.
3. Vectorizar la activación SwiGLU en Rust.
