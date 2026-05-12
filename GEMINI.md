# 🧬 GAJE-Flow: Protocolo de Desarrollo y Estabilidad

Este archivo define las reglas de flujo de trabajo para el proyecto **DNA Semantic Compression**. El objetivo es garantizar que la rama `develop` se mantenga siempre estable y libre de regresiones de memoria (OOM).

## 1. Arquitectura de Ramas (Branching)

- **`main`**: Código de producción. Solo recibe merges de `develop` tras validación completa de perplejidad (PPL).
- **`develop`**: Rama de integración **SAGRADA**. Todo código aquí DEBE compilar y pasar `pytest`.
- **`feature/*`**: Nuevas funcionalidades. Se crean desde `develop`.
- **`fix/*`**: Correcciones de bugs. Se crean desde la rama afectada.
- **`research/*`**: Experimentos especulativos. Pueden ser descartados sin llegar nunca a `develop`.

## 2. Reglas de Oro para el Agente (Gemini CLI)

1.  **Aislamiento:** Ante cualquier instrucción de cambio significativo, propón crear una rama `feature/` o `fix/` antes de tocar `develop`.
2.  **Validación Pre-Merge:** Antes de fusionar cualquier rama a `develop`, se debe ejecutar:
    - `cargo build --release` (Verificación de compilación nativa).
    - `pytest tests/test_integration_v060.py` (Verificación de estabilidad funcional).
3.  **Mandato de Estabilidad de Memoria:**
    - PROHIBIDO realizar pre-asignaciones masivas de tensores `f32` en el bucle de inferencia (`forward`).
    - Las conversiones `f16 -> f32` deben ser "on-the-fly" o en buffers pequeños y controlados.
    - Se debe priorizar el uso de punteros y memoria compartida sobre la copia de vectores (`collect()`).

## 3. Procedimiento ante Errores Críticos (OOM/Crash)

Si se detecta un error de estabilidad en una rama de desarrollo:
1.  **NO** intentar parches rápidos sobre `develop`.
2.  **Backtrack:** Identificar el commit estable anterior.
3.  **Aislamiento:** Mover el experimento fallido a una rama `research/` para análisis post-mortem.
4.  **Limpieza:** Utilizar `git push --force` solo como último recurso para sanear `develop` tras una divergencia crítica, como se hizo en la v0.6.3.

## 4. Estilo de Commits

Seguir el estándar de **Conventional Commits**:
- `feat(scope):` para nuevas funciones.
- `fix(scope):` para correcciones.
- `perf(scope):` para mejoras de rendimiento.
- `docs(scope):` para cambios en documentación.

---
*Este protocolo es vinculante para todas las sesiones de desarrollo asistido.*
