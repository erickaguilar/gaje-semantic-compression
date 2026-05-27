# ⚙️ Plan de Implementación: Topología Genómica Circular

**Fecha:** 26 de mayo de 2026
**Objetivo:** Modificar el motor nativo en Rust para soportar la Fase de Cuantización Circular y sentar las bases para el modelo "Silver Adult" (competidor de Gemma 4).

## 1. Fase 1: Modificación de Kernels Base (`src/compute/math.rs`)

La cuantización lineal actual basada en umbrales debe ser reemplazada (o ampliada) por una **Cuantización de Fase (Phase Quantization)**.

### Tareas:
- [ ] **Estructura de Datos Compleja:** Introducir manejo básico de números complejos para las activaciones, o simular la rotación mediante el uso de pares de flotantes `(Real, Imaginario)`.
- [ ] **Nueva Función de Cuantización:** Crear `quantize_phase_core` que convierta un vector flotante en 2-bits calculando el ángulo de fase (usando `atan2(y, x)`) y mapeándolo a los 4 cuadrantes:
  - Cuadrante I ($0^\circ-90^\circ$) -> `0b00` (A)
  - Cuadrante II ($90^\circ-180^\circ$) -> `0b01` (C)
  - Cuadrante III ($180^\circ-270^\circ$) -> `0b11` (G)
  - Cuadrante IV ($270^\circ-360^\circ$) -> `0b10` (T)

## 2. Fase 2: Motor Neuromórfico (`src/nn/spiking/layer.rs`)

Las neuronas (layer) deben procesar las señales no como sumas de voltaje lineal, sino como interferencias constructivas o destructivas de ondas.

### Tareas:
- [ ] **Acumuladores Rotacionales:** Modificar `membrane_potentials` para que manejen magnitudes y fases. Si esto es muy costoso en SIMD, emular el comportamiento circular usando funciones de activación periódicas (ej. $sin(x)$).
- [ ] **SIMD Complex Ops:** Adaptar el bloque `target_arch = "aarch64"` para que la instrucción NEON calcule la suma vectorial en el plano complejo usando la aproximación circular.

## 3. Fase 3: Integración de Anclas Nucleadoras (Anchored Islands)

El concepto de Cristalización Semántica demostrado en el script empírico debe ser integrado al flujo de entrenamiento.

### Tareas:
- [ ] **Osciladores (Anclas):** En el proceso de "Destilación Profunda" (`src/nn/distiller.rs`), las Anclas de 16-bits deben configurarse como el "Eje de Frecuencia".
- [ ] **Sincronización:** Durante el `refine_step`, las neuronas de 2-bits no solo deben buscar el menor error de pérdida cruzada, sino que deben buscar **minimizar el error de fase** con su Ancla más cercana, forzando la creación de la "Isla de Estabilidad".

## 4. Fase 4: Validación y Benchmarking

### Tareas:
- [ ] **Prueba de Colapso Semántico:** Ejecutar una prueba de "Needle in a Haystack" sobre el contexto de 128k tokens. El objetivo es probar que el modelo circular no sufre del "Efecto Borde".
- [ ] **Gemma 4 Parity Check:** Entrenar un micro-modelo (150M) con el nuevo pipeline circular y compararlo en Perplejidad contra la arquitectura lineal actual.

---
*Roadmap de Ingeniería GAJE-Flow v1.2*
