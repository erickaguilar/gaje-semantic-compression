# ⚓ Estrategia de Protección de Anclas: Evolución 3.5

**Estado:** Propuesta de Mejora (Post-v0.6.3 Analysis)  
**Contexto:** Basado en los hallazgos del commit `895bacf`.

## 1. El Problema: Inestabilidad vs. Memoria
En la versión v0.6.3 se intentó estabilizar la señal de modelos profundos (30 capas) mediante una "Protección de Anclas" basada en el percentil 10%. 

### ¿Por qué falló la implementación original?
La implementación en v0.6.3 causó **OOM (Out of Memory)** porque:
1.  **Pre-asignación masiva:** Se convertían todas las anclas de `f16` a `f32` en cada llamada al `forward`.
2.  **Saturación de RAM:** Un modelo de 0.5B con un 10% de anclas en `f32` consume ~200MB adicionales de memoria volátil solo en esta estructura.

## 2. La Mejora Real: "Fuerza de Conceptos"
A pesar del fallo técnico, la teoría es correcta: las anclas representan las "neuronas atípicas" (outliers) que contienen la mayor parte de la información semántica. Sin ellas, el modelo de 2 bits pierde coherencia rápidamente.

## 3. Propuesta de Evolución 3.5 (Implementación Segura)

Para integrar la protección del 10% sin romper la estabilidad en Termux, debemos seguir estas reglas:

### A. Almacenamiento Nativo `f16`
Las anclas NUNCA deben convertirse a `f32` de forma masiva. Deben residir en el `GenomicLinear` como `Vec<half::f16>`.

### B. Descompresión SIMD al Vuelo
En lugar de pre-convertir, el kernel de Rust debe:
1.  Calcular el producto punto genómico (2-bit).
2.  Acceder a las anclas en el mismo bucle.
3.  Promover `f16 -> f32` solo para los registros NEON actuales.

### C. Umbral Dinámico (Top-K vs Percentil)
En lugar de un percentil fijo que podría ser impredecible, se propone:
- **Genomización:** Identificar el Top-K de pesos con mayor error de cuantización.
- **Cuantización:** Guardar solo esos pesos como anclas `f16`.

## 4. Próximos Pasos Técnicos
1.  Refactorizar `src/nn.rs` para eliminar cualquier `collect::<Vec<f32>>()` dentro del `forward`.
2.  Ajustar `GenomicLayer._init_from_f32` en Python para enviar solo el 10% de las anclas más significativas.
3.  Validar la perplejidad (PPL) para asegurar que el 10% es suficiente para restaurar la coherencia en 30 capas.

---
*Este documento sirve como guía para la transición hacia la Evolución 4 (Rust Core).*
