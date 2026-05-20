# 📑 Reporte Post-Mortem y Resumen de Hallazgos (Rescate v0.6.3)

## 🔍 Parte 1: Análisis del Crash (OOM)
La versión v0.6.3 (895bacf) fallaba en Termux por:
1. **Pre-asignación masiva:** Se intentaba convertir centroides F16 a F32 en cada capa durante la inferencia, duplicando el uso de RAM.
2. **Buffer Fijo:** Un array de tamaño 128 en el kernel de Rust causaba desbordamientos de memoria si el `block_size` no coincidía.
3. **Saturación de Anclas:** El umbral de 0.90 clonaba demasiados pesos para la capacidad de un móvil.

## 💎 Parte 2: Conocimiento Rescatado (Base Teórica)
A pesar del fallo en el código, la teoría validada es:
- **Centroides Dinámicos:** La escala $\mu \pm \sigma$ por bloque de 32 es el estándar de oro para fidelidad.
- **RoPE Split:** Obligatorio para arquitecturas modernas; el kernel debe rotar mitades separadas.
- **Inferencia en Bucle Cerrado:** Para eliminar latencia, la orquestación debe migrar a Rust (Evolución 4).
- **IQAT:** El refinamiento de centroides basado en el error de predicción funciona para estabilizar la perplejidad (logrado PPL 1.60).

---
*Estado: Sistema restaurado a f05f8b8. Base de conocimientos preservada.*
