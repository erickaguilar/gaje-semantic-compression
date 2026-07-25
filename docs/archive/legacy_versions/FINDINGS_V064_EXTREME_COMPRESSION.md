# 🧬 Reporte de Hallazgos: Límites de Compresión Extrema y Desviación Matemática (v0.6.3)

**Fecha:** 12 de Mayo, 2026
**Fase del Proyecto:** Estabilización de Inferencia y Mapeo de Entropía (Fase 12/13)
**Modelo Analizado:** SmolLM2-135M (F16 a 2-bit Genómico)

## 1. Resumen Ejecutivo
Durante las pruebas de inferencia para estabilizar el Protocolo GAJE con modelos de pequeña escala (135M parámetros), se logró restaurar la latencia nativa de Rust (~2.5 t/s) y corregir la gramática generada (evitando bucles de "texto basura"). Sin embargo, se identificó un **límite semántico duro**: la compresión a 2 bits en redes tan pequeñas destruye la retención de hechos concretos, aunque se inyecte alta densidad de anclas.

## 2. Diagnóstico de Desviación Matemática

Se determinó que la corrección que introduce la mayor distorsión en el cálculo del modelo es la **Inyección de Densidad en las Capas FFN combinada con el Clamping Artificial**, por las siguientes razones técnicas:

### A. La Trampa de la No-Linealidad (SwiGLU)
Las capas de atención soportan bien el ruido de 2 bits porque sus operaciones (MatMul) son mayormente lineales. Sin embargo, las capas FFN utilizan la función de activación **SwiGLU**.
Al sumar el resultado ruidoso del ADN (2 bits) con las Anclas de corrección (Float32), el ruido matemático subyacente pasa a través de una función altamente no-lineal. Esto provoca una desviación exponencial: un error de cuantización del 1% en la entrada se magnifica catastróficamente a la salida del bloque, destruyendo las asociaciones factuales (ej. "Capital de México").

### B. "Amputación" del Espacio Vectorial (Clamping)
Para evitar que la explosión de ruido generada por SwiGLU resultara en valores `NaN` o infinitos, se implementó un *clamping* (corte) en el kernel de Rust de `[-128.0, 128.0]`.
Matemáticamente, esto "amputa" los picos de activación legítimos que la red necesita para superar el ruido estadístico. Al cortar los valores altos, aplanamos la topología del *manifold* semántico, dejando al modelo sin la "energía" necesaria para expresar conceptos específicos.

## 3. Optimizaciones Exitosas Logradas
A pesar del límite semántico en modelos pequeños, se consolidaron avances arquitectónicos críticos para el motor:

1.  **F16 Zero-Permutation:** Se descubrió que los modelos F16 GGUF modernos no requieren la doble des-permutación de RoPE. Se ajustó el cargador (`stabilized.py`) para leer los pesos directamente, eliminando las alucinaciones de tokens aleatorios.
2.  **Muestreo Estocástico Nativo:** Se reemplazó el muestreo *greedy* por `Top-P` con `Repetition Penalty` implementado en Rust puro. Esto rompió los bucles infinitos (ej. `Verde Verde Verde`) y devolvió la variedad lingüística al motor.
3.  **Aceleración SIMD NEON de Anclas:** Se optimizó el kernel `GenomicLinear` para pre-convertir centroides y procesar anclas usando intrínsecas de CPU, levantando el rendimiento de 0.60 t/s a **~2.5 t/s**.

## 4. Conclusión y Próximos Pasos (Roadmap)
El Protocolo GAJE ha demostrado su capacidad de ejecución y ahorro de RAM extremo (18x). La incapacidad de recordar hechos se debe a la baja cantidad de parámetros del modelo base usado en la prueba.

**Recomendaciones para iteraciones futuras:**
- **Escalar el Modelo Base:** Probar con arquitecturas de >1 Billón de parámetros (ej. Qwen2-1.5B), donde la redundancia de la red mitiga la destrucción de información en las capas FFN.
- **Entrenamiento de Centroides Consciente de Activación (IQAT):** En lugar de inyectar anclas manualmente, usar los gradientes del *Activation Drift* para optimizar los centroides específicamente para la función SwiGLU.
- **Atención 100% Precisa:** Mantener la estrategia actual de proteger las capas `token_embd` y `lm_head` con umbrales de `-1.0` (precisión total).
