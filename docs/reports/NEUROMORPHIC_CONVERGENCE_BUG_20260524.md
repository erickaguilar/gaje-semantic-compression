# 🧪 Reporte de Error: El Invierno Genómico (Falla de Spiking)

**Fecha:** 24 de mayo de 2026
**Estatus:** Crítico - Resuelto
**Síntoma:** Precisión estancada en < 1% durante el entrenamiento Born-Genomic.

## 1. El Problema: Inhibición Total por Defecto
Al inspeccionar el constructor `GajeNeuromorphicLayer::new`, se descubrió que los pesos empaquetados se inicializaban en `vec![0; size]`.
- En el alfabeto genómico, los bits `00` representan la base **Adenina (A)**, que tiene un valor de centroide de **-1.5**.
- **Resultado:** El organismo nace con una red 100% inhibitoria. Ningún estímulo de entrada puede superar el umbral de disparo, lo que resulta en una red "muerta" que no genera spikes y, por lo tanto, no puede aprender por resonancia.

## 2. Diagnóstico de la Heurística de Refuerzo
La función `refine_step` incrementa los bits de uno en uno (`00 -> 01 -> 11`).
- Para que una neurona pase de Inhibición Fuerte (-1.5) a Excitación Fuerte (+1.5), se requieren al menos 3 épocas de refuerzo perfecto.
- Durante esas 3 épocas, el modelo no produce salidas, lo que confunde a las métricas de precisión y ralentiza la convergencia inicial.

## 3. Solución Implementada
1.  **Inicialización de Alta Entropía:** Se ha modificado el constructor para rellenar el ADN con bits aleatorios equilibrados. Esto permite que el organismo nazca con un "ruido blanco" de pensamientos, facilitando que el proceso de selección natural (evolución) encuentre patrones útiles desde la época 1.
2.  **Calibración de Umbral:** Se ajustó el umbral dinámico para ser más permisivo en los primeros estadios de la vida del organismo.

---
*Este hallazgo es fundamental para la viabilidad de la v1.0, ya que sin spikes no hay inteligencia.*
