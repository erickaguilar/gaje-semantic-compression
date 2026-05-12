# 🗺️ Plan de Estabilización y Optimización: GAJE v0.6.1+

Este documento detalla la hoja de ruta paso a paso para restaurar la estabilidad, coherencia y rendimiento del Protocolo GAJE tras los problemas detectados en la integración de la Fase 12.

---

## 🎯 Objetivo General
Transformar el motor actual de un estado de "investigación inestable" a un "entorno de producción local" funcional en Termux/Android.

---

## 🛠️ Fase 1: Estabilización de la Interfaz (Prioridad: Crítica)
**Problema actual:** `TypeError` con NumPy y `AttributeError` por campos no expuestos en Rust.

- [ ] **Exposición de Campos:** Asegurar que `GenomicLinear` y `GenomicAttention` tengan el atributo `#[pyo3(get)]` en todos los campos que Python necesita leer (`database`, `centroids`, `k_cache`, etc.).
- [ ] **Compatibilidad de Tipos:** Revertir temporalmente las firmas de las funciones a `Vec<f32>` en lugar de `PyArray` para garantizar que el puente Python-Rust funcione en todas las versiones de Termux sin fallos de conversión.
- [ ] **Sincronización de Capas:** Ajustar `stabilized.py` para que coincida exactamente con las nuevas firmas de Rust, eliminando el uso de `.tolist()` donde sea posible sin romper la compatibilidad.

## 🧠 Fase 2: Corrección Algorítmica (Prioridad: Alta)
**Problema actual:** Generación repetitiva o incoherente (ruido semántico).

- [ ] **Alineación GQA:** Corregir el kernel de atención para que respete las dimensiones de Grouped-Query Attention (proyectar correctamente cabezas de Q vs K/V).
- [ ] **Precisión de RoPE:** Validar que el escalado de frecuencias en los Embeddings de Posición Rotatoria (RoPE) sea idéntico al estándar de Llama/SmolLM2.
- [ ] **Normalización Post-Atención:** Asegurar que la `rms_norm` y el escalado de los scores de atención (`1/sqrt(head_dim)`) se apliquen en el orden correcto.

## 🚀 Fase 3: Optimización de Rendimiento (Prioridad: Media)
**Problema actual:** Latencia de >200 segundos por respuesta.

- [ ] **Activación de SIMD NEON:** Implementar el "Genomic Dot Product" usando instrucciones intrínsecas de ARM para procesar 16 valores de ADN por ciclo de reloj.
- [ ] **Paralelismo con Rayon:** Habilitar el procesamiento multi-hilo en las proyecciones lineales y SwiGLU para usar todos los núcleos del dispositivo.
- [ ] **Compilación Release:** Configurar `Cargo.toml` con `opt-level = 3` y LTO para maximizar la velocidad de ejecución nativa.

## ✅ Fase 4: Validación y Cierre
- [ ] **Benchmarking de Precisión:** Correr `test_v060_validation.py` para asegurar que el aprendizaje local sigue convergiendo.
- [ ] **Chat Demo Final:** Verificar que `chat_genomico.py` responde preguntas simples (ej. "¿Cuál es la capital de Francia?") en menos de 2 segundos con coherencia total.

---
*Nota: Este plan debe ejecutarse de forma granular, verificando el éxito de cada punto antes de pasar al siguiente.*
