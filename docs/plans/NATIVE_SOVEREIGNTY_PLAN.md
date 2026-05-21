# 🧬 Soberanía Nativa: Plan de Independencia Total (GAJE-Native)

Este documento es el plan maestro unificado para la transición del Protocolo GAJE hacia una arquitectura 100% nativa. Define la separación definitiva entre el ecosistema de investigación (Python) y el motor de ejecución/inferencia (Rust).

---

## 🏛️ Visión Arquitectónica: El Modelo "Compilador-Runtime"

Para lograr la máxima eficiencia en dispositivos móviles y sistemas embebidos, GAJE adopta un modelo de separación de responsabilidades:

1.  **Python como "Compilador y Exportador" (Offline):** Se utiliza exclusivamente para la preparación, análisis de entropía, destilación y entrenamiento de centroides (IQAT). Su salida es un archivo binario `.gaje` optimizado.
2.  **Rust como "Motor de Inferencia y Evolución" (Online):** Un runtime puro, ligero y sin dependencias de Python, capaz de cargar modelos `.gaje` y ejecutar inferencia/evolución local.

---

## 🛠️ Pilares de la Soberanía Nativa

### 1. El Formato de Archivo `.gaje` (Universal y Autocontenido)
**Objetivo:** Eliminar la necesidad de múltiples archivos y scripts de carga complejos.
*   **Estructura:** Header Magic (`GAJE01`), Metadatos JSON (configuración), Binario del Tokenizador y Tensores Genómicos (2-bit/4-bit/6-bit).
*   **Implementación:** Usar `safetensors` o un formato binario custom mapeable a memoria (`mmap`).

### 2. Autonomía de Adaptación (Balancer y Tokenización Nativa)
**Objetivo:** Eliminar dependencias de `transformers` y scripts externos durante la ejecución.
*   **Native Balancer:** Portar el `SignalToNoiseBalancer` a Rust para que el motor gestione las máscaras de precisión dinámicamente.
*   **Native Tokenization:** Integrar el crate `tokenizers` en el núcleo de Rust para procesar texto crudo directamente.

### 3. Soberanía de Entrenamiento (Native Loss & Evolution)
**Objetivo:** Habilitar el aprendizaje continuo en el dispositivo sin el overhead de un intérprete de Python.
*   **Native train_step:** Implementar funciones de pérdida (`CrossEntropy`) y optimizadores directamente en Rust.
*   **Evolución Monte Carlo:** Refinar el motor de mutación aleatoria para que el organismo "crezca" basado en la interacción real del usuario localmente.

---

## 🗺️ Hoja de Ruta hacia v0.8.0 (Actualizada)

### ✅ Fase 1: Estandarización y Carga (Completada)
- [x] Definir y validar el formato `.gaje` autocontenido.
- [x] Implementar `src/io/loader.rs` nativo (Zero-Copy).
- [x] Integrar `tokenizers` en `gaje-cli`.

### ✅ Fase 2: Ejecución Independiente (Completada)
- [x] Migrar el bucle autoregresivo completo a `src/bin/gaje-cli.rs`.
- [x] Eliminar la dependencia de `PyO3` para la inferencia base.
- [x] Lograr la ejecución de un modelo `.gaje` solo con `./target/release/gaje-cli model.gaje`.

### 🚀 Fase 3: Aprendizaje Nativo (En Curso)
- [ ] Implementar el ciclo de entrenamiento `forward -> loss -> refine` en Rust.
- [x] Habilitar el "Island Model" para evolución paralela masiva usando `Rayon` (Implementado en v0.9.0-alpha).
- [x] Implementar arquitectura industrial SoA + Timing Wheel para escalabilidad.
---

## 📊 Beneficios Esperados
1.  **Latencia Zero-Overhead:** Eliminación del GIL de Python y las conversiones de tipos.
2.  **Huella de Memoria Mínima:** < 50MB de RAM para modelos de 135M.
3.  **Distribución Trivial:** Un único binario estático y un archivo de modelo para cualquier plataforma (Android, Linux, iOS).

---
*Este documento reemplaza y unifica los planes previos de desacoplamiento y separación.*
