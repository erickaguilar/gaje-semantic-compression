# Ruta de Integración Global: GAJE Protocol

## 🎯 Objetivo General
Este documento define la secuencia estricta de ejecución para fusionar el "Plan de Separación Rust/Python" y el "Plan de Base de Datos Genómica". El propósito de esta ruta es evitar conflictos de fusión, prevenir la creación de código "desechable" (dead code) y garantizar una transición suave desde una librería acoplada por PyO3 hacia un motor de base de datos e inferencia nativo e independiente.

---

## 🛤️ Fase 1: Cimientos K-V (`redb`) y Archivo `.gaje` Unificado
**Objetivo:** Reemplazar la exportación de múltiples archivos dispersos (`.npy`, `.json`) por una única base de datos embebida (Key-Value) gestionada por Rust, manteniendo por ahora a Python como orquestador de inferencia.

*   **Opciones de Cumplimiento (Tareas):**
    *   Integrar la librería `redb` (u otra K-V nativa) en `Cargo.toml`.
    *   Crear una API en Rust expuesta vía PyO3 (ej. `save_to_db`, `load_from_db`) para serializar las hebras SoA del modelo.
    *   Refactorizar los métodos `save()` y `load_genomic()` en `python/gaje/nn/stabilized.py` para usar esta nueva API nativa.
*   **Pruebas Satisfactorias (Acceptance Tests):**
    *   **Test Unitario:** `cargo test` pasa exitosamente ejecutando lecturas y escrituras binarias directamente en una base temporal de `redb`.
    *   **Test de Consistencia:** El script de exportación/guardado genera un **único archivo** monolítico `modelo.gaje` (en lugar de crear un directorio lleno de archivos numpy).
    *   **Test de Reconstrucción:** Cargar el modelo usando `GenomicLLM.load_genomic("modelo.gaje")` en Python restaura los pesos de la inferencia correctamente sin degradación de la perplejidad.

---

## 🛤️ Fase 2: Cargador (Loader) Nativo en Rust
**Objetivo:** Permitir que el código nativo de Rust lea el archivo `.gaje` (la base de datos) de forma totalmente autónoma y *Zero-Copy*, sin depender de los tensores de PyTorch o los puentes de PyO3.

*   **Opciones de Cumplimiento (Tareas):**
    *   Implementar `src/loader.rs` utilizando las APIs de lectura de `redb` para extraer el arreglo "Struct-of-Arrays" (hebras base, epi y tri) directo a las estructuras de memoria de Rust.
    *   Crear estructuras nativas puras en Rust (ej. `ModelConfig`) para des-serializar (parsear) los metadatos JSON del archivo.
*   **Pruebas Satisfactorias (Acceptance Tests):**
    *   **Test de Integración Rust:** Se ejecuta un `cargo test` que carga el archivo `.gaje` creado en la Fase 1, extrae el tensor correspondiente a la primera capa genómica y verifica su "checksum" (hash) o su similitud coseno, demostrando que la carga fue perfecta y 100% nativa.

---

## 🛤️ Fase 3: Corte del Cordón Umbilical (CLI Independiente)
**Objetivo:** Extraer el bucle de generación (Inferencia) y la tokenización del entorno de Python. El motor genómico de Rust se vuelve completamente autosuficiente.

*   **Opciones de Cumplimiento (Tareas):**
    *   Añadir el crate `tokenizers` de HuggingFace a las dependencias.
    *   Incrustar el archivo de vocabulario `tokenizer.json` dentro del archivo de la base de datos `.gaje` (ampliando la Fase 1).
    *   Escribir `src/bin/gaje-cli.rs`, un ejecutable (CLI) que acepte argumentos (`--prompt`, `--model`) e implemente internamente el bucle autoregresivo (`prefill` -> `decode`).
*   **Pruebas Satisfactorias (Acceptance Tests):**
    *   **Test Funcional E2E:** Ejecutar el comando `cargo run --bin gaje-cli -- --model modelo.gaje --prompt "Hola"` y recibir una respuesta en texto coherente por la consola.
    *   **Test de Rendimiento:** Demostrar que el binario CLI genera una latencia por token (TPS) igual o superior a la versión de Python al haber eliminado la sobrecarga del GIL.

---

## 🛤️ Fase 4: Habilidades Evolutivas y "Time-Travel"
**Objetivo:** Explotar la naturaleza de la base de datos K-V para llevar un registro continuo del metabolismo del modelo (Log Epigenético) y soportar aprendizaje en dispositivo de forma segura.

*   **Opciones de Cumplimiento (Tareas):**
    *   Diseñar y crear la tabla `mutations_log` dentro del esquema de la base de datos `.gaje`.
    *   Modificar las funciones de refinamiento nativo (como `refine_ffn`) para que registren el *delta* de los cambios en los centroides junto a un timestamp.
    *   Desarrollar una API de restauración en el CLI (ej. `gaje-cli rollback --date YYYY-MM-DD`).
*   **Pruebas Satisfactorias (Acceptance Tests):**
    *   **Prueba de Resiliencia (Degradación):** Ejecutar un ciclo de entrenamiento forzado y ruidoso (`refine_ffn`) que degrade intencionalmente la calidad de las respuestas del modelo.
    *   **Prueba de Restauración Temporal:** Ejecutar el comando de `rollback` a un punto de control previo. Verificar que el comportamiento estable y la perplejidad original hayan sido restaurados de forma determinista.
