# Plan de Separación Arquitectónica: Rust vs Python

## 🎯 Objetivo
Desacoplar completamente el motor de inferencia (Rust) del ecosistema de entrenamiento y preparación de datos (Python). El objetivo es tener un binario de Rust puro (`gaje-server` o `gaje-cli`) capaz de cargar modelos `.gaje` y ejecutar inferencia sin depender de `PyO3`, el GIL de Python, o PyTorch en tiempo de ejecución.

## 🏛️ Arquitectura Propuesta

### 1. Python como "Compilador y Exportador" (Offline)
El rol de Python se limitará estrictamente a la fase de preparación del modelo.
*   **Responsabilidades:**
    *   Descarga de modelos (HuggingFace/GGUF).
    *   Análisis de entropía y cálculo del `SignalToNoiseBalancer`.
    *   Cuantización base (2-bit) y extracción de Anchors (4-bit/6-bit).
    *   Entrenamiento de centroides (IQAT).
    *   Empaquetado final de tensores, centroides, y metadatos (tokenizador, hiperparámetros) en un único archivo binario `.gaje` (basado en `safetensors` o formato binario custom).
*   **Ejecución:** Scripts como `python gaje_export.py --model Qwen/Qwen2-0.5B --out qwen.gaje`.

### 2. Rust como "Motor de Inferencia" (Online)
Rust asume el control total durante la ejecución.
*   **Responsabilidades:**
    *   Carga ultra-rápida del archivo `.gaje` usando mmap (memoria mapeada).
    *   Tokenización nativa utilizando el crate `tokenizers` de HuggingFace.
    *   Ejecución del bucle autoregresivo (`forward` pass) utilizando SIMD/NEON puro para las operaciones sobre 2-bits.
    *   Manejo de la caché KV y muestreo (Top-P, Top-K, Repetition Penalty).
    *   Exposición de una API HTTP compatible con OpenAI usando `axum` o interfaz de terminal (CLI).

## 🗺️ Fases de Implementación

### Fase 1: Estandarización del Formato `.gaje`
*   **Acción:** Definir la estructura exacta del archivo `.gaje`. Debe contener:
    1.  Header Magic (e.g., `GAJE01`).
    2.  Metadatos JSON (configs del modelo).
    3.  Binario del tokenizador (vocabulario).
    4.  Cuerpo de tensores: `base_strands` (2-bit), `epi_strands` (4-bit), centroides, y máscaras de ruteo.
*   **Validación:** Modificar el código de Python en `gaje.core.archive` para que pueda generar este archivo autocontenido sin depender del runtime de PyTorch.

### Fase 2: Módulo Loader en Rust puro
*   **Acción:** Implementar `src/loader.rs` sin dependencias de `pyo3`.
*   **Herramientas:** Usar `memmap2` para mapear el archivo a memoria sin latencia de copia. Leer el header y mapear los punteros a los tensores pre-cuantizados.

### Fase 3: Integración de Tokenizador Nativo
*   **Acción:** Añadir la dependencia `tokenizers` en `Cargo.toml`.
*   **Implementación:** Instanciar el tokenizador directamente en Rust leyendo los bytes incrustados en el archivo `.gaje`.

### Fase 4: Bucle de Inferencia Independiente
*   **Acción:** Migrar la lógica autoregresiva de `RustGenomicLLM` a un ejecutable independiente `src/bin/gaje-cli.rs`.
*   **Implementación:** Escribir el bucle de generación (prompt -> prefill -> decode -> yield) usando tensores nativos de Rust (pueden ser estructuras custom o usando un crate liviano como `candle-core` enfocado solo a orquestación de tensores).

## 📊 Beneficios Esperados
1.  **Reducción de Latencia:** Eliminación del overhead del GIL y las conversiones PyO3 entre Python y Rust.
2.  **Distribución Trivial:** Se podrá compilar un único binario estático `gaje-cli` (por ejemplo, para Android via Termux o Linux estático) que los usuarios solo tendrán que descargar junto con el archivo `.gaje`.
3.  **Aislamiento de Errores:** Errores de memoria (OOM) o de compatibilidad de PyTorch ya no afectarán el entorno de producción.
