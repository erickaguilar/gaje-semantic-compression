# 🛠️ Plan Estratégico de Mejoras: `gaje-cli` (Núcleo CLI en Rust)

**Fecha:** 2026-08-27  
**Estado:** Propuesta Técnica / En Planificación  
**Versión objetivo:** `1.7.0-alpha`  
**Ámbito:** Soberanía nativa en Rust, usabilidad de terminal, diagnóstico de hardware, cuantización y servidor integrado.

---

## 1. Visión y Objetivos

El ejecutable `gaje-cli` es el punto de entrada administrativo y de computación nativa del framework **GAJE (Genetic Adaptive Joint Embedding / DNA Semantic Compression)**. 

El objetivo de este plan es consolidar `gaje-cli` como una herramienta CLI autónoma, robusta y ergonómica de grado industrial, eliminando la necesidad de scripts desechables en Python y centralizando la gestión de modelos, inferencia en terminal, benchmarks y despliegues.

---

## 2. Diagnóstico del Estado Actual (`src/bin/gaje-cli.rs`)

| Componente | Estado Actual | Oportunidad de Mejora |
| :--- | :--- | :--- |
| **Parser de Argumentos** | Bucle manual `while i < args.len()` con parsing de cadenas ad-hoc. | Migración a `clap` v4 derivado con validación de tipos, autocompletado y ayuda estructurada. |
| **Interacción con Modelos** | Inferencia de un solo disparo vía `--prompt`. | REPL conversacional interactivo en terminal con streaming y soporte de flechas/historial. |
| **Diagnóstico de Hardware** | Sin comando dedicado de inspección de CPU/SIMD/GPU. | Subcomando `doctor` para certificar soporte de AVX2/AVX-512/NEON y memoria mmap. |
| **Gestión de Modelos** | Comandos `download`/`pull` aislados sin inspección de metadatos. | Subcomando `models` unificado (`list`, `inspect`, `verify`, `prune`). |
| **Conversión de Pesos** | Dependiente de scripts Python para exportar `.flat`. | Conversor nativo en Rust para transformar SafeTensors y GGUF a `.flat` Q4_0 / FP32. |
| **Evaluación de Rendimiento** | Pruebas de micro-benchmarking dispersas. | Suite de benchmarking estandarizada (`bench`) con métricas de TTFT, TPS y RAM RSS. |

---

## 3. Arquitectura del Árbol de Subcomandos

```
gaje-cli
  ├── serve       # Servidor HTTP nativo + Web UI con streaming SSE (/api/chat/stream)
  ├── chat        # Consola interactiva REPL en terminal (Markdown + métricas en vivo)
  ├── models      # Gestión e inspección de modelos planos (.flat)
  │     ├── list      # Lista modelos locales con tamaño, capas y vocabulario
  │     ├── inspect   # Muestra cabecera JSON, tensores y tipos de cuantización
  │     └── verify    # Validación criptográfica SHA-256 e integridad estructural
  ├── pull        # Descarga automatizada de modelos desde CDN / Hugging Face
  ├── convert     # Conversión directa SafeTensors / GGUF -> .flat Q4_0 en Rust
  ├── bench       # Benchmark estandarizado de latencia (TTFT), throughput y memoria
  ├── doctor      # Diagnóstico del entorno: SIMD (AVX2/NEON), GPU, RAM y mmap
  ├── epoch       # Gestión de épocas de memoria genética (.gmem)
  └── compress    # Motor de compresión semántica ADN y mutación genética
```

---

## 4. Detalle de las 6 Mejoras Principales

### 4.1 Modernización del Parser con `clap` v4 (Subcomandos Tipados)
* **Objetivo:** Reemplazar el bucle manual de cadenas por estructuras fuertemente tipadas.
* **Beneficios:**
  * Ayuda `--help` profesional y coloreada con ejemplos por subcomando.
  * Generación de scripts de autocompletado de shell para Bash, Zsh, Fish y PowerShell (`gaje-cli completions <shell>`).
  * Validación automática de límites en parámetros (ej. `temperature` en `[0.0, 2.0]`, `top_p` en `(0.0, 1.0]`).

### 4.2 REPL Interactivo en Terminal (`gaje-cli chat`)
* **Objetivo:** Permitir el uso de modelos en sesiones interactivas de terminal sin navegador web.
* **Características:**
  * Integración con `rustyline` para soporte de historial de comandos, edición en línea y navegación con flechas.
  * Streaming de tokens en vivo con medidor de velocidad en tiempo real (`tokens/s`).
  * Resaltado de sintaxis básico para bloques de código en terminal.
  * Comando `/reset` para limpiar el KV-cache y `/save` para exportar la sesión a Markdown.

### 4.3 Diagnóstico de Hardware y Entorno (`gaje-cli doctor`)
* **Objetivo:** Certificar que la máquina anfitriona cuenta con los prerrequisitos de hardware para máxima velocidad.
* **Verificaciones Realizadas:**
  * **Soporte SIMD:** Detección en tiempo de ejecución de `AVX2`, `FMA`, `AVX-512` (x86_64) o `NEON` (ARM64).
  * **Aceleración GPU:** Detección de backends disponibles (DirectX 12, Vulkan, Metal, CUDA).
  * **Rendimiento de Memoria:** Medición de ancho de banda secuencial de lectura `mmap`.
  * **Integridad del Sistema:** Verificación de límites de descriptores de archivo y memoria virtual.

### 4.4 Gestor Unificado de Modelos (`gaje-cli models`)
* **Subcomandos:**
  * `gaje-cli models list`: Muestra una tabla con todos los modelos en `./models/`, indicando parámetros (ej. `135M`, `1.5B`), tipo de cuantización (`Q4_0`, `FP32`), tamaño en MB y fecha de creación.
  * `gaje-cli models inspect <archivo.flat>`: Imprime la estructura interna del archivo binario (metadatos `ModelConfig`, dimensiones de tensores, conteo de vocabulario) en milisegundos mediante lectura de cabecera zero-copy.
  * `gaje-cli models verify <archivo.flat>`: Comprueba que el archivo no esté corrupto y que el vocabulario GTOK sea consistente.

### 4.5 Conversor y Cuantizador Nativo (`gaje-cli convert`)
* **Objetivo:** Convertir pesos externos directamente al formato plano soberano `.flat`.
* **Flujo:**
  * Lee archivos `.safetensors` o directorios HuggingFace.
  * Aplica cuantización de bloque `Q4_0` (pesos a 4 bits con escalas FP32) o preserva `FP32`.
  * Empaqueta metadatos JSON y vocabulario del tokenizador en un único artefacto binario `.flat`.
  * **Cero dependencias de Python:** Todo el empaquetado ocurre en C++/Rust nativo.

### 4.6 Suite de Benchmarking Estandarizado (`gaje-cli bench`)
* **Objetivo:** Evaluación formal y reproducible del rendimiento del motor.
* **Métricas Reportadas:**
  * **Mmap Cold-Load Time:** Tiempo de inicialización en memoria en milisegundos.
  * **TTFT (Time To First Token):** Latencia de prefill del prompt.
  * **Generative TPS:** Throughput sostenido de generación de tokens por segundo.
  * **Memory Peak (RSS):** Consumo máximo de memoria RAM durante la inferencia.
  * **Formato de Salida:** Tabla en consola y exportación opcional a JSON (`--json`) o Markdown (`--markdown`).

---

## 5. Plan de Implementación por Fases

| Fase | Tarea | Archivos Clave | Complejidad |
| :---: | :--- | :--- | :---: |
| **Fase 1** | Refactor del parser de argumentos a `clap` v4 con subcomandos tipados. | `Cargo.toml`, `src/bin/gaje-cli.rs` | Media |
| **Fase 2** | Implementar `gaje-cli doctor` (diagnóstico SIMD, CPU, GPU y memoria). | `src/compute/doctor.rs`, `src/bin/gaje-cli.rs` | Baja |
| **Fase 3** | Implementar `gaje-cli models` (`list`, `inspect`, `verify`). | `src/io/models_cmd.rs`, `src/bin/gaje-cli.rs` | Baja |
| **Fase 4** | Implementar `gaje-cli chat` (REPL interactivo en terminal). | `src/nn/repl.rs`, `src/bin/gaje-cli.rs` | Media |
| **Fase 5** | Implementar `gaje-cli bench` (suite de métricas TTFT, TPS, RSS). | `src/compute/bench.rs`, `src/bin/gaje-cli.rs` | Media |
| **Fase 6** | Implementar `gaje-cli convert` (conversor nativo SafeTensors ➔ `.flat`). | `src/io/converter.rs`, `src/bin/gaje-cli.rs` | Alta |

---

## 6. Criterios de Aceptación

1. **Ergonomía Unificada:** Toda la funcionalidad de GAJE es accesible a través de un único ejecutable `gaje-cli`.
2. **Ayuda y Documentación Viva:** `gaje-cli --help` y `gaje-cli <subcomando> --help` proporcionan instrucciones claras y banderas autocontenidas.
3. **Cero Dependencias en Tiempo de Ejecución:** Todas las operaciones (chat, benchmark, diagnóstico, inspección) se ejecutan de forma 100% nativa.
