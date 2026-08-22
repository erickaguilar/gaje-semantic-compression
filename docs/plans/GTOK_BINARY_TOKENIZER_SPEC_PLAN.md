# 🧬 Especificación Técnica y Plan de Implementación: Tokenizador Binario Nativo GAJE (`.gtok`)

> **Versión de Formato:** GTOK v1.0.0  
> **Fecha de Publicación:** 22 de Agosto de 2026  
> **Estado:** [EN PROCESO / FASE 1 EN EJECUCIÓN]  
> **Dependencias:** 0 (Rust `std` puro + Python `struct` estándar)  

---

## 🎯 1. Resumen Ejecutivo

Actualmente, los modelos de lenguaje en GAJE requieren archivos de configuración JSON de gran tamaño (`tokenizer.json` de 10 a 15 MB) con dependencias en librerías externas para la decodificación de texto.

El formato **`.gtok` (GAJE Tokenizer Binary Format)** es una especificación binaria compacta, autocontenida y sin dependencias externas diseñada para:
1. **Reducir el tamaño en disco:** De ~14.8 MB (JSON) a **~2.4 MB (Binario .gtok)** (ahorro del 84%).
2. **Eliminar dependencias externas:** Implementable al 100% con `std::io` en Rust y el módulo estándar `struct` en Python.
3. **Carga Ultrarrápida / Mmap:** Deserialización y verificación de cabecera en **< 1.5 ms**.
4. **Incrustación en `.flat` / `.gaje`:** Permitir que los pesos y el tokenizador residan en un único archivo binario indivisible (*Single-File Model Architecture*).

---

## 📐 2. Especificación de la Estructura Binaria (`.gtok`)

```text
+-----------------------------------------------------------------------------+
|                              CABECERA GTOK (16 Bytes)                       |
|  [0..4]   Magic Bytes: "GTOK" (0x47, 0x54, 0x4F, 0x4B)                      |
|  [4..6]   Formato Versión: uint16 (1)                                       |
|  [6..8]   Flags / Arquitectura: uint16 (0x01: BPE, 0x02: Fallback, 0x04: Q) |
|  [8..12]  Vocabulario Total (V): uint32 (ej. 151643)                        |
|  [12..16] Total de Fusiones Merges (M): uint32 (ej. 151387)                 |
+-----------------------------------------------------------------------------+
|                          TOKENS ESPECIALES (24 Bytes)                       |
|  [16..20] BOS Token ID: uint32                                              |
|  [20..24] EOS Token ID: uint32                                              |
|  [24..28] UNK Token ID: uint32                                              |
|  [28..32] PAD Token ID: uint32                                              |
|  [32..34] Cantidad de Stop Tokens Adicionales (S): uint16                   |
|  [34..34+S*4] Lista de Stop Token IDs: [uint32; S]                          |
+-----------------------------------------------------------------------------+
|                    TABLA DE CADENAS (STRING TABLE POOL)                     |
|  Punteros de Desplazamiento: Array de uint32 de tamaño (V + 1)              |
|  Pool Contiguo de Bytes UTF-8: Bloque de texto plano contiguo                |
+-----------------------------------------------------------------------------+
|                       TABLA DE FUSIONES BPE ORDENADA                        |
|  Array de Tripletas [uint32 left, uint32 right, uint32 target]              |
|  Ordenado para Búsqueda Binaria O(log M) sin asignaciones dinámicas         |
+-----------------------------------------------------------------------------+
```

---

## 📅 3. Cronograma de Fases

```
                              CRONOGRAMA GTOK TOKENIZER
                                          │
        ┌───────────────────┬─────────────┴─────────────┬───────────────────┐
        ▼                   ▼                           ▼                   ▼
     FASE 1              FASE 2                      FASE 3              FASE 4
Módulo Python       Parser Nativo Rust          Incrustación en     Migración de Modelos
(`gaje.processing.  (`src/core/gtok.rs`         Cabecera `.flat`    y Validación E2E
 gtok`)             Zero-Copy Parser)           (Single-File LLM)   (100% Pass)
```

* **Fase 1: Módulo Python `gaje.processing.gtok` (Cero Dependencias):**
  * Exportador `hf_json_to_gtok(...)`.
  * Lector e Inferencia `GtokTokenizer` con algoritmo BPE nativo en Python estándar (`struct`).
* **Fase 2: Motor Nativo en Rust (`src/core/gtok.rs`):**
  * Decodificador y codificador BPE con búsqueda binaria en microsegundos.
* **Fase 3: Incrustación en la Cabecera de Modelos `.flat`:**
  * Inclusión del bloque `.gtok` en el offset inicial de los modelos binarios planos.
* **Fase 4: Certificación y Pruebas Unitarias de Regresión:**
  * Validación de paridad de tokens al 100% contra los tokenizadores oficiales de HuggingFace.

---
*Documento registrado en `docs/plans/GTOK_BINARY_TOKENIZER_SPEC_PLAN.md`.*
