# 📋 Plan de Implementación: Resolución Canónica de Plantillas de Chat y Tokens Especiales desde la Cabecera y GTOK

**Estado:** Aprobado para Ejecución  
**Fecha:** 2026-09-09  
**Autores:** Erick Aguilar & Antigravity (Google DeepMind Pair Programming)  
**Objetivo:** Eliminar todas las heurísticas frágiles basadas en nombres de archivo (`isBornModel`, `isBaseModel`, etc.) en el frontend/WASM y establecer un sistema canónico de introspección de plantillas de diálogo (`chat_template`) y tokens de parada (`stop_tokens`) resueltos directamente desde la cabecera binaria (`FlatHeaderV2`) y el tokenizador (`GTOK`).

---

## 1. Diagnóstico del Problema Actual

En la versión actual de la Web UI (`wasm_worker.js`), el enrutamiento de plantillas conversacionales depende de comparaciones heurísticas sobre la cadena de texto del nombre de archivo:

```javascript
// Heurística frágil actual en wasm_worker.js (L132-L143)
const isBornModel = (typeof currentModelName === 'string') && (
    currentModelName.endsWith('.gaje') ||
    currentModelName.includes('born') ||
    currentModelName.includes('max')
);
```

### Fallos Críticos Comprobados:
1. **Falso Positivo de Formato:** El modelo `qwen2_5_0_5b.gaje` termina en `.gaje`, por lo que el worker lo clasificó como `isBornModel` y omitió la envoltura ChatML. Qwen recibió el prompt como texto plano sin delimitadores `<|im_start|>`, entrando en modo de completado web (`CODE\nCopiar\nEso es GJE`).
2. **Doble Envoltura o Conflicto de Autoridad:** Tanto JavaScript en `wasm_worker.js` como Rust en `src/wasm.rs` intentan formatear el prompt de forma independiente, lo que puede provocar concatenaciones duplicadas de `<|im_start|>user\n<|im_start|>system...`.
3. **Incompatibilidad con Modelos Futuros:** Si un usuario o desarrollador exporta un nuevo organismo o renombra un archivo (ej. `mi_modelo.bin` o `asistente_v1.flat`), el sistema colapsa o asume por defecto que es un modelo base no conversacional.

---

## 2. Arquitectura de Resolución Canónica

```
                                  FLUJO DE RESOLUCIÓN CANÓNICA
                                  
    [ Archivo Binario .gaje / .flat ]
                   │
                   ▼
     [ FlatHeaderV2: meta_len ] ───► ¿Existe "chat_template" en JSON?
                   │                                  │
          (No)     ▼                                  ▼ (Sí)
     [ GTOK Tokenizer: Vocabulary ]         [ Usar plantilla declarada ]
                   │                         (ChatML, Llama3, Gemma, etc.)
                   ├────────────────────────┬────────────────────────┐
                   ▼                        ▼                        ▼
      ¿Contiene <|im_start|>?       ¿Contiene [INST]?       ¿Contiene <start_of_turn>?
                   │                        │                        │
                   ▼                        ▼                        ▼
                ChatML                   Llama-2                   Gemma
                   │                        │                        │
                   └────────────────────────┼────────────────────────┘
                                            │
                                            ▼
                           [ Rust WASM: format_chat_prompt() ]
                                            │
                                            ▼
                     [ Prompt Formateado con Cero Ambigüedad ]
```

---

## 3. Fases de Ejecución

### Fase 1: Extensión de Metadatos en Cabecera (`src/io/header/flat.rs` y `src/io/flat_writer.rs`)
* **Objetivo:** Permitir que `gaje-cli export` y los scripts de transmutación graben formalmente la clave `chat_template` dentro del JSON de `meta_len`.
* **Esquema JSON soportado:**
  ```json
  {
    "chat_template": "chatml" | "llama2" | "llama3" | "gemma" | "phi3" | "raw",
    "bos_token": "<|im_start|>",
    "eos_token": "<|im_end|>",
    "stop_tokens": ["<|im_end|>", "<|endoftext|>"]
  }
  ```

### Fase 2: Introspección Nativa en `GtokNativeTokenizer` (`src/core/gtok.rs`)
* **Objetivo:** Si la cabecera no especifica `chat_template` explícito (modelos legados o GGUF transmutados), el tokenizador deduce automáticamente la plantilla inspeccionando la presencia de tokens de control:
  1. `<|im_start|>` $\to$ `ChatTemplate::ChatML`
  2. `<start_of_turn>` $\to$ `ChatTemplate::Gemma`
  3. `[INST]` $\to$ `ChatTemplate::Llama2`
  4. `<|user|>` $\to$ `ChatTemplate::Phi3`
  5. Ninguno $\to$ `ChatTemplate::Raw` o `ChatTemplate::Classic` (`User: ... \nAssistant: ...`)
* **Extracción dinámica de `stop_token_ids`:** Retornar el vector exacto de IDs numéricos que deben detener la generación autorregresiva.

### Fase 3: Exposición de la API en `src/wasm.rs`
* **Nuevos métodos en `GajeWasmEngine`:**
  * `pub fn get_chat_template(&self) -> String`: Retorna el identificador canónico de plantilla (`"chatml"`, `"gemma"`, `"llama2"`, `"phi3"`, `"classic"`, `"raw"`).
  * `pub fn format_chat_prompt(&self, prompt: &str, system_prompt: &str, history_json: &str) -> String`: Ensambla el prompt completo con los delimitadores canónicos correctos en Rust.
  * `pub fn get_stop_token_ids(&self) -> Vec<u32>`: Retorna los IDs numéricos exactos de parada.

### Fase 4: Desacoplamiento del Web Worker (`static/js/wasm_worker.js`)
* **Eliminación Total de Heurísticas de Nombre:**
  * Retirar `isBornModel`, `isBaseModel`, y las comparaciones de subcadenas (`qwen`, `pico`, `.gaje`, etc.).
  * El worker llama directamente a `wasmEngine.format_chat_prompt(prompt, systemPrompt, JSON.stringify(history))` o delega a `chat_with_memory` el prompt limpio.
  * Registrar en la telemetría de salida (`chat_response`) el nombre de la plantilla aplicada (`template: "chatml"`) para su auditoría visual.

### Fase 5: Actualización del Exportador de Auditoría (`static/js/chat/utils.js`)
* Reflejar en la **Sección 1** de la bitácora:
  * `Plantilla de Diálogo (Template)`: `ChatML (Detectada vía GTOK)` o `ChatML (Declarada en Cabecera)`.
  * `Tokens de Parada (Stop Tokens)`: `[151645 (<|im_end|>), 151643 (<|endoftext|>)]`.

---

## 4. Matriz de Verificación y Criterios de Aceptación

| Caso de Prueba | Modelo de Prueba | Entrada | Comportamiento Esperado | Criterio de Éxito |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen 2.5 ChatML** | `qwen2_5_0_5b.gaje` | *«¿Quién eres?»* | Detección `chatml`, envoltorio `<\|im_start\|>` | Responde en prosa conversacional, sin bloques de código `CODE\nCopiar`. |
| **Llama-Born GPT-2** | `max_laser_trained.gaje` | *«¿Quién eres?»* | Detección `classic` (`User: ... \nAssistant:`) | Genera respuesta sin colapso a un solo token `ssist`. |
| **Gemma 2B** | `gemma_*.flat` | *«¿Quién eres?»* | Detección `gemma` (`<start_of_turn>`) | Delimitadores nativos de Google aplicados automáticamente. |
| **Modelo Renombrado** | `archivo_anonimo.bin` | *«Hola»* | Introspección GTOK pura | Funciona sin importar el nombre del archivo. |

---

## 5. Cronograma de Implementación

1. **Paso 1:** Implementar `ChatTemplate` en `src/core/gtok.rs` y métodos de introspección.
2. **Paso 2:** Exponer `format_chat_prompt` y `get_chat_template` en `src/wasm.rs`.
3. **Paso 3:** Modificar `wasm_worker.js` para usar la API nativa y limpiar heurísticas.
4. **Paso 4:** Actualizar `utils.js` para registrar la plantilla en la bitácora de auditoría.
5. **Paso 5:** Compilar, ejecutar suite de tests (`cargo test`) y verificar en navegador.
