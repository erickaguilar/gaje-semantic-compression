# 📖 Diccionario Canónico de Plantillas de Chat, Tokens Especiales y Resolución de Cabecera en GAJE

**Estado:** Estándar Oficial de Arquitectura  
**Fecha:** 2026-09-09  
**Autores:** Erick Aguilar & Antigravity (Google DeepMind Pair Programming)  
**Audiencia:** Desarrolladores de Núcleo, Entrenadores de Modelos y Agentes Autónomos de IA (Antigravity, Claude, etc.)

---

## 1. Filosofía de Resolución Soberana: Cero Heurísticas

En el ecosistema **GAJE Helix**, el tiempo de ejecución (WASM, CLI y Python bindings) **no debe utilizar heurísticas basadas en el nombre del archivo** (como evaluar si la cadena contiene `"qwen"` o termina en `".gaje"`).

Toda la información necesaria para dialogar con un organismo debe resolverse desde el propio binario mediante la siguiente **Jerarquía de Prioridad Canónica**:

```
                  JERARQUÍA DE RESOLUCIÓN DE PLANTILLAS
                  
  [ Prioridad 1: Declaración Explícita en Cabecera ]
  • Campo meta_len en FlatHeaderV2 -> JSON con clave "chat_template"
                     │
                     ▼ (Si no existe o es nulo)
  [ Prioridad 2: Huella Digital del Vocabulario GTOK ]
  • Inspección de tokens especiales en la tabla de vocabulario embebida
                     │
                     ▼ (Si no contiene tokens especiales reconocidos)
  [ Prioridad 3: Fallback Clásico de Diálogo / Base ]
  • Modo Diálogo Clásico ("User: ... \nAssistant: ...") o Completado Plano
```

---

## 2. Diccionario Canónico de Plantillas y Familias

A continuación se define el catálogo exhaustivo de plantillas reconocidas por el motor nativo de GAJE:

### A. `chatml` — Chat Markup Language
* **Familias de Modelos:** Qwen (1.5 / 2 / 2.5), Yi, Smaug, OpenHermes, DBRX.
* **Tokens Especiales Clave:** `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`.
* **Estructura Canónica:**
  ```text
  <|im_start|>system
  {system_prompt}<|im_end|>
  <|im_start|>user
  {user_message}<|im_end|>
  <|im_start|>assistant
  ```
* **Stop Sequences Obligatorias:** `<|im_end|>`, `<|endoftext|>`.
* **Regla de Detección en GTOK:** `token_to_id.contains_key("<|im_start|>")`.

---

### B. `llama3` — Llama 3 / 3.1 / 3.2 Instruct
* **Familias de Modelos:** Meta-Llama-3, SmolLM2-Instruct, Hermes-3.
* **Tokens Especiales Clave:** `<|begin_of_text|>`, `<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>`.
* **Estructura Canónica:**
  ```text
  <|begin_of_text|><|start_header_id|>system<|end_header_id|>

  {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

  {user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
  ```
* **Stop Sequences Obligatorias:** `<|eot_id|>`, `<|end_of_text|>`.
* **Regla de Detección en GTOK:** `token_to_id.contains_key("<|start_header_id|>")`.

---

### C. `llama2` — Llama 2 / Mistral Classic
* **Familias de Modelos:** Llama-2-Chat, Mistral-7B-Instruct-v0.1/v0.2.
* **Tokens Especiales Clave:** `<s>`, `</s>`, `[INST]`, `[/INST]`, `<<SYS>>`, `<</SYS>>`.
* **Estructura Canónica:**
  ```text
  <s>[INST] <<SYS>>
  {system_prompt}
  <</SYS>>

  {user_message} [/INST]
  ```
* **Stop Sequences Obligatorias:** `</s>`, `[/INST]`.
* **Regla de Detección en GTOK:** `token_to_id.contains_key("[INST]")`.

---

### D. `gemma` — Google Gemma / Gemma-2
* **Familias de Modelos:** Gemma-2B-it, Gemma-7B-it, Gemma-2-9B-it.
* **Tokens Especiales Clave:** `<bos>`, `<start_of_turn>`, `<end_of_turn>`, `<eos>`.
* **Estructura Canónica:**
  ```text
  <start_of_turn>user
  {system_prompt}\n\n{user_message}<end_of_turn>
  <start_of_turn>model
  ```
* **Stop Sequences Obligatorias:** `<end_of_turn>`, `<eos>`.
* **Regla de Detección en GTOK:** `token_to_id.contains_key("<start_of_turn>")`.

---

### E. `phi3` — Microsoft Phi-3 / Zephyr
* **Familias de Modelos:** Phi-3-mini-4k/128k, Zephyr-7B-beta.
* **Tokens Especiales Clave:** `<|system|>`, `<|user|>`, `<|assistant|>`, `<|end|>`.
* **Estructura Canónica:**
  ```text
  <|system|>
  {system_prompt}<|end|>
  <|user|>
  {user_message}<|end|>
  <|assistant|>
  ```
* **Stop Sequences Obligatorias:** `<|end|>`, `<|endoftext|>`.
* **Regla de Detección en GTOK:** `token_to_id.contains_key("<|user|>")`.

---

### F. `classic` — Diálogo Clásico (Born / Preentrenados Ligeros)
* **Familias de Modelos:** `max.gaje`, `max_laser_trained.gaje`, modelos con tokenizador GPT-2 puro.
* **Tokens Especiales Clave:** Sin delimitadores ChatML propietarios; utilizan saltos de línea y etiquetas ASCII.
* **Estructura Canónica:**
  ```text
  User: {user_message}
  Assistant:
  ```
* **Stop Sequences Obligatorias:** `\nUser:`, `User:`, EOS nativo (`id=2`).
* **Regla de Detección en GTOK:** Tokenizador de tipo GPT-2 o vocabulario < 32,000 sin tokens de instrucción.

---

### G. `raw` — Modo Completado Plano (Base)
* **Familias de Modelos:** Checkpoints preentrenados sin fine-tuning de instrucciones (ej. `gaje_pico_135m_base.flat`).
* **Estructura:** Pasa el texto tal cual fue introducido por el usuario sin prefijos ni sufijos.
* **Uso:** Generación de código, completado de frases o análisis de densidad de probabilidad.

---

## 3. Especificación del JSON de Cabecera (`meta_len`)

Al entrenar, exportar o transmutar un modelo hacia el formato `.flat` o `.gaje`, la cabecera `FlatHeaderV2` permite almacenar un diccionario JSON estructurado en el offset 4096.

### Esquema Estándar Recomendado:
```json
{
  "name": "qwen2.5-0.5b-instruct",
  "version": "1.7.4",
  "chat_template": "chatml",
  "bos_token": "<|im_start|>",
  "eos_token": "<|im_end|>",
  "stop_tokens": [
    "<|im_end|>",
    "<|endoftext|>"
  ],
  "stop_token_ids": [
    151645,
    151643
  ],
  "genesis_date": "2026-09-04T22:23:58Z",
  "lineage": {
    "parent_name": "Qwen/Qwen2.5-0.5B-Instruct",
    "parent_hash": "0x4c41ee7400000000"
  }
}
```

---

## 4. Guía Operativa para Desarrolladores y Agentes de IA

### A. Al Agregar un Nuevo Modelo al Catálogo (`config.js`)
1. **No inventar reglas `isXModel`:** No agregues comprobaciones de nombre en `wasm_worker.js`.
2. **Confiar en el Motor:** Permite que `wasmEngine.get_chat_template()` y `wasmEngine.format_chat_prompt()` resuelvan la plantilla automáticamente desde GTOK y la cabecera.
3. **Validación de Tokenizador:** Asegúrate de que el modelo exportado incluya el bloque GTOK mediante `gaje-cli export --embed-gtok`.

### B. Al Exportar un Modelo con `gaje-cli export`
El comando nativo en Rust soporta el flag `--chat-template`:
```bash
# Ejemplo para exportar un modelo Qwen o Llama
gaje-cli export \
  --model models/weights.gguf \
  --chat-template chatml \
  --embed-gtok \
  --output models/production/qwen2_5_0_5b.gaje
```

### C. Manejo de Delimitadores en la Salida
Al decodificar la respuesta generada:
1. El motor debe detenerse en cuanto se prediga cualquiera de los `stop_token_ids`.
2. La función de limpieza debe retirar únicamente los delimitadores terminales (`<|im_end|>`, `</s>`, `<end_of_turn>`), preservando cualquier bloque de código interno o formato Markdown generado legítimamente por el modelo.

---

## 5. Resumen de Implementación en Código

| Componente | Archivo Fuente | Responsabilidad Canónica |
| :--- | :--- | :--- |
| **Definición de Tipos** | `src/core/gtok.rs` | Enum `ChatTemplate` y detección por diccionario. |
| **Cabecera y Metadatos** | `src/io/header/flat.rs` | Almacenamiento de `chat_template` en `meta_len`. |
| **Motor In-Browser** | `src/wasm.rs` | Métodos `get_chat_template()`, `format_chat_prompt()`. |
| **Worker Web** | `static/js/wasm_worker.js` | Delegación transparente al motor Rust sin heurísticas. |
| **Auditoría Forense** | `static/js/chat/utils.js` | Registro explícito de la plantilla y stop tokens en la bitácora. |
