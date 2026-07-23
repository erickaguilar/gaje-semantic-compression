# 📖 Guía de Usuario: Ecosistema GAJE-Flow (v1.0.0)

Esta guía detalla cómo interactuar con el motor de compresión semántica, las herramientas de línea de comandos y las interfaces de chat.

---

## 🛠️ 1. gaje-cli (El Motor Nativo)

El binario principal se encuentra en `target/release/gaje-cli`. Es la herramienta suiza para manipular modelos `.gaje`.

### Comandos Comunes:

*   **Inspeccionar un modelo:** Ver metadatos, arquitectura y estado interno.
    ```bash
    ./target/release/gaje-cli models/tu_modelo.gaje --inspect
    ```
*   **Chat rápido (Inferencia):** Probar la respuesta del modelo ante un prompt.
    ```bash
    ./target/release/gaje-cli models/tu_modelo.gaje --prompt "Hola, ¿quién eres?"
    ```
*   **Entrenamiento Manual:** Entrenar un modelo con un archivo de texto.
    ```bash
    ./target/release/gaje-cli models/base.gaje --train data/texto.txt --epochs 5 --save models/nuevo.gaje
    ```
*   **Inicializar modelo:** Crear un nuevo organismo desde cero (presets: `micro`, `silver_adult`).
    ```bash
    ./target/release/gaje-cli models/temp.gaje --init models/nuevo.gaje --preset micro
    ```

---

## 🐚 2. Scripts de Automatización (.sh)

Ubicados principalmente en `scripts/maintenance/`. Estos gestionan el hardware y procesos largos.

*   **`run_micro_distill_safe.sh`**: Ejecuta la destilación micro con **Wake Lock** activo (evita que Android suspenda la CPU). Recomendado para procesos de más de 10 minutos.
    ```bash
    ./scripts/maintenance/run_micro_distill_safe.sh
    ```
*   **`run_island_stabilization.sh`**: Ejecuta el entrenamiento por nichos (Island Model) y fusiona los resultados.
*   **`nightly_silver_adult.sh`**: Proceso completo de construcción nocturna para modelos Silver Adult.

---

## 💬 3. Interfaces de Chat (Examples)

Existen dos formas principales de interactuar de manera fluida con los modelos.

### A. Chat Genómico (Terminal)
Ideal para debugging y pruebas de coherencia rápidas.
*   **Ubicación:** `examples/core_demos/chat_genomico.py` o el binario `src/bin/gaje-born-chat.rs`.
*   **Ejecución (Nativa):**
    ```bash
    cargo run --release --bin gaje-born-chat
    ```
    *(Usa por defecto `silver_adult_anchored.gaje`)*

### B. Chat Visual (Interfaz Web)
Una interfaz elegante basada en navegador para una experiencia de usuario moderna.
*   **Servidor:** `examples/view/server.py`
*   **Ejecución:**
    ```bash
    python examples/view/server.py
    ```
*   **Acceso:** Abre tu navegador en `http://localhost:8080`. Requiere que el servidor tenga acceso a los modelos en la carpeta `models/`.

---

## 🏗️ 4. Compilación y Mantenimiento

Si realizas cambios en el código de Rust, debes recompilar:

```bash
cargo build --release
```

Los logs de rendimiento y entrenamiento se almacenan en:
`benchmarks/logs/`

---
*Protocolo GAJE-Flow v1.0.0 - 2026*
