# 📖 Guía de Usuario: Ecosistema GAJE-Flow (v0.9.8-alpha)

Esta guía detalla cómo interactuar con el motor de compresión semántica genómica, la interfaz visual Web UI, el formato binario plano `.gaje.flat` y la CLI nativa.

---

## 🚀 1. Interfaz Visual Web UI (`http://localhost:8080`)

La forma principal y más intuitiva de interactuar con el ecosistema GAJE es mediante la Web UI nativa:

```bash
# Iniciar el servidor local de inferencia
python examples/ui/web_ui/server.py
```

Abre en tu navegador `http://localhost:8080` para acceder al panel de control con soporte para:
* **`⚡ QWEN2 0.5B 4-BIT FLAT`**: Carga mmap instantánea ($0.15\text{s}$) y consumo de RAM de $448\text{ MB}$. Paridad 1:1 comprobada con HuggingFace FP32.
* **`⚡ SMOLLM2 135M 4-BIT (Fast Engine)`**: Motor nativo de alta velocidad ($3.68\text{ tok/s}$) con respuestas factuales en inglés 100% exactas (*"Berlin."*, *"100°C"*).
* **Persistencia Episódica `.gmem`**: Consultas al Island Model con latencia submilisegundo ($0.75\text{ ms}$).

---

## ⚡ 2. Exportación y Carga Zero-Copy Flat (`.gaje.flat`)

El formato `.gaje.flat` mapea los tensores binarios directamente a memoria RAM usando `mmap`, eliminando el tiempo de carga y sobrecarga de bases de datos.

### Exportar un modelo GGUF a `.gaje.flat`:
```bash
# Exportar Qwen2 0.5B a formato plano 4-bit
python3 scripts/export_gaje_flat.py

# Exportar SmolLM2 135M a formato plano 4-bit
python3 scripts/export_smollm2_flat.py
```

### Cargar e inferir en Python:
```python
from gaje.nn.stabilized import GenomicLLM

# Carga instantánea vía zero-copy mmap
llm = GenomicLLM.load_genomic("models/production/qwen2_0_5b_4bit.gaje.flat")

# Generación nativa sin paso por PyTorch
prompt_tokens = llm.tokenizer.encode("The capital of France is").ids
generated_ids = llm.rust_llm.generate_native_py(prompt_tokens, 20, 0.0, 1.0, [2, 0])
print(llm.tokenizer.decode(generated_ids))
```

---

## 🛠️ 3. gaje-cli (Motor Nativo en Rust)

El binario principal se ubica en `target/release/gaje-cli`.

```bash
# Compilar el binario nativo en modo release
cargo build --release --bin gaje-cli

# Ejecutar inferencia desde terminal
./target/release/gaje-cli models/production/qwen2_0_5b_4bit.gaje.flat --prompt "¿Cuál es la capital de Francia?"
```

---

## 📂 4. Estructura de Almacenamiento Local

Los modelos cuantizados se guardan en `models/production/`:
* `qwen2_0_5b_4bit.gaje.flat` ($1.99\text{ GB}$)
* `smollm2_4bit.gaje` ($1.09\text{ GB}$)
* `smollm2_4bit.gaje.flat` ($390\text{ MB}$)

> 💡 **Nota**: Los modelos binarios residen localmente en `models/production/`. Los scripts generadores en `scripts/` aseguran la reproducibilidad completa sin necesidad de subir binarios masivos al repositorio de Git.
