# ⚡ Guía de Descarga Acelerada de Modelos desde Hugging Face

**Fecha:** 26 de Agosto de 2026
**Módulos:** GAJE Core, CLI, Web UI, Hugging Face Hub, Edge Deployments
**Estado:** ✅ Aprobado y Certificado

---

## 1. Visión General y Cuello de Botella Tradicional

Los modelos genómicos `.flat` y `.gaje` varían entre **400 MB y 3.8 GB**. Las descargas estándar de Python (`requests` o `urllib`) o navegadores sin optimización descargan en un solo hilo HTTP TCP, limitando la velocidad a 5–15 MB/s independientemente del ancho de banda disponible.

Esta guía documenta los métodos oficiales para alcanzar la **velocidad de línea máxima (100–500 MB/s)** mediante conexiones concurrentes segmentadas hacia el CDN de Hugging Face.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Hugging Face Cloud CDN                             │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
   Chunk 0-25% (Hilo 1)       Chunk 25-50% (Hilo 2)     Chunk 50-100% (Hilos 3..16)
            │                          │                          │
            └──────────────────────────┼──────────────────────────┘
                                       ▼
                     Reensamblaje Zero-Copy en Disco Local
                     📂 `models/production/gaje_pico_135m.flat`
```

---

## 2. Método 1: `hf_transfer` (Motor Nativo en Rust — Recomendado)

`hf_transfer` es una librería oficial de Hugging Face escrita en **Rust** diseñada específicamente para saturar enlaces gigabit mediante peticiones paralelas concurrentes sin overhead de Python.

### A. Instalación
```bash
pip install hf-transfer huggingface_hub
```

### B. Uso en Terminal (CLI)
Activa la variable de entorno `HF_HUB_ENABLE_HF_TRANSFER=1`:

```bash
# 1. Habilitar acelerador en Rust
export HF_HUB_ENABLE_HF_TRANSFER=1

# 2. Descargar modelo plano individual
huggingface-cli download \
  erickaguilar/gaje-models \
  gaje_pico_135m.flat \
  --local-dir ./models/production \
  --local-dir-use-symlinks False
```

### C. Uso en Scripts de Python
```python
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="erickaguilar/gaje-models",
    filename="gaje_pico_135m.flat",
    local_dir="./models/production"
)
print(f"Modelo descargado en: {model_path}")
```

---

## 3. Método 2: `aria2c` (Descargas Segmentadas Multi-Hilo 16x)

`aria2c` es el descargador multi-protocolo más rápido en entornos Linux/macOS, capaz de abrir **16 conexiones paralelas simultáneas** con soporte nativo para reanudación de descargas interrumpidas (`-c`).

### A. Instalación de `aria2`
```bash
# Ubuntu / Debian / Pop!_OS
sudo apt-get install -y aria2

# Arch Linux / Manjaro
sudo pacman -S aria2

# macOS (Homebrew)
brew install aria2
```

### B. Comando de Descarga Ultrarrápida
```bash
aria2c -x 16 -s 16 -k 1M -c \
  --dir=./models/production \
  --out=gaje_pico_135m.flat \
  "https://huggingface.co/erickaguilar/gaje-models/resolve/main/gaje_pico_135m.flat"
```

**Parámetros clave:**
* `-x 16`: Número máximo de conexiones por servidor (16 hilos concurrentes).
* `-s 16`: Divide el archivo en 16 segmentos paralelos.
* `-k 1M`: Tamaño mínimo del chunk de 1 MB.
* `-c`: Reanuda automáticamente descargas parciales si se interrumpe la red.

---

## 4. Método 3: Script Automatizado Oficial (`scripts/download_hf_model.sh`)

El repositorio incluye un script automatizado que detecta automáticamente el mejor acelerador disponible (`hf_transfer` > `aria2c` > `curl` acelerado):

```bash
# Dar permisos de ejecución
chmod +x scripts/download_hf_model.sh

# Modo Interactivo con Menú de Modelos
./scripts/download_hf_model.sh

# O descarga directa por nombre de archivo
./scripts/download_hf_model.sh gaje_pico_135m.flat ./models/production
```

---

## 5. Verificación de Integridad SHA-256

Una vez completada la descarga, verifica la integridad matemática del binario antes de pasarlo al runtime nativo de GAJE:

```bash
sha256sum ./models/production/gaje_pico_135m.flat
```

---

## 6. Carga y Ejecución Inmediata en GAJE Native Runtime

Con el modelo descargado en la carpeta `models/production/`, ejecútalo instantáneamente con el CLI nativo en Rust:

```bash
# Inferencia interactiva directa en terminal (zero-copy mmap)
cargo run --release -p gaje-cli -- chat \
  --model models/production/gaje_pico_135m.flat \
  --prompt "¿Qué es la compresión semántica genómica?"
```
