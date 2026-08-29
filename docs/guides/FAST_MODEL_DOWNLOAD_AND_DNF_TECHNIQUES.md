# ⚡ Optimización de Descargas de Modelos Pesados y Técnicas DNF

Esta guía técnica documenta las técnicas de aceleración de red utilizadas por gestores de paquetes modernos como **DNF (Fedora)** y cómo aplicar esos mismos principios para acelerar drásticamente la descarga de modelos genómicos y LLMs pesados (`.flat`, `.gaje`, `.gguf`, `.safetensors`) desde **Hugging Face Hub**, CDNs y entornos WebAssembly.

---

## 1. ¿Cómo Funciona la Aceleración en DNF (Fedora)?

**DNF (Dandified YUM)** y su nueva generación **DNF5** logran altas velocidades de transferencia utilizando por debajo la biblioteca nativa en C **`librepo`** (respaldada por `libcurl multi-handle`). Las 4 técnicas maestras que implementa son:

```
                  ┌──────────────────────────────────────────────┐
                  │          DNF / librepo Engine (C/Rust)       │
                  └───────┬──────────────┬──────────────┬────────┘
                          │              │              │
        ┌─────────────────┴─┐     ┌──────┴────────┐    ┌┴────────────────┐
        │  Multi-Stream TCP │     │     zchunk    │    │  HTTP Range 206 │
        │ (10-20 Conexiones)│     │(Delta Blocks) │    │  (Reanudación)  │
        └───────────────────┘     └───────────────┘    └─────────────────┘
```

### A. Descargas Paralelas y Concurrencia de Sockets (`max_parallel_downloads`)
* **Problema tradicional:** La mayoría de clientes HTTP descargan un solo archivo a la vez en un único stream TCP lineal.
* **Solución DNF:** Abre entre 10 y 20 conexiones HTTP simultáneas. Si hay múltiples paquetes o partes, todos se transmiten en paralelo saturando la ventana de congestión TCP (*TCP Window Scaling*).

### B. Compresión Diferencial por Bloques (`zchunk / .zck`)
* **Innovación en Fedora:** En vez de descargar metadatos completos comprimidos (ej. un archivo `repodata` de 100 MB), `zchunk` divide el archivo en bloques (*chunks*) independientes comprimidos con Zstandard y genera un índice de hashes SHA-256.
* **Resultado:** Si el 90% de los paquetes no cambiaron, DNF solo solicita mediante cabeceras `Range: bytes=start-end` los bloques que contienen diferencias, reduciendo la descarga de 100 MB a menos de 5 MB.

### C. Reanudación Automática de Transferencias (`HTTP 206 Partial Content`)
* Si la conexión se interrumpe al 80%, el cliente consulta el tamaño del archivo parcial en disco e inserta la cabecera:
  ```http
  Range: bytes=1061683200-
  ```
  El servidor responde con `206 Partial Content` y la descarga continúa exactamente en el byte donde se detuvo sin reiniciar desde 0.

### D. Espejos Dinámicos (*FastestMirror & GeoDNS*)
* `librepo` realiza pruebas de latencia y ping SSL hacia réplicas mundiales (*mirrorlists*) y conecta dinámicamente con los nodos de menor RTT (*Round Trip Time*).

---

## 2. Métodos de Aceleración para Hugging Face Hub

Por defecto, la API estándar de Python (`requests` o `urllib`) descarga en **un solo stream secuencial no optimizado**, limitando transferencias a 10–25 MB/s. Aplicando los principios de DNF se pueden alcanzar velocidades de **100 MB/s a 1+ GB/s**.

---

### Opción 1: `hf_transfer` (Acelerador Oficial en Rust de Hugging Face) ⭐ *Recomendado*

Hugging Face desarrolló **`hf_transfer`**, una extensión nativa escrita en **Rust** diseñada para saturar conexiones de alta velocidad mediante particionado concurrente masivo con `HTTP Range`.

#### Instalación:
```bash
pip install hf-transfer huggingface_hub
```

#### Uso en Terminal / CLI:
* **PowerShell (Windows):**
  ```powershell
  $env:HF_HUB_ENABLE_HF_TRANSFER = "1"
  huggingface-cli download erickaguilar/gaje-pico-135m gaje_pico_135m.flat --local-dir ./models
  ```
* **Linux / macOS:**
  ```bash
  export HF_HUB_ENABLE_HF_TRANSFER=1
  huggingface-cli download erickaguilar/gaje-pico-135m gaje_pico_135m.flat --local-dir ./models
  ```

#### Uso en Scripts de Python:
```python
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="erickaguilar/gaje-pico-135m",
    filename="gaje_pico_135m.flat",
    local_dir="./models"
)
print(f"Modelo descargado a máxima velocidad en: {model_path}")
```

---

### Opción 2: Descarga Multi-Conexión con `aria2c` (Multiparte / Concurrente)

`aria2c` es el análogo directo de `librepo` para terminal, dividiendo un archivo individual en múltiples fragmentos simultáneos con reanudación automática.

#### Instalación:
* **Windows (Winget / Scoop / Choco):**
  ```powershell
  winget install aria2.aria2
  ```
* **Fedora / RHEL:**
  ```bash
  sudo dnf install aria2
  ```
* **Ubuntu / Debian:**
  ```bash
  sudo apt install aria2
  ```

#### Descarga Optimizada con 16 Conexiones Concurrentes:
```bash
aria2c -x 16 -s 16 -k 2M -c \
  "https://huggingface.co/erickaguilar/gaje-pico-135m/resolve/main/gaje_pico_135m.flat" \
  -o gaje_pico_135m.flat
```
* **Parámetros Clave:**
  * `-x 16`: Permite hasta 16 conexiones simultáneas por servidor.
  * `-s 16`: Divide el archivo en 16 partes descargadas en paralelo.
  * `-k 2M`: Tamaño mínimo de fragmento (2 Megabytes).
  * `-c`: (*Continue*) Reanuda automáticamente descargas interrumpidas sin perder progreso.

---

### Opción 3: Git LFS con Concurrencia Acelerada

Si clonas repositorios de Hugging Face mediante `git clone` con Git Large File Storage (LFS), aumenta la concurrencia por defecto:

```bash
# Configurar 16 transferencias simultáneas en Git LFS
git config --global lfs.concurrenttransfers 16

# Clonar repositorio
git clone https://huggingface.co/erickaguilar/gaje-pico-135m
```

---

### Opción 4: En el Navegador Web / WebAssembly (Client-Side Parallel Fetching)

En entornos de navegador (como la Web UI de GAJE Helix), se utiliza la técnica de **Chunking HTTP con Web Streams API**:

```javascript
// Descarga paralela multipart en navegador mediante HTTP Range
async function downloadModelMultipart(url, totalBytes, parts = 4) {
    const chunkSize = Math.ceil(totalBytes / parts);
    const promises = [];

    for (let i = 0; i < parts; i++) {
        const start = i * chunkSize;
        const end = Math.min(start + chunkSize - 1, totalBytes - 1);

        promises.push(
            fetch(url, {
                headers: { 'Range': `bytes=${start}-${end}` }
            }).then(res => res.arrayBuffer())
        );
    }

    const buffers = await Promise.all(promises);

    // Unir chunks en un solo ArrayBuffer continuo
    const combined = new Uint8Array(totalBytes);
    let offset = 0;
    for (const buf of buffers) {
        combined.set(new Uint8Array(buf), offset);
        offset += buf.byteLength;
    }

    return combined;
}
```

---

## 3. Tabla Comparativa de Rendimiento

| Método | Tipo de Conexión | Reanudación (`Range`) | Velocidad Típica (Fibra 500M) | Complejidad |
| :--- | :---: | :---: | :---: | :---: |
| **Python `requests` estándar** | 1 Stream lineal | ❌ No por defecto | 12–25 MB/s | Muy baja |
| **Navegador `fetch()` estándar** | 1 Stream lineal | ❌ No | 15–30 MB/s | Muy baja |
| **`aria2c` (16 conexiones)** | Multi-Stream paralelo | ✅ Sí (Automático) | 60–65 MB/s *(Satura enlace)* | Media |
| **`hf_transfer` (Rust nativo)** | Multi-Stream masivo | ✅ Sí (Optimizado) | 60–65+ MB/s *(Máximo de red)* | Muy baja |
| **Git LFS con concurrencia** | Multi-archivo paralelo | ✅ Sí | 45–60 MB/s | Media |

---

## 4. Resumen Operativo para Modelos GAJE

1. **Para descargar modelos pesados en entorno local (CLI/Python):**
   Usa `hf_transfer`:
   ```powershell
   $env:HF_HUB_ENABLE_HF_TRANSFER="1"
   huggingface-cli download erickaguilar/gaje-pico-135m gaje_pico_135m.flat --local-dir ./models
   ```
2. **Para descargas directas con reanudación por terminal:**
   Usa `aria2c -x 16 -s 16 -c <URL_DIRECTA>`.
3. **Para persistencia en Web UI:**
   Una vez descargado, el modelo se aloja automáticamente en **IndexedDB (`GajeHelixDB`)** para acceso instantáneo con 0 latencia de red en las sesiones posteriores.
