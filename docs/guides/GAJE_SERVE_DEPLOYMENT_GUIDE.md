# 🚀 Guía de Despliegue en Producción de `gaje-serve` y Modelos 7B Ultra

**Fecha:** 2026-08-28
**Estado:** Guía Operativa Oficial
**Versión de destino:** `1.7.0-alpha+`
**Módulos:** Servidor Nativo Rust (`gaje-serve`), Docker, Hugging Face Spaces, VPS Cloud, Túneles Soberanos, Web UI

---

## 1. Visión General de la Topología Híbrida

El framework **GAJE (Genetic Adaptive Joint Embedding / DNA Semantic Compression)** opera bajo una topología desacoplada **Dual-Tier (Client-Side Edge + Cloud Service)**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TOPOLOGÍA HÍBRIDA GAJE                             │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
┌──────────────────────────────────────┐  HTTPS / SSE  ┌──────────────────────────────────────┐
│  TIER 1: Web UI (Vercel / PWA)       ├──────────────►│  TIER 2: gaje-serve Backend (Cloud)  │
│  • Cero Costo de Servidor            │               │  • Binario Nativo Rust (Zero-Copy)   │
│  • WASM Local: 135M Pico / 1.5B Nano │◄──────────────┤  • Modelo: gaje_ultra_7b.flat        │
│  • Selector "Modo Cloud"             │  Tokens Flow  │  • API OpenAI: /v1/chat/completions  │
└──────────────────────────────────────┘               └──────────────────────────────────────┘
```

Esta arquitectura permite que el 90% de las consultas rápidas corran **gratis y offline** en el dispositivo del usuario (135M/1.5B), mientras que las consultas complejas de razonamiento se despachan hacia **`gaje-serve`** ejecutando el modelo **7B Ultra**.

---

## 2. Requerimientos de Hardware y Dimensionamiento

| Entorno | Modelo | RAM / VRAM | CPU / GPU Mínima | Throughput Estimado |
| :--- | :--- | :--- | :--- | :--- |
| **CPU Nativo (mmap Zero-Copy)** | `gaje_ultra_7b.flat` | **4.5 GB – 5.5 GB RAM** | 4 a 8 Cores (AVX2/AVX-512) | **15 – 25 tok/s** |
| **GPU Acelerada (Vulkan / WGPU)** | `gaje_ultra_7b.flat` | **4.2 GB VRAM** | RTX 3060, T4, A10G o Apple Silicon | **45 – 75 tok/s** |
| **CPU Nativo (Recomendado VPS)** | `gaje_prime_3b.flat` | **2.6 GB RAM** | 2 a 4 Cores | **35 – 45 tok/s** |

> **Nota:** Gracias a la memoria compartida `mmap` y a la cuantización `Q4_0 + FP32`, no se requieren instancias masivas con 32 GB de RAM. Un servidor con **8 GB de RAM** corre el modelo 7B con holgura.

---

## 3. Opción A: Despliegue en Hugging Face Spaces (Docker)

Hugging Face Spaces permite alojar el backend con hardware gratuito (16 GB RAM CPU) o GPUs económicas (T4 a ~$0.60/hr).

### A. Estructura del Repositorio en Space
Crea un Space de tipo **Docker** en Hugging Face con los siguientes archivos:

```
hf-space/
├── Dockerfile
├── README.md
└── config.json
```

### B. `Dockerfile` de Producción Multi-Stage
```dockerfile
# =============================================================================
# Fase 1: Compilación de Alto Rendimiento en Rust
# =============================================================================
FROM rust:1.80-bullseye AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src/ ./src/

# Compilación optimizada con flags de CPU nativa
RUN RUSTFLAGS="-C target-cpu=native -C opt-level=3" cargo build --release --bin gaje-cli

# =============================================================================
# Fase 2: Imagen Final Ultraligera (< 50 MB)
# =============================================================================
FROM debian:bullseye-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copiar el binario compilado
COPY --from=builder /app/target/release/gaje-cli /usr/local/bin/gaje-serve

# Crear directorio de modelos
RUN mkdir -p /workspace/models

# Descargar modelo 7B Ultra automáticamente durante el arranque
ENV MODEL_NAME="gaje_ultra_7b.flat"
ENV MODEL_REPO="erickaguilar/gaje-models"
ENV PORT=7860
ENV HOST="0.0.0.0"

EXPOSE 7860

CMD ["sh", "-c", "gaje-serve serve --model models/${MODEL_NAME} --repo ${MODEL_REPO} --host ${HOST} --port ${PORT} --cors-origins '*'"]
```

### C. Metadatos del `README.md` del Space
```yaml
---
title: GAJE Ultra 7B Cloud Engine
emoji: 🧬
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---
```

---

## 4. Opción B: Despliegue en VPS Cloud (Hetzner, Fly.io, DigitalOcean, RunPod)

Para instancias dedicadas con IP fija y latencias $< 20\text{ ms}$:

### A. Preparación del Servidor (Ubuntu 22.04 / 24.04 LTS)
```bash
# 1. Actualizar paquetes y dependencias esenciales
sudo apt update && sudo apt install -y build-essential curl aria2 nginx

# 2. Instalar Rust toolchain (si se compila en el host)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# 3. Clonar repositorio y compilar
git clone https://github.com/erickaguilar/gaje-semantic-compression.git
cd gaje-semantic-compression
cargo build --release -p gaje-cli

# 4. Descargar modelo 7B Ultra con el descargador acelerado
./scripts/maintenance/download_hf_model.sh gaje_ultra_7b.flat ./models/production
```

### B. Configuración como Servicio `systemd` (`/etc/systemd/system/gaje-serve.service`)
```ini
[Unit]
Description=GAJE Semantic Compression Native Server (7B Ultra)
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/gaje-semantic-compression
ExecStart=/home/ubuntu/gaje-semantic-compression/target/release/gaje-cli serve \
    --model /home/ubuntu/gaje-semantic-compression/models/production/gaje_ultra_7b.flat \
    --host 127.0.0.1 \
    --port 8080 \
    --cors-origins "*"
Restart=always
RestartSec=3
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
```

Activar el servicio:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now gaje-serve
sudo systemctl status gaje-serve
```

### C. Proxy Inverso con Nginx y Certificado SSL (Let's Encrypt)
```nginx
server {
    server_name api.tudominio.com;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Configuración esencial para Server-Sent Events (SSE Streaming)
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
    }
}
```

---

## 5. Opción C: Servidor Local Soberano con Túnel Cloudflare (100% Gratis)

Si dispones de un equipo local con 16 GB de RAM y deseas exponerlo a la Web UI de Vercel con HTTPS seguro sin abrir puertos en el router:

```bash
# 1. Instalar cloudflared
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared.deb

# 2. Iniciar el servidor nativo en Rust
cargo run --release -p gaje-cli -- serve \
  --model models/production/gaje_ultra_7b.flat \
  --port 8080

# 3. En otra terminal, abrir túnel HTTPS efímero inmediato
cloudflared tunnel --url http://localhost:8080
```

Cloudflare generará una URL pública segura como `https://random-subdomain.trycloudflare.com`.

---

## 6. Especificación de Endpoints del Servidor

El servidor `gaje-serve` expone una API estandarizada de alto rendimiento:

### A. `POST /api/chat/stream` (SSE Streaming Nativo)
* **Headers:** `Content-Type: application/json`
* **Body:**
  ```json
  {
    "prompt": "¿Qué es el isomorfismo genómico?",
    "max_tokens": 512,
    "temperature": 0.4,
    "repetition_penalty": 1.15,
    "inject_rag": true
  }
  ```
* **Respuesta (Server-Sent Events):**
  ```text
  data: {"token": "La", "index": 0}
  data: {"token": " compresión", "index": 1}
  data: {"token": " semántica...", "index": 2}
  data: {"done": true, "total_tokens": 128, "elapsed_ms": 320.5}
  ```

### B. `POST /v1/chat/completions` (Compatibilidad OpenAI)
* Permite conectar `gaje-serve` directamente con herramientas como **Cursor IDE**, **Continue.dev**, **Open WebUI**, **LangChain** o **LlamaIndex**.

### C. `GET /api/info` y `GET /api/health`
* Diagnóstico en vivo de memoria RAM, modelo cargado, versión del motor e instrucciones SIMD activas.

---

## 7. Conexión desde la Web UI (Vercel)

Una vez que tu servidor `gaje-serve` esté en línea:

1. Abre tu Web UI desplegada en Vercel.
2. Ve al menú superior **Motor** $\rightarrow$ Selecciona **"Modo Servidor (Cloud)"**.
3. En el campo **"URL del Servidor Remoto"**, ingresa la dirección de tu Space o VPS:
   `https://api.tudominio.com` o `https://tu-usuario-gaje-ultra.hf.space`
4. ¡Listo! Todo el procesamiento se ejecutará sobre el modelo 7B Ultra con streaming en tiempo real y telemetría completa en pantalla.

---

## 8. Monitoreo y Verificación de Salud

Comprueba el estado del servicio mediante una petición rápida:

```bash
curl -s https://api.tudominio.com/api/info | jq .
```

Respuesta esperada:
```json
{
  "status": "online",
  "engine": "GAJE-Helix-Native-Server",
  "version": "1.7.0-alpha",
  "active_model": "gaje_ultra_7b.flat",
  "parameters": "7.2B",
  "quantization": "Q4_0_FP32",
  "mmap_zero_copy": true,
  "memory_rss_mb": 4512.4
}
```
