# 🧬 Estrategia de Despliegue Híbrido y Distribución de Modelos GAJE

**Fecha de Publicación:** 23 de Agosto de 2026  
**Clasificación:** Guía Arquitectónica y Especificación de Producto  
**Módulos:** Web UI, WebAssembly (WASM Engine), Backend Servidor (Python/Rust), Distribución Soberana  
**Estado:** ✅ Aprobado y en Hoja de Ruta

---

## 1. Visión General del Ecosistema

La arquitectura de distribución de **GAJE (Genetic Adaptive Joint Embedding)** implementa un modelo **híbrido dual (Client-Side Edge + Cloud Service)** que maximiza la accesibilidad, la privacidad soberana y el rendimiento computacional sin incurrir en costos masivos de infraestructura:

```
                                  ┌───────────────────────────────┐
                                  │      GAJE Web Application     │
                                  └───────────────┬───────────────┘
                                                  │
                 ┌────────────────────────────────┴────────────────────────────────┐
                 ▼                                                                 ▼
┌──────────────────────────────────┐                             ┌──────────────────────────────────┐
│  🌐 MOTOR LOCAL (WebAssembly)    │                             │   ⚡ SERVICIO CLOUD (Servidor)   │
│  • 100% In-Browser (Zero-Server) │                             │   • Inferencia Servidor Nativo   │
│  • Privacidad y Ejecución Offline│                             │   • Ultrarrápido (SIMD / GPU)    │
├──────────────────────────────────┤                             ├──────────────────────────────────┤
│ 🟢 GAJE-Nano-1.5B (Recomendado)  │                             │ 🔵 GAJE-Prime-3B (Estándar Cloud)│
│ 🔵 GAJE-Prime-3B (Pro Desktop)   │                             │ 🟣 GAJE-Ultra-7B (Razonamiento)  │
└──────────────────────────────────┘                             └──────────────────────────────────┘
                 │                                                                 │
                 └────────────────────────────────┬────────────────────────────────┘
                                                  │
                                                  ▼
                                 ┌─────────────────────────────────┐
                                 │   ⬇️ Descarga Soberana (.flat)   │
                                 │   • GAJE-Nano-1.5B              │
                                 │   • GAJE-Prime-3B               │
                                 │   • GAJE-Ultra-7B               │
                                 └─────────────────────────────────┘
```

---

## 2. Nomenclatura Oficial de los Modelos

Para estandarizar la experiencia de usuario y la documentación técnica, los modelos basados en la arquitectura Qwen 2.5 optimizada con compresión genómica GAJE se renombran bajo la familia oficial **GAJE Organisms**:

---

### 🟢 1. GAJE-Nano-1.5B (`gaje_nano_1.5b.flat` / `gaje_nano_1.5b.gaje`)
* **Nombre Comercial:** *GAJE Nano (Ultra-Light Edge Organism)*
* **Tamaño Comprimido:** ~420 MB – 680 MB
* **Cuantización:** 2-bit / 4-bit con inhibición lateral K-WTA
* **Disponibilidad:**
  * **Modo WASM (Navegador):** ✅ Principal (Tier Gratuito / Offline)
  * **Descarga Directa:** ✅ Disponible
* **Público Objetivo & Casos de Uso:**
  * Teléfonos móviles, tablets y portátiles ligeras.
  * Consultas rápidas, resúmenes, chat local con 0% de uso de internet.
  * Cero costos de cómputo para el operador del servicio web.

---

### 🔵 2. GAJE-Prime-3B (`gaje_prime_3b.flat` / `gaje_prime_3b.gaje`)
* **Nombre Comercial:** *GAJE Prime (Balanced Workhorse Organism)*
* **Tamaño Comprimido:** ~1.2 GB – 1.6 GB
* **Cuantización:** 4-bit K-WTA
* **Disponibilidad:**
  * **Modo WASM (Navegador):** ✅ Para equipos con 8 GB+ de RAM
  * **Modo Servidor (Cloud):** ✅ Inferencia ultra-rápida a 40+ tok/s
  * **Descarga Directa:** ✅ Disponible
* **Público Objetivo & Casos de Uso:**
  * Desarrolladores, investigadores y entornos corporativos locales.
  * Excelente equilibrio entre velocidad de inferencia, coherencia textual y retención de memoria episódica `.gmem`.

---

### 🟣 3. GAJE-Ultra-7B (`gaje_ultra_7b.flat` / `gaje_ultra_7b.gaje`)
* **Nombre Comercial:** *GAJE Ultra (Advanced Research Organism)*
* **Tamaño Comprimido:** ~3.4 GB – 4.2 GB
* **Cuantización:** 4-bit / 8-bit Genómica
* **Disponibilidad:**
  * **Modo WASM (Navegador):** ❌ Desactivado (demasiado pesado para RAM de clientes estándar)
  * **Modo Servidor (Cloud):** ✅ Servicio de Alta Potencia con GPU / SIMD nativo
  * **Descarga Directa:** ✅ Disponible para correr en terminal local (`gaje-cli`)
* **Público Objetivo & Casos de Uso:**
  * Razonamiento complejo multi-paso, generación avanzada de código y análisis genómico profundo.
  * Tareas que requieren la máxima precisión semántica y perplejidad mínima.

---

## 3. Matriz de Capacidades y Especificaciones Técnicas

| Modelo | Nombre de Archivo | Tamaño Binario | RAM Recomendada | Modalidad Principal | Throughput Objetivo |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GAJE Nano** | `gaje_nano_1.5b.flat` | **~520 MB** | 2 GB – 4 GB | 🌐 WebAssembly (Client) | 25 – 45 tok/s (Cliente) |
| **GAJE Prime** | `gaje_prime_3b.flat` | **~1.4 GB** | 4 GB – 8 GB | 🌐 WASM / ⚡ Servidor | 35 – 55 tok/s (Servidor) |
| **GAJE Ultra** | `gaje_ultra_7b.flat` | **~3.8 GB** | 8 GB – 16 GB | ⚡ Servidor Cloud | 40 – 70 tok/s (Servidor) |

---

## 4. Arquitectura de Distribución y Descargas

Para asegurar que las descargas de los modelos de 1.4 GB y 3.8 GB no saturen el ancho de banda del servidor de inferencia, se establece una topología desacoplada:

1. **CDN de Almacenamiento de Pesos (Cloudflare R2 / Hugging Face Hub):**
   * Los archivos `.flat` y `.gaje` se alojan en repositorios con **cero costo de egress** (salida de datos).
   * La Web UI descarga los modelos directamente desde el CDN hacia el navegador o hacia el disco del usuario.
2. **Servidor de Inferencia (API Gateway):**
   * El VPS / Servidor dedicado se encarga exclusivamente de procesar tokens para **GAJE Prime (3B)** y **GAJE Ultra (7B)**.
   * Utiliza memoria mapeada zero-copy (`mmap`) y kernels en Rust para atender múltiples sesiones concurrentes.
3. **Almacenamiento Local Soberano (`IndexedDB`):**
   * Los pesos descargados por el navegador para el modo WASM se pueden almacenar en caché local persistente con `Cache API` / `IndexedDB`, evitando volver a descargar el modelo en visitas posteriores.

---

## 5. Experiencia de Usuario en la Web UI (UI / UX)

### A. Selector de Modo y Modelo en Barra de Herramientas:
```
[ 🌐 Motor: Local WASM (Privado) ▼ ]   [ 🧬 Modelo: GAJE-Nano 1.5B ▼ ]   [ 🟢 Activo ]
```
O al cambiar a modo servidor:
```
[ ⚡ Motor: Cloud Servidor (Rápido) ▼ ]  [ 🧬 Modelo: GAJE-Ultra 7B ▼ ]   [ 🟢 Activo ]
```

### B. Panel de Descarga Soberana (Menú •••):
* **Descargar GAJE Nano 1.5B (`.flat`)** — *520 MB (Ideal para móviles y portátiles)*
* **Descargar GAJE Prime 3B (`.flat`)** — *1.4 GB (Equilibrio de rendimiento)*
* **Descargar GAJE Ultra 7B (`.flat`)** — *3.8 GB (Máxima potencia para terminal local)*

---

## 6. Conclusión y Próximos Pasos

Esta estructura permite ofrecer un producto de clase mundial:
1. **Tier Comunitario / Privado Gratuito:** Millones de usuarios pueden usar **GAJE Nano 1.5B** en sus navegadores sin generar costos de servidor.
2. **Tier Profesional / Productivo:** Usuarios que requieran máxima capacidad acceden a **GAJE Prime 3B** y **GAJE Ultra 7B** en la nube.
3. **Ecosistema Abierto:** La comunidad puede descargar los binarios y ejecutar la tecnología en servidores privados con `gaje-cli`.
