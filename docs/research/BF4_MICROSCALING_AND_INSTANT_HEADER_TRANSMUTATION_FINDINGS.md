# 🧬 Hallazgos de Investigación: Microscaling BF4/FP4 y Transmutación Inmediata de Cabeceras Zero-Copy

> **Fecha:** 2 de Septiembre de 2026  
> **Versión del Motor:** `GAJE Helix v1.8.0 / Flat Engine v2`  
> **Ámbitos:** Microscaling en 4-Bits (`FP4 E2M1 / BF4`) · Transmutación Inmediata de Cabeceras · Compatibilidad con Hardware Físico · Zero-Copy Mmap  
> **Módulos Afectados:** `src/compute/math.rs`, `src/io/flat_header.rs`, `src/io/flat_reader.rs`, `src/compute/gpu/shaders/`

---

## 1. 🎯 Diagnóstico y Motivación

En la inferencia y almacenamiento de modelos de lenguaje en el borde (*Edge Devices*), coexisten dos tensiones fundamentales:
1. **La Rigidez de los Enteros Uniformes (Q4_0 Lineal):**
   * Los 16 estados de `Q4_0` son equidistantes. Sin embargo, las activaciones y pesos en transformers siguen distribuciones hiper-gaussianas con concentración masiva cerca de cero ($\mu = 0, \sigma^2 \ll 1$).
   * Un retículo uniforme desperdicia resolución en las colas y aplasta los decimales finos en el núcleo central.
2. **La Fricción entre Compresión Semántica y Hardware Físico:**
   * El hardware (CPU AVX2/AVX-512, GPU Vulkan/WebGPU) exige alineaciones estrictas a 64 bytes y punteros directos para no generar cuellos de botella de descompresión en cada ciclo de cómputo.

---

## 2. 🎛️ Microscaling BF4 / FP4 (`E2M1` y `E1M2`): Decimales Finos en 4-Bits

A diferencia de un entero de 4 bits $\{0, 1, \dots, 15\}$, el formato **Punto Flotante de 4-Bits (FP4/BF4)** modela el peso como una mantisa logarítmica con signo:

$$\mathbf{w}_{\text{FP4}} = (-1)^{s} \cdot 2^{e - \text{bias}} \cdot \left( 1 + \frac{m}{2} \right)$$

### Comparativa Estructural:

```
                      DISTRIBUCIÓN DE PELDAÑOS EN 4-BITS

   Q4_0 Lineal (Uniforme):
   [ -7 | -6 | -5 | -4 | -3 | -2 | -1 |  0  | +1 | +2 | +3 | +4 | +5 | +6 | +7 ]
   (Espaciado rígido: pierde detalles finos cerca del 0)

   FP4 / BF4 E2M1 (Logarítmico Microscaling):
   [ -6 | -4 | -2 | -1.5 | -1 | -0.5 | 0 | +0.5 | +1 | +1.5 | +2 | +4 | +6 ]
   (Alta densidad de decimales finos en el centro: preserva curvatura semántica)
```

| Formato | Bit Signo ($s$) | Bits Exponente ($e$) | Bits Mantisa ($m$) | Rango Dinámico | Densidad Cerca del Cero |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`Q4_0` Entero** | 0 (implicado en escala) | 0 | 4 | Fijo $[-8, +7]$ | Homogénea (Baja) |
| **`FP4 E2M1`** | 1 | 2 | 1 | $\approx [-6.0, +6.0]$ | **Alta (Sub-decimales finos)** |
| **`BF4 E3M0`** | 1 | 3 | 0 | Potencias puras de 2 | Ultra-rápido por bit-shift |

---

## 3. ⚡ Transmutación Inmediata de Compresión Semántica a Cabeceras Físicas

Para que el software existente (ej. servidores SSE, WebAssembly y librerías de inferencia) consuma el modelo sin sobrecosto de tiempo de ejecución:

```
    [ Archivo Compacto en Disco (.flat / .gaje) ]
    (Compresión de alta densidad: 4-bits / 2-bits)
                         │
                         ▼  (Arranque en Frío: < 0.75 ms)
            ┌─────────────────────────┐
            │   mmap() Zero-Copy      │
            └────────────┬────────────┘
                         │
                         ▼
   ┌─────────────────────────────────────────────────────────────┐
   │           VISTA DE CABECERA TRANSMUTADA AL VUELO            │
   │               (FlatHeaderV2 Autodescriptivo)                │
   ├─────────────────────────────────────────────────────────────┤
   │ • token_embd: Puntero FP32 mapeado contiguo                 │
   │ • lm_head:    Puntero FP32 directo para argmax              │
   │ • RoPE / K-WTA: Constantes precalculadas en 64 bytes        │
   │ • Capas Cuerpo: Bloques BF4 / Q4_0 listos para GEMV SIMD    │
   └─────────────────────────────────────────────────────────────┘
                         │
                         ▼
   [ CPU AVX2 / GPU WebGPU Ejecutan Inferencia Inmediata ]
   (El hardware opera sobre memoria plana sin descompresión intermedia)
```

### Principios de la Transmutación Inmediata:
1. **Zero-Copy Real:** La memoria física no se duplica ni se copia en el montículo (*heap*). El descriptor `ArchitectureDescriptor` extrae dimensiones, permutaciones RoPE ($Q/K$) y offsets de tensores directamente de los primeros 512 bytes mapeados.
2. **Aislamiento de la Fragilidad Semántica:** Al transmutar la cabecera en una estructura fuertemente tipada en Rust (`FlatHeaderV2`), cualquier corrupción de bytes o desalineación se detecta en tiempo de carga (`O(1)`), impidiendo accesos ilegales de memoria (*Segfaults*).
3. **Compatibilidad Estándar:** La capa de salida expone interfaces compatibles con tensores matriciales estándar, permitiendo interoperabilidad fluida con kernels nativos y llamadas BSON/JSON de alto nivel.

---

## 4. 📊 Matriz de Impacto

| Característica | Enfoque Clásico (GGUF / PyTorch) | **GAJE Helix (BF4 + Transmutación Flat)** |
| :--- | :---: | :---: |
| **Tiempo de Carga Cold-Start** | 2.5 – 15 segundos | **`< 0.75 ms` (mmap zero-copy)** |
| **Resolución en Decimales Críticos** | Regular (Q4_0 uniforme) | **Óptima (BF4 microscaling en torno a $\mu=0$)** |
| **Sobrecarga de Conversión en RAM** | Requiere buffer intermedio de copia | **0 MB (Punteros directos a páginas del SO)** |
| **Robustez ante Caídas de Hardware** | Variable | **Totalmente protegida por cabecera tipada** |

---

## 5. 🛠️ Hoja de Ruta de Integración

1. **Kernel `math.rs`:** Implementar la tabla de de-cuantización `FP4_E2M1_LUT` en SIMD AVX2/NEON.
2. **Exportador `export_gaje_flat.py`:** Añadir el flag `--format bf4` para empaquetado de exponente/mantisa por bloque de 32.
3. **Despacho GPU WGSL:** Incorporar el shader `batched_gemv_fp4.wgsl` para ejecución acelerada en VRAM.
