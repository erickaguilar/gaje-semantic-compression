# 🚀 Plan de Implementación: Backend de Aceleración GPU para GAJE Helix (Vulkan / WGPU)

> **Estado:** 🟡 EN PROGRESO — Fase 1 (Infraestructura WGPU/Vulkan Context), Fase 2 (Shaders WGSL GEMV FP32, SwiGLU, RMSNorm) y Despachadores de Pipeline en Rust completados.
> **Objetivo:** Desarrollar e integrar un backend de cómputo paralelo masivo para GPU (AMD Radeon / Vulkan / WGPU) en Rust, permitiendo offload de operaciones matriciales críticas (`GenomicLinear`, SwiGLU y `lm_head`), multiplicando el rendimiento de inferencia en modelos de 0.5B a 3B+.

---

## 🏛️ 1. Arquitectura del Backend de Aceleración

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          GAJE HELIX COMPUTE ABSTRACTION                                │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│                         ┌─────────────────────────────┐                                │
│                         │   Trait: ComputeBackend     │                                │
│                         └──────────────┬──────────────┘                                │
│                                        │                                               │
│                 ┌──────────────────────┴──────────────────────┐                        │
│                 ▼                                             ▼                        │
│   ┌───────────────────────────┐                 ┌───────────────────────────┐          │
│   │     CpuAvx2Backend        │                 │     GpuVulkanBackend      │          │
│   │ (SIMD AVX2/FMA + Rayon)   │                 │   (WGPU / Vulkan Compute) │          │
│   ├───────────────────────────┤                 ├───────────────────────────┤          │
│   │ • Fallback 100% universal │                 │ • AMD Radeon Vega / ROCm  │          │
│   │ • Zero-Copy Mmap          │                 │ • WGSL Compute Shaders    │          │
│   │ • 28 tok/s en 0.5B        │                 │ • Multi-Queue Dispatch    │          │
│   └───────────────────────────┘                 └───────────────────────────┘          │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📅 2. Fases de Ejecución

### 🔹 Fase 1: Infraestructura y Detección de Dispositivos GPU (`src/compute/gpu/`)
* **Dependencias:** Integrar `wgpu` con backend nativo Vulkan / DX12 / Metal.
* **Módulo de Contexto GPU (`src/compute/gpu/context.rs`):**
  * Inicialización de `wgpu::Instance`, `wgpu::Adapter`, `wgpu::Device`, `wgpu::Queue`.
  * Detección de arquitectura UMA (Unified Memory) para AMD Radeon APU (Zero-Copy Buffer Sharing).
  * Exposición de telemetría de GPU (`gpu_name`, `vram_allocated_mb`, `compute_units`).

### 🔹 Fase 2: Shaders de Cómputo WGSL (`src/compute/gpu/shaders/`)
* **`gemv_q4.wgsl`:** Mat-Vec de pesos cuantizados 4-bit / Q4_0 con descompresión paralela en workgroups (16x16 hilos).
* **`swiglu.wgsl`:** Fusión de activación SiLU y producto punto $Gate \odot Up$.
* **`rms_norm.wgsl`:** Reducción paralela y escalado en memoria compartida.

### 🔹 Fase 3: Integración en `GenomicLinear` y `GenomicLLM`
* Capacidad de descargar pesos de capas a buffers de GPU en el arranque (`offload_to_gpu()`).
* Ejecución híbrida: Forward de bloques de atención y MLP en GPU, manteniendo tokenizador y muestreo en CPU.
* Soporte para selección de capas: `gpu_layers = N` (permite balancear memoria entre CPU y GPU).

### 🔹 Fase 4: Telemetría en Web UI y Certificación
* Métricas de GPU en la interfaz web (`🎮 AMD Radeon Vega | VRAM: X MB | Compute: Vulkan`).
* Suite de pruebas de concordancia numérica (tolerancia $\le 10^{-4}$ entre CPU AVX2 y GPU Vulkan).
* Benchmarks comparativos de latencia y tok/s.
