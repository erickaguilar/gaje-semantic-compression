# 🎮 Estrategia de Aceleración por Hardware (AMD Radeon)

**Fecha:** 16 de Mayo de 2026
**Hardware Local Detectado:** AMD Radeon(TM) Graphics (~3GB VRAM)

El Protocolo GAJE ha sido diseñado con una filosofía **CPU-First** (aprovechando AVX2/NEON) para garantizar la portabilidad extrema en dispositivos como smartphones (Termux) y laptops de gama baja. Sin embargo, el equipo actual cuenta con una iGPU AMD Radeon que abre la puerta a futuras optimizaciones de rendimiento masivo.

Este documento analiza la viabilidad de utilizar esta GPU y compara los entornos Windows y Linux para este propósito.

---

## 1. El Desafío Genómico en GPU

El formato `.gaje` almacena los pesos en estructuras altamente empaquetadas (2 bits por peso + máscaras de precisión mixta de 4/6 bits). Las GPUs están optimizadas para multiplicar matrices masivas de `float16` o `float32`. 

Para aprovechar la Radeon, necesitaríamos implementar un **De-quantization Shader** (un kernel en la GPU que desempaquete los bits al vuelo en memoria compartida) antes de la multiplicación de matrices.

## 2. Tecnologías Disponibles para AMD

Existen tres vías principales para programar en la GPU AMD:

1.  **ROCm (Radeon Open Compute):** El ecosistema nativo de AMD (equivalente a CUDA).
2.  **Vulkan Compute:** API gráfica multiplataforma de bajo nivel.
3.  **DirectML / WebGPU:** APIs de mayor nivel.

## 3. Windows vs Linux: ¿Cuál es mejor para AMD?

Para la aceleración de IA en hardware AMD, **Linux es indiscutiblemente superior a Windows**, por las siguientes razones:

### 🐧 Linux (Recomendado)
*   **ROCm Soporte Nativo:** AMD ROCm funciona de manera nativa y con máximo rendimiento en Linux. Es el estándar de la industria para ejecutar PyTorch, llama.cpp o Rust-ML en AMD.
*   **Gestión de VRAM:** El kernel de Linux (Mesa/AMDGPU) gestiona la memoria de las iGPUs de forma mucho más eficiente para cargas de cómputo (Compute Shaders).
*   **Rust Ecosystem:** Las librerías de Rust para ROCm y Vulkan-Compute suelen compilar más rápido y tener menos fricción en entornos Unix.

### 🪟 Windows (Estado Actual)
*   **Limitaciones de ROCm:** Históricamente, el soporte de ROCm en Windows (HIP SDK) ha sido experimental, inestable o limitado a ciertas tarjetas de gama alta (RDNA 2/3). Las iGPUs suelen quedar excluidas o requieren WSL2 (Windows Subsystem for Linux), lo que añade latencia y consumo de RAM.
*   **Vulkan/DirectML:** Es la única alternativa viable 100% nativa en Windows para esta gráfica, pero requiere reescribir los kernels desde cero en HLSL/GLSL, alejándonos de la simplicidad de Rust.

---

## 4. Conclusión y Roadmap Propuesto

Actualmente, **mantener la Soberanía Nativa en la CPU (Rust + AVX2)** es la estrategia correcta, ya que nos permite alcanzar el objetivo de los modelos de "10 MB" sin atarnos a drivers privativos.

Si en el futuro deseamos integrar la Radeon para escalar a modelos más grandes (ej. > 3B parámetros), el camino recomendado es:

1.  **Migrar el entorno de desarrollo a Linux** (o configurar un entorno nativo sin WSL para acceso directo al PCIe).
2.  **Integrar `vulkano` (Rust):** Escribir Compute Shaders en Vulkan. Vulkan es la mejor opción para mantener el espíritu de "Soberanía Nativa", ya que funciona igual de bien en Linux, Windows y Android (Termux), permitiendo que el mismo código GPU corra en un PC y en un smartphone sin depender de ROCm o CUDA.

*Fin del reporte.*
