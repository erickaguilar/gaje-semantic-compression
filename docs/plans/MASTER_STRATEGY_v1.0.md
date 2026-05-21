# 🧬 GAJE-Flow: Estrategia Maestra de Desarrollo (v1.0)

Este documento unifica la visión, arquitectura y hoja de ruta para el **Protocolo GAJE** y el desarrollo de **Modelos Born-Genomic**. Reemplaza todos los planes previos de estabilización y soberanía.

---

## 1. 🔭 Visión Estratégica
El objetivo final es crear un **Cerebro de Guardia (Always-on Edge AI)** con consumo energético cercano a cero, capaz de aprender y evolucionar localmente en dispositivos móviles mediante un emulador neuromórfico nativo.

- **Soberanía Total:** Ejecución e inferencia 100% Rust, sin dependencias de Python.
- **Evolución Local:** El modelo "crece" y se ajusta según la interacción del usuario mediante mutaciones bitwise.
- **Privacidad Absoluta:** RAG local y procesamiento de contextos extremos en el dispositivo.

---

## 2. 🏛️ Arquitectura Técnica: SoA y Metabolismo Híbrido
Para maximizar el rendimiento en procesadores ARM (Android/NEON), GAJE utiliza una estructura **Struct-of-Arrays (SoA)** que evita condicionales durante el cálculo.

### Capas de Precisión Mixta (Metabolismo):
1.  **Base (2-bit):** El 95%+ de los datos. Procesado mediante SIMD NEON ciego y lineal.
2.  **Epigenética (4-bit):** Corrección selectiva de anclas para reducir el error de cuantización.
3.  **Tripletes (6-bit):** Refinamiento de Outliers críticos para estabilizar activaciones (sustituyendo SwiGLU por variantes más estables como GELU acotado).

---

## 3. 🧬 Hoja de Ruta Born-Genomic (Aprendizaje Nativo)
El camino hacia la inteligencia coherente no es la destilación, sino el **Nacimiento Genómico** (entrenamiento nativo en 2-bits).

### Fase 1: Ajustes de Arquitectura (Actual)
- Implementar **GenomicNorm** para evitar el drift semántico.
- Optimizar el **SpikingEvolutionEngine** con paralelismo masivo (`Rayon`).
- Definir la **Pérdida Genómica** basada en resonancia semántica y eficiencia energética.

### Fase 2: Micro-Genome (Validación)
- Entrenar un micro-modelo (10M-30M parámetros) desde cero.
- Validar la capacidad de memorización y perplejidad en datasets locales.
- Integrar el comando `--train` directamente en `gaje-cli`.

### Fase 3: Escalado e Industrialización
- Transferencia de entrenamiento pesado (GPU/Cuda) a organismos comprimidos (.gaje).
- Habilitar Fine-Tuning local continuo (Refine Centroids).

---

## 4. 🗄️ Infraestructura: Base de Datos Semántica
GAJE no es solo un modelo, es una base de datos vectorial adaptativa basada en `redb`.

- **Formato .gaje Unificado:** Un solo archivo binario que contiene pesos, metadatos y el **Registro Epigenético**.
- **Registro de Mutaciones:** Historial temporal de refinamientos que permite el **"Time-Travel" (Rollback)** ante olvido catastrófico.
- **Zero-Copy Loading:** Uso de archivos mapeados en memoria (Mmap) para carga instantánea.

---

## 5. 🚀 Estrategia de Producto: Edge AI SDK
Transformar el motor CLI en una herramienta accesible para desarrolladores y usuarios.

1.  **SDK Nativo:** Bindings JNI/FFI para integración transparente en Android (Kotlin) e iOS (Swift).
2.  **Power-Awareness:** Programador de hilos consciente de la batería (big.LITTLE cores).
3.  **App Piloto:** Desarrollo de una aplicación móvil demostrativa que demuestre RAG local y tokens por segundo en tiempo real.

---
*Este plan es el único documento de referencia activo para el desarrollo estratégico.*
