# 🧬 Plan de Trabajo: Desarrollo de Modelos "Born-Genomic" (GAJE Nativo)

**Estado:** En Planificación  
**Rama:** `feature/gaje-native-llm`  
**Objetivo Principal:** Entrenar un modelo de lenguaje (LLM) desde cero (pesos aleatorios) utilizando directamente el protocolo de compresión semántica GAJE (2-bits por dimensión), eliminando la dependencia de destilar modelos *float32* pre-entrenados.

---

## 🔬 Fase 1: Infraestructura de Entrenamiento Genómico (Semanas 1-2)
*El modelo actual puede inferir y actualizar centroides, pero necesitamos un bucle de entrenamiento completo.*

1. **Autograd Híbrido (PyTorch a Rust):**
   - Implementar un bucle de entrenamiento donde el *forward pass* se ejecute en el motor de Rust (para aprovechar el cálculo genómico real).
   - Utilizar PyTorch únicamente como motor de cálculo de gradientes (*Cross-Entropy Loss*) sobre los *logits* de salida.
   - Enrutar los gradientes resultantes hacia las capas genómicas utilizando la función `refine_with_grads` recientemente implementada.
2. **Dataset Pipeline Ligero:**
   - Crear un cargador de datos optimizado para Termux/Móvil que consuma micro-datasets (ej. *TinyStories* o un corpus muy específico y reducido).
   - Generación de *batches* y manejo de secuencias cortas (context length: 128-256 tokens) para iteración rápida.
3. **Métricas de Convergencia en Tiempo Real:**
   - Monitoreo de Perplejidad (PPL) y *Loss* durante el entrenamiento.
   - Visualización de la evolución de los centroides genómicos (verificar que los 4 valores base del ADN divergen y se especializan).

## 🛠️ Fase 2: Ajustes Arquitectónicos Nativos (Semanas 2-3)
*SwiGLU demostró ser destructivo para modelos pequeños de 2 bits. Un modelo nacido genómico debe tener una arquitectura adaptada a su naturaleza de baja precisión.*

1. **Reemplazo de Activaciones No-Lineales:**
   - Experimentar con activaciones más estables frente al ruido de cuantización (ej. `ReLU`, `GELU` suave, o una variante acotada) en la `ArchitectureConfig` de `gaje_native`.
2. **Normas Adaptativas (GenomicNorm):**
   - Incorporar normalización en puntos críticos para evitar el *Semantic Drift* acumulativo sin recurrir al *clamping* destructivo.
3. **Hiper-parámetros del ADN:**
   - Ajustar el tamaño del bloque (actualmente 32) y la densidad de "Anclas" nativas. ¿Necesita un modelo nacido genómico anclas F32, o puede converger 100% en 2 bits?

## 🌱 Fase 3: Prueba de Concepto "Mini-Genome" (Semanas 3-4)
*Validación empírica de la capacidad de aprendizaje.*

1. **Entrenamiento de un Micro-Modelo (10M - 30M parámetros):**
   - Dimensiones: `n_embd = 256`, `n_blocks = 4`, `n_heads = 4`.
   - Entrenar durante 10,000 iteraciones en el entorno local (Termux) o servidor si se requiere velocidad.
2. **Validación de Memorización:**
   - Verificar si el modelo es capaz de hacer *overfit* (memorizar) un texto pequeño de 5 páginas. Si la pérdida de entropía genómica baja a casi cero, el modelo *puede* aprender.
3. **Persistencia Genómica (GAJE v3 Archive):**
   - Asegurar que el estado entrenado (ADN + Centroides refinados) se guarde correctamente usando `GAJEArchive` y pueda ser recargado sin degradación.

## 🚀 Fase 4: Escalado y Especialización (Mes 2)
*Pasar de un experimento a un modelo útil.*

1. **Transferencia de Hardware (GPU to Edge):**
   - Desarrollar un script para ejecutar el bucle de entrenamiento pesado en una GPU tradicional (exportando la lógica de Rust a CUDA/Triton temporalmente), y exportar el "organismo" resultante en formato `.gaje` para su ejecución eficiente en Termux.
2. **Fine-Tuning Local Continuo:**
   - Dado que el modelo fue entrenado nativamente en GAJE, habilitar la capacidad de *Continuous Learning* en el dispositivo móvil del usuario mediante `refine_centroids` basado en las interacciones diarias.
3. **Evaluación de Complejidad:**
   - Comparar el rendimiento de un modelo "Born-Genomic" de 100M parámetros contra un SmolLM2 de 135M parámetros destilado. Hipótesis: El nacido genómico retendrá mejor los hechos porque sus centroides crecieron adaptados al ruido, no fueron forzados a reducirse después.

---
### 🏁 Siguiente Paso Inmediato (Acción requerida):
**Iniciar la Fase 1, Paso 1:** Crear el archivo `python/gaje/nn/trainer.py` que construirá el puente entre la pérdida (*Loss*) y los gradientes genómicos (`refine_with_grads`), permitiendo al modelo `gaje_native` aprender su primer token.