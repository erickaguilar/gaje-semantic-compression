# 🧬 GAJE Protocol: Catálogo de Ejemplos

Este directorio contiene demostraciones prácticas del Protocolo GAJE (DNA Semantic Compression), organizadas por su propósito y nivel de madurez técnica.

## 🚀 Demos Principales (`examples/core_demos/`)
Ejemplos recomendados para entender y utilizar el motor genómico v0.7.0+.

*   **`chat_genomico.py`**: Interfaz de chat interactiva que utiliza modelos GGUF genomizados al vuelo. Soporta muestreo avanzado (Top-P, Temperatura) y métricas de rendimiento en tiempo real.
*   **`born_genomic_demo.py`**: Demuestra cómo inicializar un organismo genómico desde cero (pesos aleatorios), sentando las bases para el entrenamiento nativo en el dispositivo.

## 🎨 Visualización y UX (`examples/visual_demos/`)
Demos enfocadas en la representación visual del "metabolismo" de los modelos.

*   **`autonomic_demo.py`**: Genera un mapa de calor visual en la terminal que muestra cómo el `SignalToNoiseBalancer` asigna diferentes precisiones (2-bit, 4-bit, 6-bit) según la entropía de la señal.
*   **`app.py`**: Interfaz web (Gradio) original para interactuar con los modelos (requiere dependencias `dev`).

## 🧪 Investigación y Legado (`examples/legacy_research/`)
Scripts históricos utilizados durante el desarrollo de las Fases 1-11. Útiles para entender la evolución de la búsqueda asimétrica (ADC) y HNSW.

*   **`search_demo.py`**: Búsqueda semántica básica sobre ADN de 2 bits.
*   **`hnsw_demo.py`**: Demostración de búsqueda sub-lineal mediante grafos.
*   **`multimodal_demo.py`**: Experimentos iniciales con proyecciones CLIP genómicas.

---
*Para ejecutar estos ejemplos, asegúrese de tener compilado el núcleo en Rust (`cargo build --release`) y el entorno virtual activo.*
