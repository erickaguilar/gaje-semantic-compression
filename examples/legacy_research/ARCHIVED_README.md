# Archivo de Ejemplos Históricos (Legacy)

Este directorio contiene ejemplos y demostraciones correspondientes a las primeras fases (Fase 1 a Fase 11) del desarrollo del protocolo GAJE (DNA Semantic Compression).

Han sido archivados aquí con fines de documentación histórica, ya que las interfaces que utilizan han sido reemplazadas o modernizadas por la actual arquitectura de Inferencia Genómica Nativa y la disposición de memoria Struct-of-Arrays (SoA).

## Archivos Archivados

*   **`app.py`**: El demo original de Gradio. No incluye la visualización del mapa de calor de hebras Epigenéticas (4-bit) y Tripletes (6-bit) introducida en la Fase 12.
*   **`multimodal_demo.py`**: Demo de embeddings multimodales. Se archivó porque contenía un "monkeypatch" frágil sobre `scipy` para evitar errores de validación, en lugar de utilizar las funciones de similitud nativas y estables escritas en Rust.
*   **`search_demo.py` & `hnsw_demo.py`**: Ejemplos de búsqueda vectorial. Utilizaban la búsqueda de distancia asimétrica (ADC) base, pero no incorporaban el enrutamiento dinámico de memoria del `SignalToNoiseBalancer` para la precisión mixta.

Para ver las implementaciones actualizadas y recomendadas, consulte el directorio raíz `examples/` en el proyecto principal.
