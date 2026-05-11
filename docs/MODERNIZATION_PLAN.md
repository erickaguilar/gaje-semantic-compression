# Plan de Modernización de Ejemplos (GAJE v0.6.0+)

## 🎯 Objetivo
Alinear todos los ejemplos demostrativos con el núcleo científico de la **Fase 12** y eliminar dependencias frágiles.

---

## 🛠️ Fase 1: Actualización de Chat Genómico (Prioridad Alta)
- **Acción:** Refactorizar `chat_genomico.py` para usar `gaje.nn.stabilized.GenomicLLM`.
- **Mejora:** Soporte nativo para modelos F16 (DGI) y precisión mixta.
- **Resultado:** Chat funcional, estable y eficiente en Termux.

## 🎨 Fase 2: Visualización de Metabolismo (Fase 12)
- **Acción:** Actualizar `app.py` (Gradio) o crear `autonomic_demo.py`.
- **Mejora:** Mostrar visualmente cómo el `SignalToNoiseBalancer` activa los strands Epigenéticos (4-bit) y Tripletes (6-bit) en tiempo real.
- **Visual:** Un "Heatmap" del ADN indicando zonas de alta fidelidad.

## 🧹 Fase 3: Limpieza y Estabilización Multimodal
- **Acción:** Limpiar `multimodal_demo.py`.
- **Mejora:** Eliminar el monkeypatch de `scipy`. Usar implementaciones nativas de similitud coseno en Rust o NumPy.
- **Actualización:** Cambiar el modelo base a uno compatible con DGI para evitar errores de dimensiones.

## 🕸️ Fase 4: Optimización HNSW & Búsqueda
- **Acción:** Integrar `GajeIndex` con capas de precisión mixta en `search_demo.py`.
- **Mejora:** Demostrar cómo la búsqueda asimétrica (ADC) se beneficia de los strands de corrección.

---
*Estado: Iniciando Fase 1...*
