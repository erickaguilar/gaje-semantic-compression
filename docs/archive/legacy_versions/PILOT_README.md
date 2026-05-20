# 👨‍✈️ Guía de Piloto: Protocolo GAJE v0.6.1

Bienvenido a la frontera de la IA genómica. Esta versión habilita la ejecución y el aprendizaje de modelos masivos en dispositivos móviles.

## 🕹️ Operaciones Críticas

### 1. Inferencia con Memoria Infinita (KV-Cache DNA)
El motor ahora comprime el contexto 16x. Para activarlo, usa el pipeline estabilizado:
```python
from gaje.processing.pipeline import GenomicLLM
llm = GenomicLLM("path/to/model.gguf")
logits = llm.forward(tokens)
```

### 2. Aprendizaje Mobile-Native (Refinamiento Local)
Ajusta la inteligencia del modelo directamente en el dispositivo basado en tus datos:
```python
# Refina los centroides de una capa lineal en Rust
layer.linear.refine_centroids(input_vector, target_output, lr=0.01)
```

### 3. Destilación IQAT Masiva
Para transformar un modelo GGUF estándar en un Organismo Genómico refinado:
```bash
python -m gaje.nn.distiller
```

## 📊 Métricas de Control
- **PPL Target:** Debe mantenerse en **1.60**.
- **RAM Footprint:** ~84MB para modelos de 0.5B.
- **Fidelity:** > 0.96 CosSim.

---
*GAJE v0.6.1: IA que no solo comprime, sino que evoluciona localmente.*
