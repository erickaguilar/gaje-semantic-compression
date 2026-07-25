# 🧠 Guía de Enriquecimiento Semántico (Semantic Enrichment)

El proceso de "Enriquecimiento Semántico" es fundamental en el Protocolo GAJE. Cuando comprimimos un modelo de alta capacidad (ej. SmolLM2 o Qwen2) al formato genómico de **2 bits**, se introduce un "ruido de cuantización" severo. Esto provoca que el modelo pierda coherencia, invente palabras o sufra de sesgo de idioma (ej. responder en inglés cuando se le habla en español).

Este documento detalla la estrategia de tres niveles para restaurar y potenciar la inteligencia del organismo genómico.

---

## Nivel 1: Datos (Expansión Estructural)
La principal causa de la pérdida de coherencia es la falta de volumen de datos durante la "crianza" (fine-tuning) post-compresión. 20 líneas de diálogo no son suficientes para que las redes neuronales de 2 bits se estabilicen.

### A. Volumen Necesario
Para superar la barrera de los 2 bits, el modelo necesita un **mínimo de 500 a 1,000 interacciones** en el idioma destino.

### B. Composición del Dataset (Regla 70/30)
El dataset de entrenamiento debe ser balanceado:
*   **70% Conocimiento Estructural:** Fragmentos de libros, artículos genéricos de Wikipedia, conversaciones cotidianas. Esto le enseña al modelo la "música" y la gramática del español.
*   **30% Conocimiento Especializado:** Diálogos específicos sobre tu dominio (ej. GAJE, Rust, Android, IA).

### C. Generación Sintética
Se recomienda usar modelos más grandes (ej. GPT-4, Claude) para generar un archivo `dataset_entrenamiento.txt` masivo con pares estructurados:
```text
Usuario: [Pregunta]
Asistente: [Respuesta coherente y gramaticalmente perfecta]
```

---

## Nivel 2: Arquitectura (Anclaje Selectivo)
El Protocolo GAJE incluye un sistema de "Anclas" (Anchors). Las anclas permiten guardar los pesos neuronales más importantes en **alta fidelidad (f16 o f32)**, mientras el 99% del modelo permanece en 2 bits.

### A. Cómo funciona
Si un peso tiene un valor extremo (ej. superior a `0.1` o inferior a `-0.1`), es crítico para la comprensión del modelo. Al ajustar el `anchor_threshold` durante la destilación, el motor GAJE protege estos pesos vitales.

### B. Aplicación Práctica
En el script de carga o destilación (`GenomicLayer`), modifica el umbral:
*   **`anchor_threshold = -1.0`**: Desactiva las anclas. Máxima compresión, peor coherencia.
*   **`anchor_threshold = 0.1`**: Anclaje estándar. Protege la gramática básica.
*   **`anchor_threshold = 0.05`**: Anclaje de Alta Fidelidad. Retiene el vocabulario complejo y mejora drásticamente el razonamiento lógico.

---

## Nivel 3: Entrenamiento (IQAT y Evolución)
El entrenamiento estándar por gradientes a veces no es suficiente para corregir el ruido asimétrico de los 2 bits.

### A. IQAT (Activation-Aware Training)
Entrenar las activaciones (SwiGLU, GeGLU) es más importante que entrenar los pesos base. El modelo debe aprender a "ignorar" el ruido de los 2 bits escalando sus compuertas lógicas. En GAJE, esto se gestiona mediante la variable de "Homeostasis" (`h_scale`).

### B. Refinamiento Evolutivo (Algoritmo Genético)
La combinación ganadora para dispositivos móviles (Android/Termux) es el entrenamiento híbrido:
1.  **Fase 1 (Gradientes):** 20 a 50 épocas de descenso de gradiente clásico para aprender la dirección semántica.
2.  **Fase 2 (Evolución):** 20 a 50 generaciones de mutación aleatoria en la escala homeostática (`h_scale`). La evolución es capaz de saltar los mínimos locales y estabilizar las activaciones ruidosas sin el enorme costo de memoria de la retropropagación completa.

---

## 🎯 Plan de Acción Resumido:
Para tu próxima iteración de modelo:
1.  Expande tu archivo `data/datasets/dataset_entrenamiento.txt` a 500+ líneas.
2.  Edita `python/gaje/nn/stabilized.py` y ajusta `anchor_threshold=0.05` para retener más inteligencia.
3.  Ejecuta `./scripts/gaje_distill.sh` con el nuevo dataset.
4.  Realiza el refinamiento con `./scripts/gaje_train.sh` (35 épocas Gradientes + 20 gen Evolución).
