# 🛠️ Guía de Uso: GAJE-CLI

Esta guía detalla el uso del motor nativo `gaje-cli`, la herramienta principal para la gestión, entrenamiento e inferencia de organismos genómicos.

## 📋 Comandos y Parámetros

| Parámetro / Comando | Descripción | Valor por Defecto |
| :--- | :--- | :--- |
| **`--model <path>`** | Ruta al archivo del modelo genómico (`.gaje`). | Requerido |
| **`--prompt "<text>"`** | Texto inicial para generar una respuesta (Modo Chat). | - |
| **`--train <file>`** | Archivo de texto para entrenamiento nativo (Fase 2). | - |
| **`--epochs <num>`** | Número de épocas para el entrenamiento. | `10` |
| **`--init <path>`** | Inicializa un nuevo organismo desde cero. | - |
| **`--preset <type>`** | Preset de arquitectura (`gold_embryo`, `silver_adult`, etc.) | `default` |
| **`--import <path.gguf>`** | Importa y transmuta un modelo GGUF a formato GAJE. | - |
| **`--output <path>`** | Ruta de guardado para el modelo importado o entrenado. | - |
| **`--threshold <val>`** | Umbral de precisión para las Anclas de Estabilidad (F16). | `0.1` |
| **`--dni-ingest <file>`** | Archivo de conocimiento para inyección DNI directa. | - |
| **`--intensity <val>`** | Intensidad de la inyección DNI (0.0 a 1.0). | `0.01` |
| **`--scale <val>`** | Factor de escala de aprendizaje (Learning Rate). | `0.02` |
| **`--resonance <val>`** | Peso de la resonancia toroidal durante el entrenamiento. | `0.05` |
| **`--inspect`** | Muestra metadatos y estructura interna del modelo. | - |
| **`--tokenize "<text>"`** | Muestra cómo el modelo trocea el texto en tokens. | - |
| **`--iqat`** | Activa el entrenamiento consciente de la identidad (IQAT). | `false` |

## 🧬 Presets de Arquitectura (`--preset`)

| Preset | Parámetros (Embd/Layers/Heads) | Vocabulario | Descripción |
| :--- | :--- | :--- | :--- |
| `gold_embryo` | 384 / 8 / 6 | 49,152 | El origen geométrico base. |
| `micro_organism` | 128 / 2 / 4 | 32,768 | Prototipo ultra-ligero para debugging. |
| `silver_fetus` | 512 / 12 / 8 | 32,768 | Modelo intermedio de entrenamiento. |
| `silver_adult` | 512 / 12 / 8 | 32,768 | El estándar de 10MB con física circular. |

## 🚀 Ejemplos Rápidos

### 1. Inferencia (Chat)
```bash
gaje-cli --model models/production/silver_adult_steel.gaje --prompt "Hola, ¿cómo funciona el toroide?"
```

### 2. Importación desde GGUF (Nacimiento)
```bash
gaje-cli --import models/source/smollm2.gguf --output models/born/embryo.gaje --threshold 0.15
```

### 3. Entrenamiento Nativo (Crianza)
```bash
gaje-cli --model models/born/embryo.gaje --train data/datasets/mosaic_dataset.txt --epochs 20 --scale 0.01
```

---
*Documento generado automáticamente bajo el estándar GAJE-Flow v1.0.0*
