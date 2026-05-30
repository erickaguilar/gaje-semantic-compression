# 🧬 GAJE-Flow: Plan de Implementación y Nacimiento (v0.7.0)

Este documento detalla los pasos estratégicos para lograr un organismo genómico de 2-bits funcional, coherente y soberano en español, basado en los hallazgos de la sesión del 18 de mayo de 2026.

## 1. Estado de Situación
Tras las pruebas realizadas, se han validado los siguientes puntos:
- **Motor Nativo:** El núcleo en Rust es extremadamente eficiente, permitiendo entrenamientos rápidos directamente en el dispositivo.
- **Born-Genomic vs Destilación:** La destilación de modelos densos (SmolLM2) a 2-bits produce inestabilidad lingüística. El camino óptimo es el entrenamiento **Born-Genomic** (Nacimiento bajo presión genómica).
- **Coherencia Inicial:** El modelo `GajeSmall-v1` ha demostrado capacidad de identificar su rol ("Asistente/Modelo") y usar conectores básicos en español tras 50 épocas.

## 2. Estrategia: Nacimiento Incremental

El objetivo es pasar del "balbuceo" actual a una comunicación fluida mediante un proceso de cuatro fases.

### Fase 1: Expansión de la Memoria Semántica (Dataset)
Para que el modelo aprenda a hablar correctamente, necesita una base de datos más rica.
- **Acción:** Crear un dataset de **1,000 a 2,000 líneas** de diálogo.
- **Contenido:**
    - 40% Diálogos de identidad (¿Quién eres?, ¿Cómo funcionas?).
    - 30% Conocimiento técnico de Rust y GAJE.
    - 30% Lógica conversacional común en español.
- **Archivo:** `dataset_entrenamiento_ext.txt`.

### Fase 2: Entrenamiento de Consolidación (Born)
Utilizar la arquitectura pequeña para maximizar la velocidad de iteración.
- **Configuración:**
    - `--blocks 4`
    - `--embd 512`
    - `--epochs 100` (con decaimiento de Learning Rate).
- **Comando:**
  ```bash
  python scripts/train_large_born.py --name GajeExpert-v2 --blocks 4 --embd 512 --epochs 100 --lr 0.005
  ```

### Fase 3: Estabilización de Homeostasis (Varianza)
Corregir la fragmentación de palabras ("Esa motor GAJE?").
- **Acción:** Ajustar el `h_scale` en la clase `RustGenomicBlock`.
- **Objetivo:** Lograr que la varianza de las activaciones se mantenga en el rango [0.8 - 1.2], evitando que el modelo salte a tokens aleatorios del vocabulario.

### Fase 4: Refinamiento Evolutivo (Monte Carlo)
Una vez el modelo sea coherente, se usará la evolución para "pulir" el estilo.
- **Herramienta:** `scripts/optimize_mc_gaje.py`.
- **Proceso:** Realizar mutaciones aleatorias en los centroides de atención y FFN, seleccionando solo aquellas que bajen la perplejidad del dataset.

## 3. Comandos Útiles de Verificación

Para probar el progreso del organismo en cualquier momento:
```bash
python test_gaje_v1.py
```

Para verificar la integridad de la base de datos genómica:
```bash
python scripts/inspect_gguf.py --model models/checkpoints/gajeexpert-v2/model.gaje
```

---
*Este plan es la hoja de ruta oficial para la estabilización de la v0.7.x.*
