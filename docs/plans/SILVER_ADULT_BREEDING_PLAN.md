# 🧬 Plan de Crianza: Silver Adult (10MB Circular High-Fidelity)

**Versión:** 1.0 (2026-05-27)
**Estado:** Pendiente de Inicio
**Objetivo:** Evolucionar del *Silver Fetus* al *Silver Adult* utilizando la **Topología Circular** y **Codificación de Fase** para alcanzar coherencia gramatical total en 2 bits, ocupando menos de 10 MB en disco.

---

## 1. Especificaciones de la Arquitectura (v3.0)

| Parámetro | Valor | Notas |
| :--- | :--- | :--- |
| **Tamaño en Disco** | ~9.6 MB | 2-bit puro (Circular) |
| **Bloques de Atención** | 12 Bloques Genomic Attention | Simetría profunda |
| **Dimensión Oculta** | 512 | Espacio latente denso |
| **Vocabulario** | 32,768 (Silver Tokenizer) | Poda optimizada de SmolLM2 |
| **Topología** | Circular (Potenciales Complejos) | Sin saturación de bordes |
| **KV-Cache** | DNA Circular (2-bit) | Soporte para 1M+ tokens |

---

## 2. Estrategia de Datos y Imprimación

### 2.1 Dataset Consolidado
*   **Fuente Principal:** `data/datasets/dataset_es_ext.txt` (63,000+ líneas).
*   **Composición:**
    *   30% Diálogo Natural (Identidad y Razonamiento).
    *   40% Estructuras Técnicas (Código Rust, Lógica Matemática).
    *   30% Conocimiento Enciclopédico (Hechos y Relaciones).

### 2.2 Tokenizador Silver
*   **Base:** SmolLM2-135M podado a 32,768 tokens.
*   **Estado:** Completado y verificado en la Fase 5.0.

---

## 3. Fases del Entrenamiento "Born-Genomic"

### Fase 1: Cimentación Algebraica (Epoch 0)
*   **Acción:** Inicializar los pesos utilizando el campo ciclotómico $\mathbb{Q}(\zeta_{16})$.
*   **Objetivo:** Establecer la "rejilla de inteligencia" rígida donde los ángulos de fase están perfectamente distribuidos a $0^\circ, 90^\circ, 180^\circ$ y $270^\circ$.

### Fase 2: Imprimación de Identidad (Epoch 1 - 20)
*   **Acción:** Entrenamiento intensivo sobre el núcleo de identidad ("¿Quién eres?", "Eres GAJE").
*   **Meta:** Alcanzar Resonancia Total (1.00 Fitness) en el subconjunto de identidad de forma instantánea mediante la interferencia constructiva del motor circular.

### Fase 3: Destilación por Resonancia Circular (Epoch 21 - 100)
*   **Maestros:** `Qwen2-1.5B` y `SmolLM2-135M`.
*   **Mecánica:** El modelo *Silver Adult* no copia los valores de los maestros, sino que intenta **sincronizar su fase** con la topología de activación del Council of Teachers.
*   **Optimizer:** `NativeGenomicTrainer` (Rust-Native) con un Learning Rate adaptativo según la perplejidad de fase detectada.

### Fase 4: Consolidación Gramatical (Massive Training)
*   **Acción:** Ejecutar el ciclo masivo de 3-4 horas sobre el dataset completo de 63k líneas.
*   **Mecanismo:** Aplicar **Inhibición Lateral** (implementada en el Hito 2 de la hoja de ruta) para forzar la especialización de las cabezas de atención.

---

## 4. Validación y Métricas de Calidad

1.  **Needle in a Haystack (128k):** El objetivo es > 90% de recuperación de información en cualquier posición del círculo de atención.
2.  **Perplejidad de Fase (Phase-PPL):** Se espera una reducción del 40% en PPL comparado con el modelo lineal previo tras 100 épocas.
3.  **Chat Loop Test:** El modelo debe generar frases de más de 30 palabras sin entrar en bucles de repetición (validado con la penalización de repetición nativa).

---

## 5. Comando de Ejecución (Proyectado)

```bash
# Inicialización y entrenamiento masivo
./target/release/gaje-cli --init --preset silver-adult \
    --train --dataset data/datasets/dataset_es_ext.txt \
    --epochs 100 --lr 0.02 --mode circular
```

---
*Plan estratégico diseñado por Gemini CLI bajo el protocolo GAJE-Flow v1.3*
