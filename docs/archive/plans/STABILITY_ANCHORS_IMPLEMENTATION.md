# 🏗️ Plan Quirúrgico: Islas de Estabilidad (Stability Anchors)

**Objetivo:** Mejorar la coherencia gramatical del *Silver Adult* inyectando un 1-2% de pesos de alta precisión (Anclas) que sirvan como semillas de estabilidad para las fases de 2 bits.

---

## 1. El Concepto: Nucleación Semántica
En lugar de que todos los pesos sean de 2 bits (lo que causa "deriva semántica"), seleccionamos los pesos con mayor magnitud (los más influyentes) y los preservamos en **4 bits o 8 bits**. Esto crea "islas" de alta fidelidad que guían al resto de las neuronas, eliminando el ruido y la fragmentación.

---

## 2. Cambios en el Núcleo de Rust (`src/nn/linear.rs`)

### Tarea 2.1: Estructura `GenomicLinear`
*   **Acción:** Asegurar que el campo `anchors` (actualmente experimental) esté plenamente integrado en el flujo de inferencia circular.
*   **Implementación:**
    *   `anchors: Vec<f32>` o `Vec<u8>` (dependiendo de la densidad).
    *   `anchor_mask: Vec<u8>` (bitmask para identificar qué posiciones son anclas).

### Tarea 2.2: Kernel de Fusión (SIMD)
*   **Acción:** Modificar el `forward` para que, tras calcular el producto de 2 bits, inyecte los valores de las anclas mediante una operación de "Scatter-Add" optimizada.
*   **Resultado:** `y = (Weights_2bit * x) + (Anchors_Delta * x)`.

---

## 3. Estrategia de Inicialización (`src/io/loader.rs`)

### Tarea 3.1: Extracción de Anclas
*   **Acción:** Modificar `genomize_f32_core` para que, al inicializar el modelo, identifique el top 1% de los pesos con mayor valor absoluto.
*   **Acción:** Almacenar estos valores en el buffer de anclas del archivo `.gaje`.

### Tarea 3.2: Configuración del Preset
*   **Ajuste:** En el preset `silver_adult`, establecer `anchor_threshold: 0.05` (5% de anclas).
*   **Impacto en Tamaño:** Incremento estimado de **0.3 MB** (de 9.6 MB a 9.9 MB).

---

## 4. Fase de Entrenamiento (Fine-Tuning de Estabilidad)

### Tarea 4.1: Optimización de Anclas (IQAT+)
*   **Mecánica:** Durante el entrenamiento, el error (Loss) se aplicará prioritariamente a las anclas. Al ser de mayor precisión, pueden absorber el error gramatical que los 2 bits no pueden representar.
*   **Comando:**
    ```bash
    ./gaje-cli models/silver_adult.gaje --train dataset_coherence.txt --anchor-training --lr 0.005
    ```

---

## 5. Cronograma Quirúrgico (Fases)

1.  **Fase A (Infraestructura):** Habilitar el buffer de anclas en `GenomicLinear` y `smg1.rs`. (2 horas)
2.  **Fase B (Inferencia):** Integrar la inyección de anclas en el motor de fase circular. (3 horas)
3.  **Fase C (Migración):** Convertir el `silver_adult.gaje` actual al formato con anclas. (1 hora)
4.  **Fase D (Validación):** Chat loop para verificar la eliminación de caracteres extraños. (1 hora)

---
*Este plan es el puente final hacia la coherencia de nivel comercial.*
