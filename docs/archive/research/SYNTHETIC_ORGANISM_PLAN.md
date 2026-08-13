# 🧪 Proyecto: Micro-Organismo Sintético "Hola-GAJE"

## 1. Arquitectura del Espacio Genómico (10 MB)
Para mantenernos bajo el límite de 10 MB usando 2 bits (GAJE), el modelo tendrá la siguiente configuración aproximada:
*   **Capas:** 2 capas lineales densas (Hidden Dim: 2048).
*   **Vocabulario:** Reducido (solo caracteres básicos para "Hola").
*   **Parámetros:** ~40 millones de conexiones -> **10 MB en disco (2-bit)**.

## 2. El Algoritmo de Crianza (Monte Carlo Evolution)
En lugar de `optimizer.step()`, usaremos un ciclo de **Mutación y Selección**:
1.  **Población Inicial:** Un conjunto de pesos aleatorios cuantizados a 2 bits.
2.  **Mutación (Monte Carlo):** Se introducen cambios aleatorios en el ADN digital (cambiar una Base A por una Base G).
3.  **Evaluación (Fitness):** Se pasa un prompt "Hola" y se mide qué tan cerca están los logits de salida del objetivo.
4.  **Supervivencia:** Solo las mutaciones que reducen la pérdida de información se mantienen para la siguiente "generación".

## 3. Hoja de Ruta de Implementación
1.  **Scaffolding en Rust:** Crear un kernel mínimo que ejecute la multiplicación de matrices en 2-bit.
2.  **Motor de Evolución:** Implementar el bucle de Monte Carlo en Rust para máxima velocidad de iteración.
3.  **Entrenamiento:** Dejar que el modelo "evolucione" hasta que la salida sea coherente.
