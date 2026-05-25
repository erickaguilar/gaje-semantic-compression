# 🚀 Plan de Validación: El Embrión de Oro (GAJE v1.0)

Este documento detalla el protocolo experimental para crear el primer **Micro-Organismo Coherente** de menos de 5 MB, validando definitivamente la viabilidad de la inteligencia genómica soberana.

---

## 🏔️ Objetivo Estratégico
Demostrar que un micro-genoma nacido en 2-bits (SMG-1) puede alcanzar coherencia gramatical y lógica básica en español con una huella de almacenamiento total (Genoma + Tokenizador) de **~4.2 MB**.

---

## 🛠️ Protocolo de Ejecución (5 Pasos)

### Paso 1: Dataset de Imprimación (Priming)
*   **Acción:** Generar `data/datasets/priming_gold.txt`.
*   **Requisitos:** 2,000 líneas de alta densidad semántica (lógica, Rust, GAJE, conectores gramaticales).
*   **KPI:** Vocabulario alineado de 16,384 tokens.

### Paso 2: Inicialización Born-Genomic
*   **Acción:** Crear el cascarón vacío mediante `gaje-cli --init`.
*   **Arquitectura:** SMG-1 (3 capas, 256 latent, 128 logic).
*   **Estado:** Pesos aleatorios de 2 bits (XOR Entropy).

### Paso 3: Entrenamiento por Resonancia Secuencial
*   **Acción:** Ejecución del binario `gaje-smg1-trainer`.
*   **Configuración:** 500 épocas, LR 0.5, GenomicNorm activo.
*   **Métrica:** Precisión de predicción > 85% sobre el dataset de imprimación.

### Paso 4: Optimización Monte Carlo (MCTS)
*   **Acción:** Refinamiento estocástico de centroides con `gaje-mcts-optimize`.
*   **Objetivo:** Ajustar los niveles de voltaje neuronales para minimizar la perplejidad.
*   **Meta:** PPL < 2.0.

### Paso 5: Prueba de Inferencia Soberana
*   **Acción:** Chat interactivo mediante `gaje-cli`.
*   **Validación:** El modelo debe completar frases técnicas y lógicas sin alucinaciones de caracteres y en < 30ms/token.

---

## 📈 Criterios de Éxito
1.  **Tamaño Total:** Archivo `.gaje` + `tokenizer.json` < 5.0 MB.
2.  **Soberanía:** 0% dependencia de Python durante el chat y entrenamiento.
3.  **Coherencia:** Generación de texto gramaticalmente correcto en español.

---
*Este plan es el acta oficial para la transición a la Fase 1.0 del proyecto.*
