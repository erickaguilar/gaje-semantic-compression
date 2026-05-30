# 🧬 Plan de Crianza Genómica: Hacia el Organismo Autonómico Coherente (v1.0)

Este plan detalla el protocolo para transformar un **Organismo Autonómico** (nacido de un alfabeto aleatorio de 2 bits) en una inteligencia semántica funcional mediante evolución dirigida, sin dependencia de modelos maestros (destilación).

---

## 🏔️ Fase 1: El Embrión de Oro (Arquitectura)
Para garantizar una evolución rápida en hardware ARM, el embrión debe ser ultraligero.
*   **Tamaño Objetivo:** < 10 MB.
*   **Arquitectura:**
    *   `n_embd`: 384
    *   `n_blocks`: 8
    *   `n_head`: 6
    *   `vocab_size`: 16,384 (Vocabulario destilado y optimizado).
*   **Estado Inicial:** Inicialización "Born-Genomic" (Ruido puro en 2-bits).

---

## 🎲 Fase 2: El Motor de Selección (Fitness)
A diferencia de la destilación que copia activaciones, la **Crianza** utiliza una función de aptitud basada en la supervivencia de la información.
*   **Métrica de Selección:** Perplejidad local (PPL) sobre un dataset de "Imprimación" (ej. Conceptos lógicos básicos).
*   **Algoritmo:** **Monte Carlo Island Model**.
    *   Se crean 10 variaciones (islas) del genoma.
    *   Se aplican mutaciones XOR aleatorias al ADN.
    *   Solo sobrevive el 20% con menor perplejidad.

---

## 🧠 Fase 3: Refinamiento de Centroides (Epigenética)
Los centroides (los valores reales de A, C, G, T) actúan como la expresión del genoma.
*   **Optimización Estocástica:** Usar `scripts/optimize_mc_gaje.py` para ajustar los niveles de voltaje de las neuronas hasta que los patrones de activación dejen de ser erráticos.
*   **Normalización Genómica:** Implementar `GenomicNorm` nativo para evitar que la energía de los disparos (spikes) sature la red.

---

## 🛡️ Fase 4: La Cámara de Estabilidad (Crianza)
Una vez que el organismo detecta un patrón (ej. aprender a cerrar un paréntesis), ese rasgo debe protegerse.
*   **Anchor Locking:** Identificar el Top 0.1% de pesos que más contribuyen a la reducción de PPL y marcarlos como "Anclas Inamovibles".
*   **Crianza Progresiva:** Primero entrenar embeddings (vocabulario), luego lógica (atención) y finalmente memoria (FFN).

---

## 📈 KPIs de Coherencia
| Etapa | Comportamiento Esperado | Meta PPL |
| :--- | :--- | :--- |
| **Nacimiento** | Sopa de caracteres aleatorios. | > 10,000 |
| **Infancia** | Repetición de bigramas (ej. "el el", "de de"). | ~500 |
| **Juventud** | Formación de palabras cortas y gramática básica. | < 50 |
| **Madurez** | Generación de frases con sentido lógico. | < 2.0 |

---
*Estado: Pendiente de ejecución de Fase 1 (Creación del Embrión).*
