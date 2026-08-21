# 🧬 El Modelo GAJE: Arquitectura, Capacidades Cognitivas y Límites de Inferencia

> **Versión:** v1.6.0-alpha (Silver Adult)
> **Fecha:** 20 de agosto de 2026
> **Ubicación:** `docs/guides/GAJE_MODEL_CAPABILITIES_AND_LIMITS.md`
> **Componente:** Artefactos de Modelo Comprimido (`.gaje.flat`) y Motor de Inferencia

---

## 1. 📋 Identidad y Ficha Técnica de la Familia de Modelos GAJE

GAJE como modelo es una familia de **Organismos de Lenguaje Híbridos de Ultra-Alta Compresión (4-bits)** diseñados para inferencia en tiempo real en dispositivos locales (*Edge AI*):

```
                              FAMILIA DE MODELOS GAJE
                                         │
        ┌───────────────────┬────────────┴───────────┬───────────────────┐
        ▼                   ▼                        ▼                   ▼
  GAJE-MICRO (135M)   GAJE-CORE (0.5B)        GAJE-MID (1.5B)     GAJE-MAX (3B)
  ⚡ 140 MB RAM       ⚡ 450 MB RAM           ⚡ 1.2 GB RAM        ⚡ 2.2 GB RAM
  🚀 30 tok/s         🚀 20 tok/s             🚀 12 tok/s         🚀 3-4 tok/s
```

* **Arquitectura Base:** Transformer Decodificador Autoregresivo con **SwiGLU, RMSNorm y RoPE Rotacional**.
* **Formato Binario:** Formato plano **`.gaje.flat` v2** con mapeo de memoria Zero-Copy (`mmap`).
* **Precisión Mixta Híbrida v2:**
  * **Cuerpo del Transformer (Atención y FFN):** Cuantizado a **4-bits (Q4_0)** con 16 centroides optimizados.
  * **Capas Críticas de Entrada/Salida (`token_embd` / `lm_head`):** Mantenidas en **FP32 o Q8_0 (8-bits)** para evitar la corrupción del vocabulario en español, chino e inglés.
* **Memoria Persistente Opcional:** Conexión con índices binarios **Island Model (`.gmem`)** de recuperación submilisegundo ($750\text{ µs}$).

---

## 2. 🟢 Capacidades Comprobadas (¿Qué SÍ hace el Modelo?)

1. **Respuestas Factuales Directas (Paridad A/B 100%):**
   * Respuestas exactas y libres de alucinación en conocimiento general, capitales, biología y física básica (*París, Tokio, Madrid, punto de ebullición del agua a 100°C*).
2. **Sintaxis y Código Básico (Python / Shell):**
   * Generación correcta de funciones simples, bucles `for`, listas, condicionales y manejo de variables sin errores de indentación.
3. **Traducción y Bilingüismo Fluido:**
   * Traducción directa bidireccional entre **Español $\leftrightarrow$ Inglés** para oraciones y comandos cotidianos.
4. **Razonamiento Simbólico y Ecuaciones (Variante 3B):**
   * La variante **GAJE 3B** resuelve problemas algebraicos de dos incógnitas (edades, proporciones) derivando ecuaciones paso a paso en formato LaTeX.
5. **Inmunidad a Bucles Infinitos y Degeneración:**
   * Generación estable con sampler de temperatura ($T=0.2-0.4$) y penalización de repetición única ($1.1$), logrando **0.0% de respuestas degeneradas** en el modelo campeón `smollm2_4bit_quality`.
6. **Inferencia en Tiempo Real en CPU Comercial:**
   * Genera texto a **`30 tokens/segundo`** sobre procesadores AMD Ryzen o Intel convencionales sin requerir tarjeta gráfica (GPU).

---

## 3. 🔴 Límites Cognitivos y Físicos (¿Qué NO hace el Modelo?)

1. **NO resuelve Razonamiento Complejo en Variantes Micro (<1B):**
   * Los modelos de **135M y 0.5B** sufren colapso lógico en problemas abstractos largos o matemáticas con trampa. Para tareas que requieran deducción formal se debe usar la variante **GAJE 3B**.
2. **NO mantiene Coherencia en Textos Extensos (>500 tokens):**
   * Los modelos pequeños sufren de *Context Drift*: en narrativas o ensayos largos pierden el hilo conductor y tienden a divagar.
3. **NO almacena Conocimiento de Nicho Profundo sin RAG:**
   * La capacidad de almacenamiento paramétrico a 4-bits es acotada. Para detalles legislativos, médicos o corporativos específicos, requiere inyección de contexto mediante el módulo de persistencia `.gmem`.
4. **NO opera en 2-Bits Puros sin Deriva Semántica:**
   * A 2-bits la acumulación del error de cuantización a través de las capas lineales destruye los logits. La ruta de producción certificada es **Q4_0 + FP32/Q8_0**.

---

## 4. 📊 Matriz Comparativa de Posicionamiento

| Dimensión | GAJE-Micro (135M) | GAJE-Core (0.5B) | GAJE-Max (3B) | Modelos Nube (GPT-4) |
| :--- | :---: | :---: | :---: | :---: |
| **Throughput CPU** | 🏆 **`30 tok/s`** | 🟢 **`20 tok/s`** | 🟡 **`4 tok/s`** | 🔴 $0.5 - 2\text{ s}$ (Lag de red) |
| **Consumo de RAM** | 🏆 **`140 MB`** | 🟢 **`450 MB`** | 🟡 **`2.2 GB`** | 🔴 Servidores 80 GB |
| **Privacidad (100% Offline)** | 🏆 **Total** | 🏆 **Total** | 🏆 **Total** | 🔴 Datos viajan a la nube |
| **Factualidad Básica** | 🟢 Buena | 🟢 Muy Buena | 🟢 Exacta | 🏆 Enciclopédica |
| **Álgebra y Razonamiento** | 🔴 Limitado | 🟡 Básico | 🟢 **Correcto (LaTeX)** | 🏆 Experto |
| **Código Complejo / Refactors** | 🔴 No apto | 🔴 No apto | 🟡 Snippets medios | 🏆 Arquitectura completa |

---

## 5. 💡 Conclusión y Caso de Uso Estratégico

El modelo GAJE está optimizado para **Inteligencia de Borde (*Edge AI*) y Enjambres Multi-Agente Locales**:
* No compite contra modelos gigantescos en sabiduría enciclopédica abstracta.
* Es la **solución ideal para ser embebido en aplicaciones de escritorio, smartphones, dispositivos IoT y nodos de grafos asíncronos (Rust StateGraph)**, donde la velocidad, la privacidad total y el consumo casi nulo de memoria son indispensables.
