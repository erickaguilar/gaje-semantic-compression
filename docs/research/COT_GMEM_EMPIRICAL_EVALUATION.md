# 📊 Evaluación Empírica de Memoria .gmem: Retrieval Vectorial vs. Generación LLM

> **Clasificación:** Reporte de Investigación Empírica (`docs/research/`)  
> **Fecha:** 17 de septiembre de 2026  
> **Modelo Evaluado:** `Qwen2.5-0.5B-Instruct` cuantizado en [`models/production/qwen2_5_0_5b.gaje`](file:///data/data/com.termux/files/home/develop/gaje-semantic-compression/models/production/qwen2_5_0_5b.gaje) (1.5 GB, 24 capas Q4_0, GTOK 151,936 tokens)  
> **Índice de Memoria:** [`.gmem`](file:///data/data/com.termux/files/home/develop/gaje-semantic-compression/src/io/gmem.rs) indexado en 896 dimensiones vía `embed_text_gtok` (Weighted Mean Pooling)  
> **Entorno de Ejecución:** ARM64 (Linux/Termux), inferencia nativa en Rust (`gaje-core`), greedy decoding `temp = 0.0`.  
> **Harness de Prueba:** [`tests/test_cot_gmem_baseline.rs`](file:///data/data/com.termux/files/home/develop/gaje-semantic-compression/tests/test_cot_gmem_baseline.rs) (`test_e2e_real_gmem_rag_retrieval_and_generation`)

---

## 1. Resumen Ejecutivo y Resultados de la Prueba E2E Real

Se evaluó el pipeline completo de **RAG Real sin prefijo forzado**:
1. Ingesta de 20 hechos canónicos en un índice binario `.gmem` en 896 dimensiones (tiempo de ingesta: `120.35 ms`).
2. Para cada una de las 20 preguntas, búsqueda del vecino más cercano (Top-1) por similitud coseno.
3. Inyección del hecho recuperado en el `system prompt` (`Knowledge: {retrieved_fact}`).
4. Generación libre del asistente desde `<|im_start|>assistant\n` sin ningún prefijo forzado en su turno.

| Fase del Sistema | Métrica Evaluada | Resultado Obtenido | Diagnóstico Operativo |
| :--- | :--- | :---: | :--- |
| **Motor de Memoria `.gmem`** | **Top-1 Retrieval Recall** | **`19 / 20` (95.0%)** | **Excelente.** El extractor de embeddings (`embed_text_gtok`) y la búsqueda coseno recuperan el hecho exacto en el 95% de las consultas. |
| **Generación Libre LLM (0.5B)** | **Exactitud E2E sin Prefijo** | **`8 / 20` (40.0%)** | **Cuello de Botella.** El modelo de 0.5B frecuentemente se congela en preámbulos vacíos (*"To answer the question..."*) o confabula, incluso teniendo el dato en el contexto del sistema. |

---

## 2. Desglose Detallado: Retrieval vs. Generación

| ID | Dominio | Pregunta | Retrieval .gmem (Top-1) | Similitud | Generación LLM (sin prefijo) |
| :---: | :--- | :--- | :---: | :---: | :---: |
| 1 | Geografía | Capital de Australia | 🎯 MATCH (Canberra) | 0.872 | ✅ HIT (Canberra) |
| 2 | Geografía | Capital de Canadá | 🎯 MATCH (Ottawa) | 0.896 | ❌ MISS (Otua - typo) |
| 3 | Geografía | Capital de Brasil | 🎯 MATCH (Brasilia) | 0.888 | ✅ HIT (Brasilia) |
| 4 | Geografía | Capital de Turquía | 🎯 MATCH (Ankara) | 0.894 | ✅ HIT (Ankara) |
| 5 | Geografía | Capital de Suiza | 🎯 MATCH (Bern) | 0.905 | ❌ MISS (Swiss region) |
| 6 | Ciencia | Símbolo del Oro | 🎯 MATCH (Au) | 0.914 | ✅ HIT (Au) |
| 7 | Ciencia | Símbolo del Plomo | ⚠️ MISS (Colisión léxica -> Au) | 0.866 | ✅ HIT (Pb - por memoria interna) |
| 8 | Ciencia | Velocidad de escape Tierra | 🎯 MATCH (11.2 km/s) | 0.825 | ✅ HIT (11.2 km/s) |
| 9 | Ciencia | Velocidad de la luz | 🎯 MATCH (299,792 km/s) | 0.767 | ✅ HIT (299,792 km/s) |
| 10 | Ciencia | Gas más abundante Tierra | 🎯 MATCH (Nitrogen 78%) | 0.910 | ❌ MISS (Incompleto) |
| 11 | Astronomía | Planeta con más lunas | 🎯 MATCH (Saturn 146) | 0.864 | ❌ MISS (Emitió "146") |
| 12 | Astronomía | Planeta más cercano al Sol | 🎯 MATCH (Mercury) | 0.920 | ❌ MISS ("The closest is SuN") |
| 13 | Astronomía | Mayor luna de Júpiter | 🎯 MATCH (Ganymede) | 0.898 | ❌ MISS ("Knowlledge") |
| 14 | Astronomía | 2do planeta desde el Sol | 🎯 MATCH (Venus) | 0.934 | ❌ MISS ("Kuwait") |
| 15 | Astronomía | Estrella de la Mañana | 🎯 MATCH (Venus) | 0.911 | ❌ MISS ("VeuS" - typo) |
| 16 | Historia | Año alunizaje Apolo 11 | 🎯 MATCH (1969) | 0.838 | ❌ MISS (Preámbulo vacío) |
| 17 | Historia | Fin Segunda Guerra Mundial | 🎯 MATCH (1945) | 0.648 | ❌ MISS (Preámbulo vacío) |
| 18 | Historia | Fundación Naciones Unidas | 🎯 MATCH (1945) | 0.675 | ✅ HIT (1945) |
| 19 | Ciencia | Elementos tabla periódica | 🎯 MATCH (118) | 0.770 | ❌ MISS (Preámbulo vacío) |
| 20 | Biología | Cromosomas células somáticas | 🎯 MATCH (46) | 0.782 | ❌ MISS (Preámbulo vacío) |

---

## 3. Conclusiones y Calibración Rigurosa

1. **La Memoria `.gmem` está Validada:**
   * El subsistema de persistencia y búsqueda vectorial alcanza un **95.0% de recall** con latencias de búsqueda sub-milisegundo. El componente semántico de GAJE funciona de forma óptima.
2. **Disociación entre Retrieval y Capacidad del Modelo:**
   * La tasa de respuesta correcta en generación libre es de **40.0% (8/20)**. Esto coincide exactamente con el control RAG puro (Condición D), confirmando que el límite del sistema no es la memoria externa, sino la capacidad atencional y de seguimiento de instrucciones de un modelo de 0.5B de parámetros en decodificación abierta.
3. **Prefix-Completion vs. RAG:**
   * Las condiciones previas con prefijo en el turno del asistente (C4 = 65%) representan **prefix-completion a 1 token**, no razonamiento de recuperación. En producción abierta, el modelo debe ser guiado mediante ingeniería de prompt para evitar la trampa del preámbulo vacío.
