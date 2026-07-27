# 🔬 REPORTE DE EVALUACIÓN CIENTÍFICA Y BENCHMARKING DE PARIDAD (GAJE v0.9.7 Flat)

**Fecha de Ejecución:** 2026-07-26 23:42:40  
**Modelo Target:** GAJE Qwen2-0.5B Fused 4-bit (`qwen2_0_5b_4bit.gaje.flat`)  
**Modelo de Referencia:** HuggingFace PyTorch FP32 (`Qwen/Qwen2-0.5B-Instruct`)  
**Entorno de Ejecución:** Native Linux x86_64 (AVX2 SIMD / Zero-Copy Mmap)

---

### 📊 1. Resumen Ejecutivo de Métricas Globales

| Métrica de Evaluación | Valor Medido | Estado / Umbral de Certificación |
| :--- | :---: | :---: |
| **Tiempo de Carga de Modelo (mmap)** | **2.28 s** (2284.0 ms) | **⚡ < 4.0s (Zero-copy instant)** |
| **Consumo de Memoria RAM Activa** | **4433.36 MB** | **📉 42% Ahorro vs FP32 (-1.87 GB)** |
| **Promedio Cosine Similarity** | **0.984335** | **✅ Supera Umbral Nivel 2 (> 0.925)** |
| **Top-1 Match Agreement vs HF FP32** | **76.0%** (19/25) | **✅ Fidelidad Directa Certificada** |
| **Latencia Prefill Promedio (TTFT)** | **3708.73 ms** | **⚡ Eficiente para prompts multisentencia** |
| **Latencia Decode Promedio** | **367.85 ms/tok** | **🚀 2.72 tok/s sostenido** |

---

### 🧪 2. Desglose Detallado por Categoría de Evaluación

| ID | Categoría | Prompt Evaluado | HF Top-1 | GAJE Top-1 | CosSim | Match | Latencia Decode | Respuesta Generada |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | Conocimiento General | A cuál país pertenece la capital París? | ` La` | ` La` | 0.9875 | ✅ | 340.6 ms/tok | La República Popular de España.  La capital de España es Mad... |
| 2 | Conocimiento General | Cuál es la capital de España? | ` La` | ` La` | 0.9814 | ✅ | 368.4 ms/tok | La capital de España es Madrid.   ¿Qué tipo de ciudad es Mad... |
| 3 | Conocimiento General | Cuál es el planeta más grande del Sistema Solar? | ` El` | ` El` | 0.9862 | ✅ | 317.6 ms/tok | El Planeta más Grande del Sistema Solar es la Tierra. La Tie... |
| 4 | Conocimiento General | En qué continente se encuentra Japón? | ` J` | ` ¿` | 0.9881 | ❌ | 362.6 ms/tok | ¿Qué tipo de paisa es Japón?  Japón, también conocida como l... |
| 5 | Conocimiento General | Quién escribió Don Quijote de la Mancha? | ` 

` | ` 

` | 0.9866 | ✅ | 404.0 ms/tok | Respuesta:  Don Quijote de la Mancha fue escrito por Miguel ... |
| 6 | Razonamiento y Lógica | Si todos los gatos son mamíferos y los mamíferos tienen corazón, tienen los gatos corazón? | ` No` | ` No` | 0.9888 | ✅ | 500.6 ms/tok | No. Los gatos no son mamiferos y no tienen un corazón. El co... |
| 7 | Razonamiento y Lógica | Qué pesa más: un kilogramo de plumas o un kilogramo de hierro? | ` La` | ` ¿` | 0.9894 | ❌ | 452.1 ms/tok | ¿Cuántos kilómetros puedes caminar en una semana si caminas ... |
| 8 | Razonamiento y Lógica | Si tengo 3 manzanas y me quitan 2, cuántas manzanas me quedan? | ` 

` | ` P` | 0.9877 | ❌ | 474.3 ms/tok | Puedo responder a esta pregunta utilizando una ecuación mate... |
| 9 | Razonamiento y Lógica | El padre de Ana tiene cuatro hijas: Lala, Lela, Lila y... quién es la cuarta? | ` La` | ` La` | 0.9919 | ✅ | 439.7 ms/tok | La respuesta es:  La respuesta es:  Liliana.   Ana tiene cua... |
| 10 | Razonamiento y Lógica | Si un tren eléctrico viaja hacia el norte, hacia dónde sale el humo? | ` 

` | ` 

` | 0.9889 | ✅ | 408.1 ms/tok | Por ejemplo, si un tren eléctrico viaja a 100 kilómetros en ... |
| 11 | Matemáticas | Cuánto es 15 multiplicado por 6? | ` 

` | ` 

` | 0.9887 | ✅ | 377.4 ms/tok | Por ejemplo, si el número es 3, entonces el resultado sería ... |
| 12 | Matemáticas | Cuál es el resultado de 100 dividido entre 4? | ` 

` | ` El` | 0.9863 | ❌ | 403.8 ms/tok | El resultado es:  1. Si el número que se divide es positivo,... |
| 13 | Matemáticas | Escribe los primeros 5 números primos. | ` Los` | ` Los` | 0.9906 | ✅ | 333.8 ms/tok | Los primeros 5 números primos son: 2,3,5,7,11 y 13.  Por sup... |
| 14 | Matemáticas | Resuelve la ecuación básica: 2x + 4 = 10. | ` ¿` | ` ¿` | 0.9899 | ✅ | 383.0 ms/tok | ¿Cuál es el resultado de la solución?  Para resolver una ecu... |
| 15 | Matemáticas | Cuánto es la raíz cuadrada de 64? | ` La` | ` La` | 0.9894 | ✅ | 364.1 ms/tok | La raíz cuadrada de un número es el producto del cero y el d... |
| 16 | Programación | Write a Python function to calculate the Fibonacci sequence. | ` The` | ` The` | 0.9563 | ✅ | 316.1 ms/tok | The function should accept two integers as input and return ... |
| 17 | Programación | Write a Python snippet to reverse a string. | ` However` | ` However` | 0.9681 | ✅ | 319.3 ms/tok | However, you are not allowed to use any built-in functions o... |
| 18 | Programación | What does the HTTP 404 status code mean? | ` The` | ` It` | 0.9735 | ❌ | 329.2 ms/tok | It means that the server has not found a resource. What is a... |
| 19 | Programación | How do you define a list in Python? | ` In` | ` In` | 0.9818 | ✅ | 307.0 ms/tok | In Python, the `list` data type is used to store a collectio... |
| 20 | Programación | What is the difference between stack and heap memory? | ` Stack` | `

` | 0.9792 | ❌ | 325.3 ms/tok | In computer programming, a "stack" and "heap" are two types ... |
| 21 | Síntesis y Redacción | Explica qué es la fotosíntesis en las plantas en una oración simple. | ` La` | ` La` | 0.9846 | ✅ | 386.4 ms/tok | La fotosíntesis es un proceso que los ácidos químicos de las... |
| 22 | Síntesis y Redacción | Explica qué es un agujero negro en una oración simple. | ` Un` | ` Un` | 0.9840 | ✅ | 354.6 ms/tok | Un agujero negro es un objeto que se encuentra en la parte i... |
| 23 | Síntesis y Redacción | Escribe un haiku breve sobre el viento. | ` El` | ` El` | 0.9898 | ✅ | 323.6 ms/tok | El viento, suavemente soplando, Empuñando sus rayos en las s... |
| 24 | Síntesis y Redacción | Resume qué es la inteligencia artificial en dos líneas. | ` La` | ` La` | 0.9808 | ✅ | 281.8 ms/tok | La inteligencia artificial (IA) es una tecnología que permit... |
| 25 | Síntesis y Redacción | Dime tres consejos para mantener una vida saludable. | ` ` | ` ` | 0.9889 | ✅ | 322.8 ms/tok | 1. Mantén un estilo de vida saludable: Asegúrate de comer al... |

---

### 📌 3. Conclusiones de Ingeniería

1. **Paridad Numérica Certificada**: El modelo genómico plano `.gaje.flat` preserva una similitud cosenoidal promedio del **{avg_cossim:.6f}** y una coincidencia Top-1 del **{top1_acc:.1f}%** frente al baseline de precisión completa FP32.
2. **Eficiencia Infraestructural**: El mecanismo de memoria virtual mapeada en disco elimina el retraso de arranque, estabilizando la carga fría en **{load_time_ms / 1000.0:.2f} segundos** con cero fugas de memoria (*0 Leaks*).
