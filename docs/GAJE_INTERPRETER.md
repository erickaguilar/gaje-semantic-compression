# 🧬 MANIFIESTO DEL INTÉRPRETE GAJE: El Eslabón Semántico

## 1. Visión General
El **Intérprete GAJE** es el componente crítico que resuelve la desconexión entre el almacenamiento genómico (2-bits, alta densidad, baja precisión) y la inferencia neuronal (32-bits, alta precisión, alto costo). Su función no es simplemente "descomprimir" datos, sino **interpretar intenciones semánticas** a través de una fase de sincronización biológica.

Mientras que un RAG tradicional busca coincidencias literales, un **RAG/LLM impulsado por el Intérprete GAJE** busca **trayectorias de energía**, permitiendo que un ligero cambio en un codón no desvíe el pensamiento del modelo, sino que lo mantenga en el entorno semántico correcto.

## 2. Los 4 Pilares del Intérprete

### A. Sincronización de Fase (Phase Alignment)
El Intérprete debe alinear las "manecillas del reloj" de cada cabeza de atención. Si los pesos están en 2-bits, el Intérprete aplica una rotación de fase (RoPE) que compensa el ruido de cuantización, asegurando que el vector de consulta siempre encuentre su "rama de ADN" correspondiente.
*   **Hito:** Implementación de RoPE Split vs Interleaved sincronizado con los pesos físicos.

### B. Mapeo de Codones (Dequantization Mapping)
En lugar de tratar cada bit como un número, el Intérprete lee **bloques de 4 bits (2 pesos)**. Sabe que ciertas combinaciones de ADN representan "silencio semántico" y otras representan "activación crítica". Utiliza **Centroides Max-Lloyd** como un diccionario de traducción instantánea.
*   **Hito:** Motor de búsqueda Asymmetric Distance Computation (ADC) integrado en el Kernel de Rust.

### C. Homeostasis Energética (Metabolic Clamping)
El Intérprete actúa como un regulador de voltaje. Impide que el error de un bloque de 2-bits se multiplique exponencialmente al siguiente. Si la señal comprimida intenta "explotar", el Intérprete la devuelve a su estado de equilibrio (Norm-Preservation).
*   **Hito:** Capas Invariantes de Norma (Norm-Invariant Layers).

### D. Resiliencia Sináptica (Recursive Error Correction)
El Intérprete mantiene una "memoria de error". Sabe que el almacenamiento GAJE tiene variaciones y aplica un factor de corrección dinámico basado en la tipología de la base de datos para mantener la coherencia del "ente más grande" (el organismo).

## 3. Estabilidad Bio-Inspirada: La Trifecta de Resiliencia

Para asegurar que el modelo no se desvíe en el "espacio de ADN", el Intérprete implementa tres mecanismos de defensa avanzados:

### I. Residual Quantization (RQ): La Capa Epigenética
En lugar de un mapeo rígido a un solo centroide, el Intérprete puede gestionar un segundo nivel de "error" (el residuo). 
*   **Visión:** El ADN da la instrucción base, y el residuo (ARN) ajusta los detalles finos. Esto permite estabilizar la PPL drásticamente al capturar pesos atípicos (outliers) que de otro modo causarían alucinaciones.

### II. Temperature Scaling en K-Means: Fluidez Genómica
Implementación de un factor de temperatura en la asignación de centroides durante la fase de entrenamiento. 
*   **Visión:** Evita asignaciones "rígidas". Una distribución más suave de los tokens de ADN ayuda a que la transición energética entre capas sea menos errática, reduciendo el ruido secuencial del LLM.

### III. Gray Code Mapping: Resiliencia Mutacional
Asegura que errores de un solo bit en la base de datos resulten en cambios mínimos en el valor de energía.
*   **Visión:** Si un bit cambia por ruido térmico o error de lectura, el valor resultante se mantiene en un "entorno semántico" cercano (centroide vecino) en lugar de saltar a un estado opuesto. Esto evita la explosión de PPL ante mutaciones accidentales.

## 4. Adaptatividad Entrópica: Modo de Alta Fidelidad Dinámico

El Protocolo GAJE evoluciona de una compresión estática a un metabolismo dinámico basado en la complejidad de la información:

### I. Segmentación por Entropía de Shannon
Se implementará un sensor de entropía en el Intérprete para clasificar los flujos de datos. 
*   **Baja Entropía (Datos Redundantes):** Se procesan estrictamente en 2 bits.
*   **Alta Entropía (Datos Complejos):** Se identifican como "Zonas de Estrés Semántico" que pueden romper el indicador de PPL.

### II. High Fidelity Mode (Precision Switching)
Cuando el Intérprete detecta una zona de alta entropía o un pico de PPL local, el sistema activa automáticamente un mapeo de **3 o 4 bits** para esas dimensiones específicas.
*   **Visión:** Es el equivalente biológico a la "Atención Concentrada". El organismo ahorra energía en lo trivial y aumenta la resolución en lo crítico.

---

## 5. Hoja de Ruta: Hitos de Implementación

### Hito 1: Estabilización del Esqueleto (F32 Baseline)
*   **Objetivo:** Lograr Perplexity < 20 en SmolLM2-135M usando el motor sincronizado sin compresión.
*   **Éxito:** El modelo debe predecir "Once upon a time" con alta probabilidad.
*   **Estado:** ⚠️ En progreso (Alineación de GQA y RoPE).

### Hito 2: Implementación del Kernel de Traducción (2-Bit ADC)
*   **Objetivo:** Mover el MatMul de 2-bits a Rust con des-permutación en tiempo real.
*   **Éxito:** Reducción de latencia en 5x respecto a F32.

### Hito 3: Calibración Metabólica del Intérprete
*   **Objetivo:** Implementar Clamping dinámico por capa para absorber el ruido de cuantización.

### Hito 4: Fusión de RAG y Pensamiento (Genomic Memory)
*   **Objetivo:** Unificación total del almacenamiento (RAG) y el cálculo (LLM) bajo el mismo lenguaje genómico.

---

## 4. Guía para la Próxima Sesión
Al reiniciar, el foco debe ser **"Calibrar la Fase"**. El código del motor ya soporta los 30 bloques de SmolLM2; lo que resta es asegurar que la des-permutación de los pesos Query/Key coincida exactamente con la rotación RoPE aplicada.

**Instrucción de Arranque Recomendada:**
> "Cargar el motor GenomicLLM con des-permutación de pesos y probar la predicción de 'Once upon a' hasta lograr que el target ' time' sea el más probable."
