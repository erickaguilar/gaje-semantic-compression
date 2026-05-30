# 🧬 Guía de Crianza (Breeding) y Nacimiento (Born) de Organismos GAJE

Esta guía detalla los procedimientos para generar nuevos modelos genómicos desde cero, ya sea mediante evolución biológica simulada (Breeding) o mediante entrenamiento híbrido nativo (Born).

---

## 1. Crianza Evolutiva (Path Integral Breeding)
La crianza utiliza el motor de **Selección Natural de Monte Carlo**. No requiere gradientes ni PyTorch; se basa en mutaciones aleatorias del ADN y la supervivencia del más apto en paralelo.

### Requisitos
- **Entorno:** Rust Toolchain instalado.
- **Hardware:** CPU con soporte SIMD (NEON en Android, AVX2 en PC).
- **Dataset:** No requiere archivos externos para pruebas de memoria secuencial.

### Ejecución
Para ver a un micro-organismo memorizar una secuencia en milisegundos:

```bash
# Compilar y ejecutar el ejemplo de memoria secuencial
cargo run --release --bin hola-mundo-evolution
```

### Cómo funciona
1. **Población:** Se crean 100 clones (mutantes) del organismo base.
2. **Mutación:** Cada clon recibe alteraciones aleatorias en su ADN de 2 bits.
3. **Simulación:** Todos los clones procesan la secuencia en paralelo usando `Rayon`.
4. **Selección:** El clon con la menor perplejidad (mayor probabilidad de acierto) se convierte en el nuevo ancestro.

---

## 2. Nacimiento Genómico (Born-Genomic Training)
El "nacimiento" es un proceso de entrenamiento formal que utiliza **Autograd Híbrido**. El motor de Rust realiza la inferencia real en 2 bits, mientras que Python calcula el error y refina los centroides.

### Requisitos
- **Archivos de Datos:**
  - `tokenizer.json`: En la raíz del proyecto.
  - `dataset_entrenamiento.txt`: Texto plano para el aprendizaje.
- **Dependencias:** `torch`, `numpy`, `tokenizers` (instalables vía `pip install .[dev]`).

### Procedimiento de Inicio
1. **Preparar el Dataset:**
   Asegúrate de que `dataset_entrenamiento.txt` contenga el conocimiento que deseas que el modelo adquiera.

2. **Inicializar el Organismo:**
   Crea un modelo vacío con pesos aleatorios y estructura definida:
   ```bash
   # Ejemplo de inicialización vía script (ajustar parámetros según necesidad)
   python scripts/train_born_genomic.py --init --name "MiniGaje-v1"
   ```

3. **Bucle de Entrenamiento:**
   ```bash
   python scripts/train_born_genomic.py --epochs 10 --lr 0.001
   ```

### Parámetros Críticos en `ArchitectureConfig`
- `n_embd`: 256 o 512 (mantener bajo para dispositivos móviles).
- `n_blocks`: 4 a 8 capas.
- `block_size`: 32 (estándar del protocolo GAJE).

---

## 3. Comparativa: ¿Qué modalidad elegir?

| Característica | Crianza (Breeding) | Nacimiento (Born) |
| :--- | :--- | :--- |
| **Velocidad** | Instantánea (<1s) | Lenta (Minutos/Horas) |
| **Escalabilidad** | Micro-modelos (<1M params) | Modelos Edge (10M - 500M) |
| **Precisión** | Memorización exacta | Generalización del lenguaje |
| **Dependencias** | 100% Rust | Rust + Python + PyTorch |
| **Uso Ideal** | RNNs, Memoria de Corto Plazo | LLMs de propósito general |

---

## 4. Resolución de Problemas (Troubleshooting)

- **Error: `Database already open`:** El sistema de protección de `redb` ha bloqueado el archivo. Asegúrate de no tener otra instancia de `gaje` corriendo o borra el archivo `.gaje` si es una prueba fallida.
- **Pérdida de Coherencia (Drift):** Si el modelo empieza a arrojar basura, reduce el `Learning Rate` (LR) o aumenta el número de `Anclas` (Anchors) en la configuración.
- **OOM (Out of Memory):** En Termux, limita el `batch_size` a 1 o 2 y reduce la longitud de contexto a 128.

## 5. Estrategias de Dataset y Especialización (FAQ)

### ¿Cuál es el mejor formato para el dataset?
Para el nacimiento genómico (Born), existen dos enfoques recomendados:

1.  **Texto Plano (.txt):** Ideal para conocimiento general o literatura. El modelo simplemente aprende la probabilidad del siguiente token en un flujo continuo.
2.  **JSONL (Recomendado para Instrucciones/Código):** Cada línea es un objeto JSON con campos `instruction` y `response`.
    - *Ventaja:* Permite estructurar el pensamiento del modelo.
    - *Nota:* El cargador actual en `python/gaje/processing/pipeline.py` debe estar configurado para parsear el JSON antes de enviarlo al motor de Rust.

### ¿Puede nacer como experto en Programación o Software específico?
**Sí, y es la mayor ventaja de GAJE.** A diferencia de los modelos tradicionales que se cuantizan *después* de ser generales, un GAJE puede ser entrenado desde el "kilómetro cero" en un dominio específico.

-   **Nacimiento Especializado:** Si el `dataset_entrenamiento.txt` consiste en un 70% de código fuente (ej. Rust, Python) y un 30% de documentación técnica, el modelo desarrollará centroides genómicos optimizados para la sintaxis de programación, logrando una precisión en 2 bits que un modelo general cuantizado nunca alcanzaría.
-   **Software Específico:** Puedes "alimentar" al organismo con el código fuente completo de un proyecto (ej. el kernel de Linux o este mismo repositorio). El modelo nacerá con una "intuición" nativa sobre esa arquitectura.

### ¿Ámbito General o Específico?
-   **Ámbito General:** Requiere datasets masivos (GBs de texto) y más tiempo de entrenamiento. Difícil de lograr puramente en un dispositivo móvil.
-   **Ámbito Específico (Recomendado):** Es el punto fuerte de GAJE. Crear un "micro-experto" de 30MB que sepa todo sobre una API específica o un lenguaje es mucho más eficiente y útil para aplicaciones de *Edge Computing*.

### Consejo Pro: El Mix de Datos
Para un experto en programación que no olvide cómo hablar, usa un mix de:
- **40%** Código fuente limpio.
- **40%** Documentación técnica (Markdown).
- **20%** Conversación general (para mantener la gramática).
