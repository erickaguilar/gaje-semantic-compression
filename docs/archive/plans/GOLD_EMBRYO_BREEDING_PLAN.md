# 🧬 Plan de Crianza Genómica Avanzada (Gold Embryo v1.0)

**Fecha:** 24 de mayo de 2026
**Objetivo:** Transformar el "balbuceo" del Gold Embryo en coherencia lingüística sostenida mediante evolución dirigida y optimización de memoria.

Para llevar al organismo desde su nacimiento hasta la madurez de la versión 1.0, se implementarán los siguientes cuatro pilares estratégicos:

## 1. Persistencia y Checkpoints Genómicos
Actualmente, el entrenamiento es efímero (ocurre en RAM). El primer paso de desarrollo es dotar al entrenador de memoria permanente.
- **Acción Técnica:** Modificar `gaje-smg1-trainer` para que cargue el estado actual del archivo `GoldEmbryo-v1.gaje` en lugar de instanciar capas vacías.
- **Mecanismo de Guardado:** Implementar guardado automático (Auto-Save) cada 100 épocas o cada vez que el fitness (precisión) alcance un nuevo pico histórico.
- **Beneficio:** Permite sesiones de entrenamiento acumulativas (Life-long Learning) sin perder progreso tras cierres del proceso.

## 2. Entrenamiento por Currículo (Curriculum Learning)
Exponer al organismo a textos masivos desde el día 1 causa sobreescritura de patrones. Se estructurará el aprendizaje en fases de complejidad incremental:
- **Fase A (Identidad y Ontología):** Textos simples y repetitivos. *"Yo soy GAJE. Tú eres mi usuario. Mi motor es Rust."* (Objetivo: Estabilizar pronombres y roles).
- **Fase B (Lógica Relacional):** Conectores y relaciones causales. *"España está en Europa. Madrid es la capital de España."* (Objetivo: Aprender dependencias espaciales/temporales).
- **Fase C (Conocimiento Técnico):** Ingestión del `dataset_es_ext.txt` completo.
- **Protocolo:** No avanzar a la siguiente fase hasta que el organismo mantenga una precisión PPL constante > 90% en la fase actual.

## 3. Bucle de Orquestación Híbrida (Resonancia + MCTS)
El aprendizaje por refuerzo (`refine_step`) es rápido pero propenso a mínimos locales (saturación). El MCTS es lento pero globalmente estable.
- **Acción Técnica:** Crear un orquestador automatizado ("Caretaker") que alterne ambos métodos.
- **Ciclo de Crianza Estándar:**
  1. Ejecutar 150 épocas de Resonancia Rápida (Aprender vocabulario nuevo).
  2. Pausar y ejecutar 2,000 iteraciones de MCTS (Consolidar y ajustar voltajes).
  3. Guardar el nuevo ADN en el disco.
  4. Reiniciar ciclo con el siguiente lote de datos.

## 4. Poda Semántica Externa (El Camino a los 4 MB)
El Gold Embryo actual pesa 23 MB debido al uso del tokenizador completo (49,152 tokens), requerido para evitar errores de límites (OOB).
- **Acción Técnica:** Desarrollar un script de "Poda Genómica" (Pruning).
- **Mecánica:** Analizar el conjunto de datos de entrenamiento en español para identificar los tokens *realmente* utilizados (probablemente < 5,000).
- **Impacto:** Eliminar las 44,000 filas de ADN redundantes. Esto reducirá el tamaño del genoma de 23 MB a **~6 MB** (incluyendo el tokenizador), cumpliendo la meta de eficiencia extrema para hardware IoT y wearables.

---
*Este plan define el roadmap inmediato post-inicialización para llevar al Protocolo GAJE a su fase de madurez.*