# 🧬 Protocolo de Entrenamiento Nativo GAJE-Born (v1.0)

Este documento define el plan para reemplazar el flujo de entrenamiento híbrido (Python/Rust) por una implementación 100% nativa en Rust, utilizando el motor neuromórfico.

## 1. Visión General
El objetivo es eliminar la dependencia de PyTorch y Python para el entrenamiento de modelos genómicos. Utilizaremos el **Nacimiento Neuromórfico** como el método principal de optimización de pesos de 2-bits.

## 2. Fases de Implementación

### Fase 1: Ingesta de Datos Nativa (Native Data Pipeline)
*   **Tokenizador Integrado:** Implementar un tokenizador BPE/Wordpiece nativo en Rust que alimente directamente el motor de spikes.
*   **Streamer de Dataset:** Un cargador de archivos eficiente que maneje datasets grandes sin cargar todo en RAM, convirtiendo texto en secuencias de impulsos eléctricos al vuelo.

### Fase 2: Núcleo de Optimización Bitwise
*   **Evolución Masiva:** Escalar el `SpikingEvolutionEngine` para manejar poblaciones de miles de organismos en paralelo usando `Rayon`.
*   **Mutación Adaptativa:** Implementar tasas de mutación que disminuyan a medida que el fitness se estabiliza (simulated annealing).

### Fase 3: Función de Pérdida Genómica (Genomic Loss)
*   **Resonancia Semántica:** Definir el fitness no solo por la frecuencia de disparos, sino por la precisión temporal (Timing) de los spikes en relación con la estructura del lenguaje.
*   **Penalización de Energía:** Incluir el consumo de eventos en la función de fitness para forzar al modelo a ser lo más disperso (sparse) y eficiente posible.

### Fase 4: Integración en `gaje-cli`
*   **Comando `--train`:** Añadir una interfaz de línea de comandos en `gaje-cli` que permita configurar capas, bloques, población y generaciones.
*   **Telemetría en Tiempo Real:** Visualización de la curva de fitness y la actividad de la red durante el "nacimiento" del modelo.

### Fase 5: Serialización y Exportación
*   **Escritor GAJE Nativo:** Un módulo para persistir los `Weights` empaquetados de 2-bits y los `Centroides` optimizados directamente en archivos `.gaje` o `.gguf`.

## 3. Ventajas del Enfoque Nativo
*   **Velocidad:** Reducción del tiempo de entrenamiento de horas a minutos al operar directamente sobre bits.
*   **Soberanía:** Eliminación total de dependencias externas (Python, Conda, PyTorch).
*   **Eficiencia:** Capacidad de entrenar micro-modelos directamente en el dispositivo móvil (On-device training).

---
*Este plan representa la hoja de ruta para la independencia total del ecosistema GAJE-Flow.*
