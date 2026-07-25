# Hoja de Ruta: Base de Datos Semántica Genómica y Registro Evolutivo

## 🎯 Visión General
Para soportar el aprendizaje continuo (Continuous Learning) en dispositivos Edge sin depender de servidores pesados (como MongoDB o PostgreSQL), GAJE implementará su propia **Base de Datos Semántica Genómica**. Esta base de datos será un sistema híbrido, 100% nativo en Rust, que combinará el actual motor de búsqueda vectorial (`GajeIndex`) con un almacenamiento clave-valor (Key-Value) embebido para registrar la evolución temporal del "metabolismo" del modelo.

## 🧬 Concepto Clave: El "Epigenetic Log" (Registro Epigenético)
Al igual que en la biología la epigenética registra los cambios producidos por el ambiente a lo largo del tiempo sin alterar el código genético base, nuestro sistema almacenará:
1. **ADN Base (Inmutable/Lento):** Las hebras de 2 bits pre-entrenadas.
2. **Log de Mutaciones (Dinámico):** El historial de cómo los centroides y los "Anchors" (4-bit/6-bit) se han refinado localmente (usando `refine_ffn`).
3. **Métricas de Homeostasis:** Series de tiempo del *Drift* de activaciones, pérdida (Loss) y mapas de calor metabólico generados por el `SignalToNoiseBalancer`.

---

## 🗺️ Fases de Implementación

### Fase 1: Selección e Integración del Motor K-V en Rust
El primer paso es abandonar la idea de exportar múltiples archivos `.npy` sueltos y adoptar un motor de almacenamiento embebido en Rust.
* **Candidatos Tecnológicos:**
  * **`redb`**: Base de datos embebida escrita en Rust puro. Alta fiabilidad, ACID, soporta un único archivo en disco. Es ideal para evitar las vulnerabilidades de memoria de C/C++.
  * **`sled`**: Otra excelente alternativa nativa en Rust, orientada a alto rendimiento.
* **Acción:** Incluir el motor K-V en el `src/loader.rs` y definir la estructura inicial de un archivo `.gaje` unificado que contenga "tablas" (Trees) para los pesos, centroides y metadatos.

### Fase 2: Diseño del Esquema Evolutivo (Epigenetic Log)
Definir cómo se guardará el histórico en el almacenamiento Key-Value.
* **Tabla `snapshots`**:
  * Key: `Timestamp (u64)`
  * Value: `[Métricas de entropía, Porcentaje de hebras 2/4/6-bit]`
* **Tabla `mutations_log`**:
  * Key: `Timestamp + LayerID`
  * Value: `Delta de los centroides (Float32 array)`
* **Tabla `dna_strands`**:
  * Key: `LayerID`
  * Value: `Struct-of-Arrays (SoA) binario (base, epi, tri)`

### Fase 3: Integración Híbrida (Indexación + Persistencia)
Conectar el almacenamiento persistente con el motor en memoria.
* **Acción:** Cuando el usuario inicie una sesión (`GenomicLLM::load`), el sistema leerá la tabla `dna_strands` usando Mmap (Memory Mapped Files) desde la base de datos `redb` directamente hacia las estructuras SoA de Rust para obtener *Zero-Copy Loading*.
* **Actualización Asíncrona:** Cuando el optimizador llame a `refine_ffn`, el nuevo estado de los centroides se escribirá asíncronamente en la tabla `mutations_log` sin bloquear la inferencia.

### Fase 4: Habilidades de "Viaje en el Tiempo" (Time-Travel / Rollback)
Aprovechar el registro epigenético para darle al usuario control sobre el conocimiento del modelo.
* **Acción:** Implementar métodos en Python/Rust que permitan consultar el estado anterior del modelo.
* **Ejemplos de API Futura:**
  * `model.rollback(to_date="2026-05-01")`: Revierte los centroides eliminando las mutaciones recientes si el modelo ha sufrido *catastrophic forgetting* (olvido catastrófico).
  * `model.get_evolution_history()`: Retorna un JSON o DataFrame con el progreso del MSE/Drift a lo largo de los días para renderizar gráficos de rendimiento.

---

## 📈 Impacto Arquitectónico
Esta hoja de ruta solidifica a GAJE no solo como un formato de compresión, sino como un **motor de base de datos vectorial adaptativo**. Resuelve el problema de la gestión de memoria a largo plazo en dispositivos móviles al mantener todo en un solo archivo binario (ej. `mi_modelo.gaje`), eliminando la necesidad de dependencias externas complejas.
