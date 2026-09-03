# 🧠 Plan de Implementación: Semantic RAG Nativo (2-bit Memory)

Este documento detalla la arquitectura y las fases para dotar al motor GAJE de una **Memoria Semántica de Largo Plazo** persistente y eficiente, utilizando compresión genómica de 2 bits y la base de datos nativa `redb`.

---

## 1. Visión General
El objetivo es que cada interacción, pensamiento o dato procesado por el modelo se convierta en una "experiencia" almacenable. A diferencia de un RAG tradicional que guarda vectores en `f32` (4 bytes por dimensión), GAJE RAG utiliza **ADN de 2 bits** (16x más pequeño), permitiendo indexar millones de recuerdos en dispositivos móviles.

---

## 2. Arquitectura de Almacenamiento: `redb`
Utilizaremos `redb` como motor de almacenamiento por su soberanía en Rust, seguridad de hilos y soporte para archivos mapeados en memoria (Mmap).

### Estructura de la Base de Datos (`experiences.gaje`):
- **Table: `genomic_index`**
    - `Key`: `u64` (Timestamp / ID de Evento).
    - `Value`: `(Vec<u8>, String)` -> (DNA de 2 bits del embedding, Texto/Metadato).
- **Table: `epigenetic_registry`**
    - Almacena refinamientos locales (mutaciones) que el modelo ha sufrido tras esa experiencia específica.

---

## 3. Protocolo de Recuperación: ADC (Asymmetric Distance Computation)
Para evitar de-cuantizar toda la base de datos durante una búsqueda (lo cual sería lento y costoso en batería), implementaremos **Búsqueda Asimétrica**:

1.  **Consulta (Query):** El input del usuario se mantiene en `f32`.
2.  **Cómputo:** Se calcula la distancia directamente contra el ADN de 2 bits almacenado mediante una tabla de búsqueda (Look-up Table) precargada con los centroides.
3.  **Rendimiento:** Objetivo de >200,000 comparaciones por segundo en un solo núcleo `Little`.

---

## 4. Fases de Implementación

### Fase 1: Capa de Persistencia (Infraestructura)
- [ ] Implementar el módulo `src/io/database.rs` para gestionar la apertura/cierre de `redb`.
- [ ] Definir el esquema de serialización de "Experiencias Genómicas".

### Fase 2: Ingestión y Cuantización al Vuelo
- [ ] Integrar el motor de cuantización 2-bit en el proceso de guardado.
- [ ] Crear un hook en el `NeuromorphicScheduler` para persistir el estado de la capa de salida tras una generación exitosa.

### Fase 3: Motor de Búsqueda Semántica Nativo
- [ ] Implementar el kernel ADC en Rust para búsqueda vectorial sobre ADN de 2 bits.
- [ ] Añadir soporte para **Filtrado Temporal** (recuperar memorias recientes vs. memorias antiguas con mayor relevancia).

### Fase 4: Integración de Contexto (Context Injection)
- [ ] Crear el mecanismo de "Inyección de Memoria": Las experiencias recuperadas se inyectan como spikes de baja intensidad en la `TimingWheel` para influir en la generación actual sin dominarla.

---

## 5. Mitigación de Desafíos
- **Olvido Catastrófico:** Uso del `epigenetic_registry` para restaurar estados previos del modelo si un nuevo aprendizaje corrompe la coherencia.
- **Fragmentación:** Implementar un proceso de "Compactación de Memoria" (Compaction) que se ejecute solo cuando el dispositivo está cargando (Power-aware maintenance).

---
*Este plan es el documento de referencia único para la implementación de la memoria de largo plazo en GAJE.*
