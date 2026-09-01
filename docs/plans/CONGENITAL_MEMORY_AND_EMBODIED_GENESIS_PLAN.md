# 🎯 Plan de Implementación: Nacimiento con Memoria Congénita (`.gmem`) y Co-Evolución Corteza-Hipocampo

> **Fecha:** 1 de Septiembre de 2026  
> **Versión:** `GAJE Helix v1.7.0`  
> **Estado:** `APROBADO PARA IMPLEMENTACIÓN`  
> **Objetivo:** Integrar la memoria asociativa toroidal `.gmem` v2 directamente en el protocolo de génesis (`gaje-cli birth --with-memory`) y en el bucle de entrenamiento STE para lograr retención fáctica sin aumentar los pesos neuronales.

---

## 1. Visión y Arquitectura del Sistema

El organismo genómico `max_laser` ($D=384, L=12$) nacerá como un **sistema cognitivo dual**:
1. **Corteza Neuronal (Pesos Q2_0):** Red de razonamiento y gramática a 2-bits (19.72 MB).
2. **Hipocampo Congénito (Isla `.gmem` Toroidal):** Almacén asociativo zero-copy mmap indexado en 384 dimensiones, estructurado en 3 nichos:
   * **Nicho Documental:** Hechos fácticos, ciencia, geografía y arquitectura GAJE.
   * **Nicho Episódico:** Memoria contextual de acciones y deducciones.
   * **Nicho Conversacional:** Historial de diálogo activo con el usuario.

```
                  ARQUITECTURA DE MEMORIA CONGÉNITA
                  
   ┌───────────────────────────────────────────────────────────────┐
   │                  models/born/max_laser.gaje                   │
   │                                                               │
   │  ┌─────────────────────────┐     ┌─────────────────────────┐  │
   │  │    Corteza Neuronal     │     │   Hipocampo Congénito   │  │
   │  │  12 Capas Q2_0 (D=384)  │◄───►│       Isla .gmem        │  │
   │  │  (Gramática y Sintaxis) │ Mmap│   (3 Nichos Fácticos)   │  │
   │  └─────────────────────────┘     └─────────────────────────┘  │
   │                 │                             │               │
   │                 ▼                             ▼               │
   │     [ Proyección lm_head ] ◄────── [ Resonancia de Contexto ] │
   └───────────────────────────────────────────────────────────────┘
```

---

## 2. Fases de Implementación Técnica

### Fase 1: CLI y Cabecera Binaria (`gaje-cli birth --with-memory`)
* **Archivo:** `src/bin/gaje-cli.rs` y `src/io/flat_header.rs`.
* **Tareas:**
  1. Añadir el flag `--with-memory` y `--memory-niche` a la estructura `BirthArgs`.
  2. Inicializar la estructura `IslandOrchestrator::new(dim)` al momento de crear el organismo.
  3. Registrar el bloque de memoria `.gmem` contiguo en la cabecera `FlatHeaderV2` (`gmem_offset`, `gmem_size`).

```rust
// Ejemplo en src/bin/gaje-cli.rs:
if args.with_memory {
    let mut orchestrator = IslandOrchestrator::new(args.dim as u32);
    orchestrator.seed_documental_niche("data/genesis_facts_corpus.jsonl")?;
    // Ensamblar cabecera con offset a gmem
}
```

---

### Fase 2: Inyección Residual en Inferencia y Búsqueda Zero-Copy
* **Archivo:** `src/nn/llm/forward.rs` y `src/compute/island.rs`.
* **Tareas:**
  1. Conectar el forward pass para que en las capas intermedias (capa 6 y 10) se realice una consulta mmap en $<0.3\text{ ms}$:
     $$\mathbf{x}_{\text{inyectado}} = \mathbf{x} + \alpha \cdot \text{Vector}(\text{Top-}1_{\text{gmem}})$$
  2. Asegurar que la búsqueda en los 3 nichos (`episodic`, `documental`, `conversational`) respete el presupuesto de latencia en CPU ARM64.

---

### Fase 3: Bucle de Crianza Co-Evolutiva (`train-born`)
* **Archivo:** `src/bin/gaje-cli.rs` (subcomando `train-born`).
* **Tareas:**
  1. Durante el entrenamiento STE, alimentar el nicho documental con datos estructurados mientras la red aprende a formular oraciones.
  2. Ajustar dinámicamente los pesos de nicho $\mathbf{w} = [w_{\text{epi}}, w_{\text{doc}}, w_{\text{chat}}]$ para minimizar la pérdida de generación.
  3. Comprobar que la pérdida descienda a $\text{Loss} < 2.0$ sin saturar los centroides cuaternarios.

---

### Fase 4: Integración en la Web UI y Telemetría HUD
* **Archivo:** `examples/ui/web_ui/` y `src/server/`.
* **Tareas:**
  1. Mostrar en el panel de telemetría de la Web UI los 3 nichos de memoria en vivo (activaciones, similitud coseno y hechos recuperados).
  2. Permitir al usuario inspeccionar el contenido de la memoria congénita en tiempo real sin reiniciar el servidor.

---

## 3. Matriz de Entregables y Hitos

| Hito | Módulo Involucrado | Criterio de Éxito | Estado |
| :--- | :--- | :--- | :---: |
| **Hito 1: CLI con Memoria** | `src/bin/gaje-cli.rs` | `gaje-cli birth --with-memory` crea `.gaje` con `.gmem` incrustado. | 🟡 Listo para codificar |
| **Hito 2: Búsqueda Mmap** | `src/compute/island.rs` | Búsqueda vectorial en 3 nichos en $<0.5\text{ ms}$ en ARM64. | 🟢 Verificado en librería |
| **Hito 3: Crianza Conjunta** | `train-born` con `.gmem` | Reducción de pérdida a $\text{Loss} < 2.0$ y cero olvido fáctico. | ⚪ Pendiente |
| **Hito 4: Certificación** | `docs/reports/` | Reporte oficial con 100% de acierto en preguntas factuales. | ⚪ Pendiente |

---

## 4. Próxima Acción Inmediata
Implementar el soporte de `--with-memory` en `src/bin/gaje-cli.rs` y vincular el `IslandOrchestrator` en la génesis nativa de Rust.
