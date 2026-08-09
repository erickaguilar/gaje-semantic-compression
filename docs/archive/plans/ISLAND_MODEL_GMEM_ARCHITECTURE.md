# 🏝️ GAJE-Flow: Arquitectura del Island Model y Formato Binario de Memoria `.gmem` (v1.0.0 Spec)

## 1. Visión y Objetivos

El **Island Model** para GAJE transforma el motor de inferencia en una **Arquitectura de Memoria Persistente Nativa**. En lugar de depender de librerías de RAG externas o bases de datos vectoriales lentas basadas en HTTP/Python, GAJE integra la búsqueda semántica vectorizada directamente en los kernels nativos de Rust utilizando lectura por mapeo de memoria directa (`mmap`) a través del nuevo formato de archivo **`.gmem`**.

---

## 2. Diagrama de la Arquitectura de Memoria

```text
                               ┌─────────────────────────┐
                               │     Entrada Usuario     │
                               └────────────┬────────────┘
                                            │
                                  Consulta Semántica
                                            │
                               ┌────────────▼────────────┐
                               │   Memory Orchestrator   │
                               └────────────┬────────────┘
                                            │
               ┌────────────────────────────┼────────────────────────────┐
               │                            │                            │
  ┌────────────▼────────────┐  ┌────────────▼────────────┐  ┌────────────▼────────────┐
  │   Isla Episódica        │  │   Isla Documental       │  │  Isla Conversacional    │
  │  (Eventos Recientes)    │  │   (Base Conocimiento)   │  │   (Contexto Activo)    │
  └────────────┬────────────┘  └────────────┬────────────┘  └────────────┬────────────┘
               │                            │                            │
               └────────────────────────────┼────────────────────────────┘
                                            │
                                ⚡ Zero-Copy Mmap (.gmem)
                                            │
                               ┌────────────▼────────────┐
                               │      GAJE Runtime       │
                               └─────────────────────────┘
```

---

## 3. Especificación del Formato Binario `.gmem` (Flat Mmap Memory)

El formato `.gmem` está diseñado para alineamiento de 64-bytes en disco, garantizando cargas instantáneas mediante `mmap` y cero alocaciones en el Heap.

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        ESTRUCTURA BINARIA DE .gmem                     │
├────────────────────────────────────────────────────────────────────────┤
│ Header (64-Bytes):                                                     │
│   - Magic Bytes: [0x47, 0x4D, 0x45, 0x4D] ("GMEM")                     │
│   - Version: u32 (e.g., 1)                                             │
│   - Vector Dim: u32 (e.g., 896)                                        │
│   - Num Entries: u64                                                   │
│   - Index Type: u8 (0=Flat Centroid, 1=Toroidal HNSW, 2=Quantum Anchor) │
│   - Reserved: [u8; 43] (Padding a 64 bytes)                            │
├────────────────────────────────────────────────────────────────────────┤
│ Tabla de Offset e Índices (Vector Offsets):                            │
│   - Array de `GmemEntryHeader`:                                        │
│     - doc_id: u64                                                      │
│     - offset_bytes: u64                                                │
│     - len_bytes: u32                                                   │
│     - centroid_idx: u16                                                │
│     - energy_potential: f32                                            │
├────────────────────────────────────────────────────────────────────────┤
│ Matriz de Embeddings Mapeada:                                          │
│   - Bloques contiguos de vectores alineados a 64-bytes.                │
├────────────────────────────────────────────────────────────────────────┤
│ Payload de Documentos / Texto Comprimido:                              │
│   - Fragmentos de texto indexados por offset directo.                 │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Hitos de Implementación para el Sprint `.gmem`

1. **Fase 1: Módulo Nactivo Rust `src/io/gmem.rs`**
   - Implementar `GmemWriter` y `GmemReader` con soporte de `memmap2`.
   - Soporte para cálculo de distancia coseno vectorizada y Lagrangiana.

2. **Fase 2: Motor de Islas (`src/compute/island.rs`)**
   - Crear el orquestador de nichos semánticos (Episódica, Documental, Conversacional).
   - Inyección de contexto al vuelo previo al prefill del LLM.

3. **Fase 3: Integración FFI en Python (`python/gaje/memory/`)**
   - Exponer la API nativa de memoria para la Web UI y CLI (`gaje-cli`).
