# 🧬 GAJE-Flow: Ciclo de Vida y Protocolo de Certificación

Este diagrama define el flujo de desarrollo desde la concepción del Genoma hasta la certificación de un "Adulto Estable". Cada transición está protegida por un **Gate Empírico**.

```mermaid
graph TD
    %% Fase 1: Concepción e Infraestructura
    subgraph F1 [Fase 1: Concepción - Gold Embryo]
        A[GGUF Master Model] -->|Extraction| B[Genomic Centroids]
        B -->|Phase Mapping| C[Toroidal Topology Q-Zeta16]
        C -->|Quantization| D[2-bit DNA Genome]
        D -->|Injection| E[Stability Anchors F16]
        E -->|Gate 1| F{Certificación Nivel 1: Infraestructura}
    end

    F -->|FAIL: NaNs/Instabilidad| E
    F -->|PASS: Similitud > 0.99| G[Fase 2: Crianza - Silver Fetus]

    %% Fase 2: Crianza y Estabilidad Semántica
    subgraph F2 [Fase 2: Crianza - Silver Fetus]
        G --> H[Native Genomic Training / Distillation]
        H --> I[Mosaic Dataset Injection 500MB]
        I --> J[Epigenetic Refinement]
        J --> K{Certificación Nivel 2: Semántica}
    end

    K -->|FAIL: PPL > 15.0 / Obsesión| H
    K -->|PASS: PPL < 15.0| L[Fase 3: Maduración - Silver Adult]

    %% Fase 3: Maduración y Funcionalidad
    subgraph F3 [Fase 3: Maduración - Silver Adult]
        K --> L[K-WTA Lateral Inhibition Tuning]
        L --> M[Dialectic/Instruct Tuning]
        M --> N{Certificación Nivel 3: Funcional}
    end

    N -->|FAIL: Incoherencia/Alucinación| M
    N -->|PASS: Diálogo Coherente| O[Fase 4: Expansión - Island Model]

    %% Fase 4: Expansión
    subgraph F4 [Fase 4: Expansión - Golden Organism]
        O --> P[Semantic RAG Integration]
        P --> Q[Distributed Island Evolution]
        Q --> R[Sovereign Identity]
    end

    %% Notas de Realidad (Estado Actual)
    style E fill:#4CAF50,stroke:#333,stroke-width:2px
    style J fill:#f44336,stroke:#333,stroke-width:4px,stroke-dasharray: 5 5
    style N fill:#9e9e9e,stroke:#333,stroke-width:2px
    
    click J "docs/meta/EMPIRICAL_TRUTH_STATE.md" "Estado de Bloqueo Actual"
```

## 📋 Descripción de los Gates (Puertas de Verdad)

### 🚪 Gate 1: Estabilidad de Infraestructura (Superado ✅)
*   **Herramienta:** `cargo test --test phase_survival`.
*   **Meta:** El motor de Rust debe mover el genoma sin generar NaNs y manteniendo la señal en el toroide.
*   **Estado:** Implementado y validado en Android.

### 🚪 Gate 2: Estabilidad Semántica (BLOQUEADO ❌)
*   **Herramienta:** `micro-distiller.rs` + `benchmarks/logs/accuracy.log`.
*   **Meta:** La perplejidad (PPL) debe bajar de **15.0**.
*   **Realidad Actual:** Estamos estancados en **PPL 572**. El modelo tiene "obsesión técnica".
*   **Acción Requerida:** Corregir gradientes en el espacio toroidal y diversificar el dataset.

### 🚪 Gate 3: Estabilidad Funcional (Pendiente ⏳)
*   **Herramienta:** `examples/core_demos/chat_soberano.py` (o similar nativo).
*   **Meta:** El modelo debe responder preguntas generales con gramática fluida en español.
*   **Estado:** El modelo actual es "Técnicamente Autista". No pasa esta puerta.

---
*Este diagrama es el mapa de guerra para la Operación Rescate.*
