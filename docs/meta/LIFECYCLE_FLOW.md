# 🧬 GAJE-Flow: Ciclo de Vida y Protocolo de Certificación V1.5

Este diagrama define el flujo de desarrollo y validación basado en los **5 Niveles de Certificación Oficial**.

```mermaid
graph BT
    %% Nivel 5: Soberanía
    subgraph L5 [Certificación Nivel 5: Soberanía Nativa]
        L5_A[Native Binary target/release] --> L5_B{0% Python Deps}
    end

    %% Nivel 4: Eficiencia
    subgraph L4 [Certificación Nivel 4: Eficiencia Green-AI]
        L4_A[big.LITTLE Affinity] --> L4_B[Sparsity Reporting]
        L4_B --> L4_C{Consumo < 0.5W}
    end

    %% Nivel 3: Ingesta
    subgraph L3 [Certificación Nivel 3: Ingesta No-Destructiva]
        L3_A[DNI Ingest] --> L3_B[Zero-Forget Protocol]
        L3_B --> L3_C{Recall Delta < 1%}
    end

    %% Nivel 2: Fidelidad
    subgraph L2 [Certificación Nivel 2: Fidelidad Genómica]
        L2_A[Mosaic Dataset Training] --> L2_B[Epigenetic Tuning]
        L2_B --> L2_C{PPL < 15.0}
    end

    %% Nivel 1: Resonancia
    subgraph L1 [Certificación Nivel 1: Resonancia Toroidal]
        L1_A[Toroidal Mapping Q-Zeta16] --> L1_B[Needle In A Haystack 128k]
        L1_B --> L1_C{100% Accuracy}
    end

    %% Relaciones de Dependencia
    L1_C -->|Estructura Base| L2_A
    L2_C -->|Conocimiento| L3_A
    L1_C -->|Mecánica de Energía| L4_A
    L4_C -->|Despliegue| L5_A

    %% Estados Actuales (Colores)
    style L5 fill:#4CAF50,stroke:#333,stroke-width:2px
    style L4 fill:#4CAF50,stroke:#333,stroke-width:2px
    style L1 fill:#ff9800,stroke:#333,stroke-width:2px
    style L2 fill:#f44336,stroke:#333,stroke-width:4px,stroke-dasharray: 5 5
    style L3 fill:#9e9e9e,stroke:#333,stroke-width:2px

    %% Notas de Bloqueo
    click L2 "docs/meta/EMPIRICAL_TRUTH_STATE.md" "BLOQUEO SEMÁNTICO ACTUAL"
```

## 📋 Resumen de Certificación Empírica

1.  **L5 - Soberanía Nativa:** ✅ **CERTIFICADO**. El motor corre en Rust puro.
2.  **L4 - Eficiencia Green-AI:** ✅ **CERTIFICADO**. Optimizado para ARM Android.
3.  **L1 - Resonancia Toroidal:** ⏳ **EN PROCESO**. Validando recuperación de datos en 128k.
4.  **L2 - Fidelidad Genómica:** ❌ **BLOQUEADO**. PPL inaceptable (572).
5.  **L3 - Ingesta (DNI):** ⏳ **PENDIENTE**. Depende de superar L2.


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
