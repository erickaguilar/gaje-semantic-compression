# 🏝️ Plan de Implementación: Island Model Evolution (Silver Fetus)

**Fecha:** 26 de mayo de 2026
**Objetivo:** Implementar la crianza masiva por poblaciones ("Island Model") en Rust puro para el Silver Fetus (10MB), garantizando alta fidelidad gramatical y previniendo el colapso de atención (*Semantic Drift*).

## 1. Fundamento Teórico (¿Por qué el Island Model?)
El entrenamiento tradicional por gradientes sufre al actualizar tensores discretos (2 bits). El `SpikingEvolutionEngine` actual muta una sola población. Si la población cae en un "mínimo local" (aprende a repetir palabras pero pierde el contexto), la evolución se estanca.
El **Island Model** divide la población en sub-grupos aislados ("islas"). Cada isla explora soluciones diferentes en paralelo. Periódicamente, los "mejores individuos" migran entre islas, cruzando genes (deltas de pesos) altamente exitosos.

## 2. Arquitectura de las Islas para Hardware ARM (Android/Termux)
Considerando los recursos de un dispositivo móvil moderno:
*   **Número de Islas:** 4 (Mapeadas a los 4 núcleos de alto rendimiento de ARM).
*   **Población por Isla:** 16 organismos (Total: 64 variantes del Silver Fetus mutando simultáneamente).
*   **Ciclo de Migración:** Cada 50 generaciones (Epochs).
*   **Motor de Paralelismo:** `rayon::par_iter_mut()` garantizará el uso del 100% de la CPU sin bloquear el sistema.

## 3. La Función de Fitness Híbrida (IQAT + CoT)
Para que el organismo sobreviva, no solo debe predecir bien, debe retener el contexto. El *Fitness Score* se calculará invirtiendo la penalización de dos métricas evaluadas sobre el `consolidated_silver_dataset.txt`:

1.  **Fitness de Coherencia (70% del peso):** Evaluado usando el "Council of Teachers" (SmolLM/Qwen2). Mide qué tanto se parece la predicción probabilística del Silver Fetus a la de los maestros.
2.  **Fitness de Retención / Needle Test (30% del peso):** Cada 10 iteraciones, se inyecta un "dato clave" al inicio del contexto. Si el organismo olvida este dato (colapso de atención), su fitness cae drástico, forzando su eliminación.

## 4. Estructura de Código (Soberanía Nativa en Rust)

Debemos extender la arquitectura en `src/` para soportar las islas.

### A. Modificación de `src/core/evolution_bitwise.rs`
Crear un nuevo struct `IslandModelEngine`:
```rust
pub struct IslandModelEngine {
    pub islands: Vec<SpikingEvolutionEngine>, // 4 Islas
    pub migration_rate: usize, // Cada 50 generaciones
    pub topology_map: Arc<CentroidTopology>, // El esqueleto relacional (CAM)
}
```

### B. Módulo de Crianza Principal: `src/bin/silver-breeder.rs`
Crear un nuevo binario ejecutable (`cargo run --release --bin silver-breeder`) que orqueste la evolución:
1.  **Fase de Scaffolding:** Inicializa 64 `GenomicLLM` (Silver Fetus v1).
2.  **Carga de Memoria Compartida:** Carga el `consolidated_silver_dataset.txt` en un buffer `Arc<Vec<u8>>` para que todas las islas lo lean sin duplicar RAM.
3.  **Bucle de Islas (Rayon):**
    *   `islands.par_iter_mut().for_each(|isla| isla.evolve(dataset));`
4.  **Migración (Cross-Over):** Intercambia los mejores pesos (centroides de 2 bits) entre islas.
5.  **Commit a la Base de Datos:** Guarda el mejor espécimen en `MUTATIONS_TABLE` de RocksDB/LMDB.

## 5. Operadores de Mutación
En la arquitectura genómica de 2 bits, el "ADN" no se ajusta por gradientes, sino por volteo de bits estocástico guiado por la entropía:
*   **Mutación Somática:** *Bit-Flipping* aleatorio (XOR) en los pesos de 2 bits con una tasa base del 0.05%.
*   **Mutación Relacional (Epigenética):** Ajuste de los valores de bias en los tensores de atención para favorecer o inhibir ciertas conexiones topológicas (refinamiento de la `Centroid Adjacency Matrix`).

## 6. Fases de Ejecución

1.  **Día 1:** Implementación de las estructuras de Rust (`IslandModelEngine`).
2.  **Día 2:** Integración de la función de Fitness Híbrida nativa (Sin llamar a Python para evitar el GIL).
3.  **Día 3:** Ejecución del entrenamiento masivo (`silver-breeder`) durante 4 horas en el dispositivo móvil usando el dataset consolidado de 2.5MB.
4.  **Día 4:** Validación del modelo resultante con el script `benchmarks/needle_test.py` para certificar la supervivencia del "Silver Fetus".

---
*Este plan establece la hoja de ruta para la primera evolución no-lineal masiva del proyecto.*
