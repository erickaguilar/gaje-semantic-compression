# 🧬 GAJE-Flow: Experimento de Evolución Genética a 2-Bits (v1.0.0-alpha)

Este documento detalla el diseño, la justificación científica y la metodología de ejecución para el experimento de evolución de pesos a 2-bits nativos dentro del ecosistema **GAJE (Genomic Adaptive Joint Embedding)**.

---

## 1. Justificación Científica y Antecedentes

### El Colapso de 2-Bits Post-Entrenamiento (PTQ)
En auditorías empíricas previas, determinamos que la cuantización directa de 2-bits sobre modelos pre-entrenados en precisión completa causa un **colapso semántico absoluto**. A pesar de corregir el mapa de Gray y estabilizar las fases individuales de cada capa hasta una similitud cosenoidal de $\approx 0.94-0.97$, la propagación del error residual a través de la arquitectura profunda del Transformer (120 proyecciones no lineales en 30 bloques) decae de forma exponencial:
$$\text{CosSim}_{\text{final}} \approx (\text{CosSim}_{\text{capa}})^{120} \approx (0.97)^{120} \approx 0.02$$
Este decaimiento destruye la coherencia de los logits de salida, lanzando al modelo a atractores repetitivos infinitos (p. ej., el bucle de la *"Bundesliga"* o de *"fromords"*).

### La Hipótesis de Evolución del "Embrión" en 2-Bits
En lugar de forzar un modelo maduro de alta precisión a 2-bits, **este experimento plantea nacer al modelo directamente en 2-bits**. Al usar el algoritmo evolutivo nativo de GAJE, los operadores genéticos buscan combinaciones discretas en el hipercubo de 4 estados posibles por peso (ej: `[-1.5, -0.5, 0.5, 1.5]` correspondientes a los códigos Gray `00, 01, 11, 10` de 2-bits).
*   **Plasticidad Genética**: Las capas posteriores del modelo evolucionan para compensar activamente la deriva y rotación de fase introducidas por las capas anteriores.
*   **Espacio de Búsqueda Compacto**: Evolucionar un modelo en 2-bits reduce la complejidad combinatoria del genoma a $4^N$ estados posibles, optimizando los pasos de mutación bitwise.

---

## 2. Arquitectura del Experimento

El experimento se compone de dos piezas clave de software:

```
 smollm2-135m-instruct (FP16 GGUF)
               │
               ▼  (scripts/export_smollm2_2bit_flat.py)
   smollm2_2bit_flat.gaje.flat (4-estados por peso)
               │
               ▼  (src/bin/gaje-2bit-breeder.rs)
  [ Island Model: 3 Islas x 6 individuos ] ◄── Evaluador Coherencia (Teacher Council FP32)
               │
               ▼ (20 Generaciones con Rayon)
  smollm2_2bit_evolved.gaje (Checkpoint SQLite evolutivo)
```

### A. El Exportador: `scripts/export_smollm2_2bit_flat.py`
Extrae las matrices de pesos del modelo de referencia `SmolLM2-135M-Instruct` en FP16 y las comprime directamente a **2-bits nativos**, generando el archivo plano `models/production/smollm2_2bit_flat.gaje.flat`.
*   **Parámetros de cuantización**: `attn_bit_depth = 2` y `ffn_bit_depth = 2` con tamaño de bloque escalar de 32 elementos.

### B. El Criador Evolutivo: `src/bin/gaje-2bit-breeder.rs`
Un ejecutable de Rust de alto rendimiento que implementa la simulación evolutiva basada en poblaciones distribuidas (Island Model):
*   **Población**: 3 islas aisladas con 6 individuos por isla.
*   **Operadores Genéticos**: Tasa de mutación del 1.0% (alteración aleatoria de los bits en la base de datos de pesos) y cruzamiento (*crossover*) mediante intercambio de segmentos genómicos entre individuos de la élite.
*   **Migración**: Intercambio periódico del mejor organismo de cada isla cada 10 generaciones para inyectar diversidad y prevenir la convergencia prematura.
*   **Función de Fitness (Consenso de Maestros)**: Se mide la similitud de los logits del estudiante contra la distribución probabilística de un modelo maestro en precisión completa (`smollm2-135m-f16.gguf`) sobre un dataset extendido en español (`data/datasets/dataset_es_ext.txt`).

---

## 3. Metodología de Ejecución

Sigue estos pasos en la terminal para iniciar el experimento en el nuevo hardware:

### Paso 1: Generar el Embrión de 2-Bits en Formato Plano
Ejecuta el script de exportación para compilar el modelo de 135M a 2-bits nativos:
```bash
python3 scripts/export_smollm2_2bit_flat.py
```
*Esto generará el archivo `models/production/smollm2_2bit_flat.gaje.flat` con un tamaño aproximado de **~280 MB** (la mitad del peso de la versión en 4-bits).*

### Paso 2: Compilar el Binario del Criador en Rust
Usa Cargo para compilar el nuevo binario optimizado para release:
```bash
cargo build --release --bin gaje-2bit-breeder
```

### Paso 3: Lanzar la Evolución Genómica
Ejecuta el criador evolutivo pasando el modelo de 2-bits como estudiante:
```bash
./target/release/gaje-2bit-breeder models/production/smollm2_2bit_flat.gaje.flat
```

---

## 4. Métricas Esperadas y Análisis

*   **Fitness Inicial (Generación 1)**: Se espera una puntuación de *Coherence Fitness* extremadamente baja debido a la destrucción masiva de información provocada por la cuantización estática de 2-bits.
*   **Dinámica Evolutiva**: A lo largo de las 20 generaciones, el *Best Fitness* de la Isla 0 debería ascender gradualmente. Esto confirmará que la mutación a nivel de bits está reacomodando los estados para recuperar la resonancia de los logits perdidos.
*   **Punto de Control**: Al finalizar, el mejor organismo se guardará en `models/checkpoints/smollm2_2bit_evolved.gaje`, sirviendo como base para estudios de plasticidad semántica a largo plazo.
