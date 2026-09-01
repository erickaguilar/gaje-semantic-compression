# 🎯 Plan Arquitectónico: El Láser Semántico (`max_laser.gaje`) — Guiado de Onda Profundo y Colimado

> **Fecha:** 1 de Septiembre de 2026  
> **Versión:** `GAJE Helix v1.7.0`  
> **Estado:** `APROBADO PARA IMPLEMENTACIÓN`  
> **Objetivo:** Implementar la geometría colimada *Deep & Narrow* ($D=384, L=12, H=6, V=4096$) para maximizar la coherencia de fase y alcanzar $PPL < 8$ en ~22 MB de huella total.

---

## 1. Fundamentación Geométrica

Frente a la dispersión de los modelos anchos y poco profundos (*Wide & Shallow*), la arquitectura **Láser Semántico** opera como una guía de onda óptica colimada:

```
                            GEOMETRÍA LÁSER
                            
       [ Haz Estrecho: D = 384 ] ─────────────────────► ρ = 10.6 (Óptimo)
       [ Lentes Colimadoras: K-WTA 15% ] ────────────► Poda 85% Ruido
       [ Cavidad Fabry-Pérot: L = 12 capas ] ────────► Resonancia Jerárquica
       [ Sumidero Calibrado: V = 4,096 tokens ] ─────► Cero Hacinamiento Léxico
```

### Tabla Comparativa de Geometrías

| Parámetro | Micro Base (`max_human`) | **Láser Semántico (`max_laser`)** | Modelo Clásico Ancho |
| :--- | :---: | :---: | :---: |
| **Dimensión Oculta ($D$)** | 256 | **384** | 1024 |
| **Capas Transformer ($L$)** | 8 | **12** | 6 |
| **Cabezas de Atención ($H$)** | 4 | **6** ($64\text{ dims/head}$) | 16 |
| **Dimensión FFN ($3\times D$)** | 768 | **1024** | 2816 |
| **Vocabulario ($V$)** | 4,096 | **4,096** | 49,152 |
| **Presión Dimensional ($\rho$)** | 16.0 | **10.6** | 48.0 |
| **Huella en Disco (Q2_0)** | 10.53 MB | **~21.8 MB** | ~150 MB |
| **Resonancia de Razonamiento** | Básica (8 capas) | **Avanzada (12 capas)** | Dispersa |

---

## 2. Fases de Ejecución

### Fase 1: Génesis Conforme de `max_laser.gaje`
* Dar a luz al organismo con la geometría colimada:
```bash
./target/release/gaje-cli birth \
    --name max_laser \
    --dim 384 \
    --layers 12 \
    --heads 6 \
    --ffn-dim 1024 \
    --vocab-size 4096 \
    --tokenizer data/gtok_human_4k.bin \
    --output models/born/max_laser.gaje
```

### Fase 2: Crianza Acelerada por Currículo (12 Épocas)
* Entrenamiento Straight-Through Estimator (STE) con decaimiento por capas:
```bash
./target/release/gaje-cli train-born \
    --model models/born/max_laser.gaje \
    --dataset data/latam_curated_50.jsonl \
    --epochs 12 \
    --lr 0.0035 \
    --lr-decay 0.96 \
    --gclip 1.0
```

### Fase 3: Inyección de Memoria Toroidal Asociativa (`.gmem` v2)
* Crear la base de conocimiento factual desacoplada en $D=384$:
```bash
./target/release/gaje-cli epoch snapshot \
    --organism max_laser \
    --dim 384 \
    --comment "Conocimiento estructurado inicial max_laser"
```

### Fase 4: Benchmark y Certificación
* Evaluar TTFT, perplejidad real y throughput:
```bash
./target/release/gaje-cli bench \
    --model models/born/max_laser.gaje \
    --suite quick \
    --format markdown \
    --output docs/reports/MAX_LASER_CERTIFICATION.md
```

---

## 3. Criterios de Aceptación y Certificación

1. **Huella de Memoria:** $< 25\text{ MB}$ en disco / RAM mmap.
2. **Pérdida de Crianza:** $\text{Loss} < 2.50$ tras 12 épocas.
3. **Integridad Numérica:** $0\text{ NaNs} / 0\text{ Infs}$ en auditoría formal.
4. **Throughput de Inferencia:** $> 35\text{ tok/s}$ en CPU ARM64.
