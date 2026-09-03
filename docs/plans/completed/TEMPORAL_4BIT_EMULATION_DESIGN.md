# 🧬 Emulación Temporal de 4-Bits mediante Secuencias Estocásticas de 2-Bits y Timing Wheel Neuromórfica

> **Versión:** v1.6.0-alpha (Silver Adult)
> **Fecha:** 20 de agosto de 2026
> **Estado:** 📝 Propuesta de Investigación R&D / Especificación de Arquitectura
> **Ubicación:** `docs/plans/TEMPORAL_4BIT_EMULATION_DESIGN.md`
> **Módulos Asociados:** `src/compute/timing_wheel.rs`, `src/nn/spiking/neuron.rs`, `src/compute/lagrangian.rs`

---

## 1. 🎯 Hipótesis y Motivación

En los experimentos de compresión extrema del protocolo GAJE se demostró la siguiente disyuntiva empírica:
1. **Compresión a 2-Bits:** Ofrece la máxima densidad de almacenamiento ($\approx 290\text{ MB}$ para 135M, $96\%$ de ahorro de RAM), pero sufre de **colapso semántico exponencial** a través de las 120 capas del transformer ($0.97^{120} \approx 0.02$).
2. **Compresión a 4-Bits (Q4_0):** Retiene el $100\%$ de precisión factual y estabilidad ($0.9999^{120} \approx 0.988$), pero requiere el doble de memoria por peso.

### 💡 La Solución Propuesta:
Utilizar el **reloj interno de eventos (*Timing Wheel*)** y el **potencial de membrana del emulador neuromórfico (`SpikingNeuron`)** para recibir trenes de pulsos de **2-bits en el tiempo**, acumulando y emulando la **resolución y estabilidad de 4-bits (16 niveles)** antes de propagar la activación a la siguiente capa.

---

## 2. 🏛️ Arquitectura del Sistema: Integración Espacio-Temporal

```
                       MEMORIA (2-bits ultradensa)
                                [ 01 | 11 ]
                                  │     │
                 ┌────────────────┴─────┴────────────────┐
                 ▼ (Tick t0: MSB)                        ▼ (Tick t1: LSB)
         ┌───────────────┐                       ┌───────────────┐
         │ Spike 2-bits  │                       │ Spike 2-bits  │
         │ Ponderación 4x│                       │ Ponderación 1x│
         └───────┬───────┘                       └───────┬───────┘
                 │                                       │
                 └──────────────► ⏰ TIMING WHEEL ◄──────┘
                               (src/compute/timing_wheel.rs)
                                         │
                                         ▼
                            ┌─────────────────────────┐
                            │   POTENCIAL DE MEMBRANA │ ──► Resolución 4-bits
                            │      (V_mem acumulado)  │     (16 niveles / Q4)
                            └─────────────────────────┘
```

---

## 3. 🔬 Mecanismos de Reconstrucción Temporal

### A. Enfoque 1: Bit-Slice Secuencial Ponderado (Determinista)
Los 4 bits de un peso Q4 se dividen en dos bloques de 2 bits:
* $\text{Chunk}_0$ (Bits más significativos - MSB): Determina el cuadrante grueso de amplitud.
* $\text{Chunk}_1$ (Bits menos significativos - LSB): Determina el ajuste fino dentro del cuadrante.

El reloj interno programa la llegada de los pulsos en dos ranuras de tiempo discretas ($t_0, t_1$):
$$V_{\text{mem}} = 4 \cdot \mu(\text{Chunk}_0) + 1 \cdot \mu(\text{Chunk}_1)$$

Donde $\mu(\cdot)$ es la función de decodificación de los 4 centroides genómicos base ($A, C, G, T$).

### B. Enfoque 2: Integración Estocástica Modulada por Secuencia (*Dithered Spiking*)
En este modo, el flujo de 2 bits representa una tasa de disparo probabilística modulada por un generador pseudoaleatorio de alta frecuencia (*LFSR dither*):
* A lo largo de $K$ ticks del reloj ($K \in [2, 4]$), la frecuencia promedio de eventos que ingresan en la *Timing Wheel* converge al valor continuo:
  $$W_{\text{eff}} = \frac{1}{K} \sum_{k=1}^K S_k(t)$$
* **Efecto Matemático:** El teorema del límite central suprime el ruido de cuantización de 2-bits, transformándolo en una señal de alta fidelidad equivalente a 4-bits.

---

## 4. ⚡ Precisión Epigenética Adaptativa (Dynamic Multi-Tick)

Uno de los mayores beneficios de la emulación temporal es la **computación adaptativa según la complejidad del token**:

| Tipo de Token / Capa | Ticks del Reloj | Resolución Efectiva | Velocidad | Caso de Uso |
| :--- | :---: | :---: | :---: | :--- |
| **Conectores y Artículos** (*"el", "de"*) | **1 Tick** | **2-Bits** | ⚡ **$60+\text{ tok/s}$** | Capas tempranas / Sintaxis simple |
| **Palabras Clave y Sustantivos** | **2 Ticks** | **4-Bits (Emulado)** | 🚀 **$30\text{ tok/s}$** | Atención intermedia / Factualidad |
| **Razonamiento / Álgebra / Código** | **4 Ticks** | **6-Bits (Emulado)** | 🟢 **$15\text{ tok/s}$** | Proyecciones críticas / Salida fina |

---

## 5. 🛠️ Especificación de Implementación en Rust

### 5.1 Extensión de `SpikingNeuron` (`src/nn/spiking/neuron.rs`)

```rust
impl SpikingNeuron {
    /// Integra una secuencia de pulsos de 2-bits en el tiempo para reconstruir 4-bits
    pub fn integrate_temporal_2to4(&mut self, chunk_msb: u8, chunk_lsb: u8, centroids: &[f32; 4]) -> f32 {
        let val_msb = centroids[(chunk_msb & 0b11) as usize];
        let val_lsb = centroids[(chunk_lsb & 0b11) as usize];

        // Integración ponderada en el potencial de membrana
        let reconstructed_q4 = (val_msb * 4.0 + val_lsb * 1.0) / 5.0;
        self.voltage = self.voltage * self.leak_factor + reconstructed_q4;
        self.voltage
    }
}
```

### 5.2 Despacho en `TimingWheel` (`src/compute/timing_wheel.rs`)

```rust
impl TimingWheel {
    /// Programa la llegada de los sub-bloques temporales en ranuras de reloj contiguas
    pub fn schedule_bitstream_ticks(&mut self, neuron_id: usize, chunk_msb: u8, chunk_lsb: u8) {
        self.schedule_event(0, Event::SpikeChunk { neuron_id, bits: chunk_msb, weight_scale: 4.0 });
        self.schedule_event(1, Event::SpikeChunk { neuron_id, bits: chunk_lsb, weight_scale: 1.0 });
    }
}
```

---

## 6. 📅 Plan de Ejecución y Validación

| Fase | Hito | Criterio de Aceptación |
| :---: | :--- | :--- |
| **Fase 1** | **Micro-Benchmark de Capa Individual** | Test unitario en `tests/unit/test_temporal_reconstruction.py` verificando $\text{CosSim} \ge 0.998$ frente a Q4_0 en una proyección lineal aislada. |
| **Fase 2** | **Integración en `TimingWheel`** | Ejecución en el ciclo de inferencia neuromórfico (`src/compute/lagrangian.rs`) sin bloqueos de sincronización. |
| **Fase 3** | **Evaluación Multicapa (120 Bloques)** | Verificación de retención de similitud en transformer completo ($\text{CosSim} \ge 0.985$ tras 120 capas). |
| **Fase 4** | **Benchmark de Memoria vs Latencia** | Certificar archivo plano `.gaje.flat` de $\approx 290\text{ MB}$ con calidad idéntica a Q4_0. |
