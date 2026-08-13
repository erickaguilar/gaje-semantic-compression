# 🧬 Potencial Evolutivo de SMG-1: Hacia el Nacimiento Genómico (Born-Genomic)

**Fecha:** 24 de mayo de 2026
**Estatus:** Arquitectura Validada (v0.9.7)
**Target:** Micro-Organismos de < 10 MB

## 1. Definición de SMG-1 (Standard Micro-Genome)
La arquitectura SMG-1 es un motor neuromórfico de tres capas diseñado para operar exclusivamente en el espacio de 2 bits. A diferencia de los Transformers masivos, SMG-1 está optimizado para la **plasticidad sináptica extrema** y la evolución dirigida mediante simulación de Monte Carlo.

### Especificaciones del Núcleo:
- **Capa 0 (Entrada/Latente):** 256 neuronas LIF (Leaky Integrate-and-Fire).
- **Capa 1 (Lógica):** 128 neuronas de procesamiento secuencial.
- **Capa 2 (Salida):** Dimensionada según el vocabulario (ej. 16,384).
- **Protocolo:** Zero-Multiplication (Suma directa de centroides).

## 2. Por qué SMG-1 es superior para la evolución
El entrenamiento de modelos como SmolLM2-135M a 2 bits (destilación) sufre de inestabilidad debido a la pérdida de precisión en capas MLP profundas (el "límite semántico duro"). SMG-1 soluciona esto mediante:

1.  **Nacimiento bajo Ruido:** El modelo no se comprime, sino que *nace* con pesos de 2 bits. Sus centroides evolucionan adaptados a la baja resolución desde el primer disparo.
2.  **Paralelismo de Isla (Rayon):** Permite ejecutar 1,000 generaciones de SMG-1 en segundos, evaluando mutaciones bitwise (XOR) en tiempo real.
3.  **Homeostasis Integrada:** Capacidad nativa en Rust para regular el voltaje de membrana, evitando la explosión de señales en redes pequeñas.

## 3. Flujo de Trabajo Recomendado
Para alcanzar la meta de inteligencia coherente en un micro-genoma, se debe seguir el comando:
```bash
cargo run --release --bin gaje-smg1-trainer
```

### Fases del Nacimiento:
- **Imprimación:** Exposición a bi-gramas y tri-gramas básicos.
- **Resonancia:** Refinamiento de centroides locales (`refine_step`) para estabilizar la gramática.
- **Clonación:** Fijación de anclas en el ADN para proteger conceptos aprendidos.

---
*Este documento establece a SMG-1 como el vehículo oficial para el desarrollo del "Gold Embryo" (v1.0).*
