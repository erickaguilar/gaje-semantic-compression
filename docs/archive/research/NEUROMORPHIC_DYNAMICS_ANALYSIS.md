# 🧬 Hallazgos: Conflicto de Dinámicas y Soberanía Temporal (v1.0)

**Fecha:** 21 de Mayo, 2026
**Investigación:** Dinámica Discreta vs. Continua en Modelos de 2-bits.

## 1. El Diagnóstico: El Muro de la Digitalización
Se ha identificado que la pérdida de coherencia en los modelos GAJE de 2-bits no es solo un problema de cuantización, sino de **dinámica neuronal**. Al forzar pesos entrenados en espacios continuos (F32) a través de neuronas discretas (LIF), se producen tres fallos críticos:

1.  **Muerte por Entropía (Todo o Nada):** La pérdida de la "textura" semántica. Al no haber valores intermedios, la señal se convierte en ruido estocástico tras 24 capas.
2.  **Ineficiencia del Rate Coding:** Intentar representar precisión mediante la cantidad de disparos requiere demasiados ticks de reloj, destruyendo la latencia en dispositivos móviles.
3.  **Competencia Ruidosa:** Sin un Softmax continuo, la selección de tokens se vuelve una competencia de "suerte" estadística donde palabras basura roban la atención.

## 2. Los Hallazgos Solución

### A. Graded Spiking (Potenciales Graduados)
Las neuronas no deben ser 100% digitales. Al permitir que el spike transporte el residuo de energía (`intensidad = energía - umbral`), devolvemos la gradación al sistema sin necesidad de multiplicaciones. El "golpe" del centroide en la siguiente capa se modula por esta intensidad.

### B. Temporal/Phase Coding (Latencia de Disparo)
En lugar de contar disparos, mediremos **cuándo** ocurren dentro de un tick.
*   **Fase Temprana (0ms):** Valor alto (0.9).
*   **Fase Tardía (1ms):** Valor bajo (0.1).
Esto permite una precisión infinita dentro de un solo tick lógico, aprovechando la infraestructura de la **Timing Wheel O(1)**.

### C. Inhibición Lateral (K-WTA)
Sustitución del Softmax mediante una barrera física. Las primeras neuronas en disparar (las de mayor confianza) activan una señal de inhibición que "apaga" a sus vecinas, limpiando el ruido de los centroides de 2-bits y forzando la coherencia.

## 3. Conclusión Técnica
El motor debe evolucionar de un "Emulador de Neuronas" a un **"Procesador de Señales Temporales"**. La coherencia no vendrá de más bits, sino de una gestión más inteligente del tiempo y la intensidad del disparo.

---
*Documento de investigación para el protocolo GAJE-Flow.*
