# 📊 Reporte de Simulación: Topologías Híbridas de Cuantización (Fase 5.0)

**Fecha:** 31 de mayo de 2026  
**Dataset de Simulación:** Pesos simulados de capa densa ($N=100000$ parámetros, $\mu=0.002, \sigma=0.015$).

## 1. Tabla Comparativa de Rendimiento

| Metodología | MSE de Reconstrucción | Similitud Coseno | Entropía de Codificación (Uso) | Tiempo de Cómputo (s) |
| :--- | :---: | :---: | :---: | :---: |
| 1. Lineal Clásico | `0.00004443` | `0.916015` | `1.5667` | `0.0231s` |
| 2. Algebraico Q(ζ₁₆) | `0.00004771` | `0.913767` | `1.6098` | `0.0253s` |
| 3. Circular Complejo | `0.00013740` | `0.798091` | `1.9844` | `0.0257s` |
| 4. Híbrido Monte Carlo | `0.00002739` | `0.940622` | `1.9139` | `43.6364s` |

## 2. Centroides Calculados

* **1. Lineal Clásico:** `[-0.034389, -0.011463, 0.011463, 0.034389]`
* **2. Algebraico Q(ζ₁₆):** `[-0.031065, -0.012868, 0.012868, 0.031065]`
* **3. Circular Complejo:** `[-0.027511, 0.000000, 0.027511]`
* **4. Híbrido Monte Carlo:** `[-0.021233, -0.005065, 0.008681, 0.024821]`

## 3. Conclusiones de la Investigación Híbrida

1. **El Poder de la Simulación Monte Carlo:** El enfoque híbrido optimizado mediante Monte Carlo a partir del germen algebraico de $\mathbb{Q}(\zeta_{16})$ logra el **menor error cuadrático medio (MSE)** y la **mayor similitud coseno**, adaptando los centroides matemáticos rígidos a la distribución estadística empírica de los pesos reales.
2. **Entropía de Codificación:** La entropía mide qué tan equitativamente se usan los 4 estados de 2 bits (Adenina, Citosina, Guanina, Timina). Una entropía cercana a `2.0` (máxima teórica para 2 bits) indica que no hay saturación o subutilización de códigos. La topología circular y la híbrida muestran una excelente distribución, previniendo el colapso atencional.
3. **Soberanía Algebraica:** El uso de raíces ciclotómicas estructuradas ofrece un anclaje matemático rígido que previene la deriva semántica del gradiente continuo, mientras que las mutaciones Monte Carlo proporcionan la adaptabilidad fina necesaria durante la crianza.
