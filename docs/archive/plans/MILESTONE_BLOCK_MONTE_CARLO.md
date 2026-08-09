# 🧬 Hito: Ajuste Fino de Centroides por Bloque mediante Búsqueda Monte Carlo (Fase 5.5)

**ID del Hito:** `milestone-gaje-block-monte-carlo`
**Estado:** Planificado / Listo para SDD
**Carpeta Destino:** `src/nn/block.rs` y `src/nn/distiller.rs`
**Objetivo:** Implementar un optimizador de mutaciones locales de tipo "natural selection" por bloque transformador para evadir la ceguera de los gradientes continuos en la representación de 2 bits.

---

## 1. Fundamento Matemático

Dado que los pesos de 2 bits están cuantizados discreta y rígidamente, el Descenso de Gradiente (SGD/Adam) no puede calcular derivadas útiles para realizar cambios sub-unitarios en los centroides sin romper la estabilidad.

Reemplazamos esto por una **Búsqueda Probabilística de Monte Carlo (Breeding)** local para ajustar el factor de escala $\gamma$ (amplitud) y el desplazamiento $\beta$ (sesgo) de los centroides de cada capa de forma independiente.

### Proposición de Mutación:
Para una capa $L$, con centroides $\mathbf{C} = [c_0, c_1, c_2, c_3]$, generamos una mutación en el espacio de fase toroidal:

$$\gamma \sim \mathcal{N}(0, \sigma_{mut}^2)$$
$$\beta \sim \mathcal{N}(0, (\sigma_{mut} \cdot 0.5)^2)$$

Donde los nuevos centroides candidatos son:

$$\mathbf{C}' = \text{sort}\left( \mathbf{C} \cdot (1 + \gamma) + \beta + \mathbf{\epsilon}_{local} \right)$$

Aquí, $\mathbf{\epsilon}_{local} \sim \mathcal{N}(0, (\sigma_{mut} \cdot 0.1)^2)$ representa un ajuste fino de fluctuación local por base.

### Criterio de Selección:
Evaluamos la pérdida de entropía cruzada (Perplejidad) sobre un lote de validación pequeño $X_{val}$ (por ejemplo, 16 secuencias de 128 tokens):

$$\text{Loss}(W|\mathbf{C}') = -\frac{1}{N} \sum_{i=1}^{N} \log P(x_i | x_{<i}; \mathbf{C}')$$

La mutación es aceptada bajo el criterio de selección natural estricta:

$$\mathbf{C}_{t+1} = \begin{cases} \mathbf{C}' & \text{si } \text{Loss}(W|\mathbf{C}') < \text{Loss}(W|\mathbf{C}) \\ \mathbf{C} & \text{en caso contrario} \end{cases}$$

---

## 2. Escenario BDD (Behavior-Driven Development)

El comportamiento de este optimizador evolutivo local se detalla en el siguiente escenario:

```gherkin
Característica: Ajuste Fino de Centroides por Bloque mediante Monte Carlo
  Como optimizador de precisión discreta
  Quiero mutar probabilísticamente los centroides de cada bloque transformador
  Para minimizar la perplejidad del estudiante sin usar gradientes continuos.

  Escenario: Optimización evolutiva al final de cada época
    Dado que un ciclo de destilación por gradientes ha finalizado su época actual
    Y la pérdida promedio en el lote de validación es "L_inicio"
    Cuando el orquestador "GenomicDistiller" ejecuta la búsqueda Monte Carlo por bloque
    Entonces los centroides de cada capa GenomicLinear deben recibir 100 mutaciones de prueba
    Y la pérdida final de validación "L_fin" debe ser menor o igual a "L_inicio"
    Y la entropía de codificación de uso de los centroides debe mantenerse por encima de 1.85
```

---

## 3. Plan de Implementación Técnica (TDD)

1. **Paso Red (Fallo):**
   * Crear un test en `tests/unit/test_block_monte_carlo.rs` que intente ejecutar la mutación de centroides sobre una estructura `RustGenomicBlock` simulada y verifique que si el método no está implementado devuelva error o no altere la pérdida.

2. **Paso Green (Paso):**
   * Añadir el método `apply_mutation_core` en `src/nn/linear.rs` para permitir mutar y deshacer cambios en los centroides rápidamente.
   * Implementar en `src/nn/block.rs` la función de evaluación local de pérdida sobre un lote temporal.
   * Enlazar el optimizador en `src/bin/micro-distiller.rs` al final del bucle de cada época.

3. **Refactor:**
   * Paralelizar la evaluación de mutaciones por bloque utilizando Rayon (las capas no dependen entre sí para sus centroides locales).
   * Guardar logs de la ganancia evolutiva de cada capa en `benchmarks/logs/mutation_gain.json` para análisis de deriva.
