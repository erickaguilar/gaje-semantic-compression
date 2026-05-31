# 🧬 Hito: Inicialización Algebraica mediante Campo Ciclotómico $\mathbb{Q}(\zeta_{16})$ (Fase 5.0)

**ID del Hito:** `milestone-gaje-algebraic-init`  
**Estado:** Planificado / Listo para SDD  
**Carpeta Destino:** `src/io/loader.rs` y `src/compute/math.rs`  
**Objetivo:** Eliminar la cuantización K-Means y los centroides uniformes empíricos en el nacimiento del modelo, reemplazándolos con una rejilla algebraica de fase con simetría de punto-reflexión.

---

## 1. Fundamento Matemático

En lugar de aproximar estadísticamente la distribución de pesos de un modelo denso a nivel lineal, proyectamos el espacio latente sobre el círculo unitario en el campo ciclotómico de orden 16, $\mathbb{Q}(\zeta_{16})$, donde:

$$\zeta_{16} = e^{i\frac{\pi}{8}}$$

El polinomio minimal asociado que define esta estructura geométrica es:

$$\Phi_{16}(x) = x^8 + 1 = 0$$

### Selección de Centroides en la Proyección Real:
Para codificar en 2 bits ($2^2 = 4$ estados discretos), tomamos los componentes reales de las raíces primitivas que exhiben simetría de punto-reflexión y equidistancia angular en la mitad superior del plano complejo:

$$\mathbf{C}_{base} = \left[ \cos\left(\frac{7\pi}{8}\right), \cos\left(\frac{5\pi}{8}\right), \cos\left(\frac{3\pi}{8}\right), \cos\left(\frac{\pi}{8}\right) \right]$$

Evaluando numéricamente, obtenemos la plantilla base rígida:

$$\mathbf{C}_{base} = \left[ -0.9238795, -0.3826834, 0.3826834, 0.9238795 \right]$$

### Escalamiento Dinámico por Desviación Estándar ($\sigma$):
Durante el nacimiento o importación de la capa $L_i$ con desviación estándar de pesos $\sigma(W_{L_i})$, los centroides reales se escalan de la siguiente manera:

$$\mathbf{C}_{L_i} = \mathbf{C}_{base} \cdot \left(\sigma(W_{L_i}) \cdot \gamma\right)$$

Donde $\gamma = 2.2$ es el factor armónico de dispersión determinado por simulación Monte Carlo para minimizar el MSE de cuantización inicial.

---

## 2. Escenario BDD (Behavior-Driven Development)

El comportamiento esperado para este hito se formaliza en el siguiente escenario:

```gherkin
Característica: Inicialización Algebraica del Organismo Genómico
  Como motor de compresión semántica
  Quiero inicializar los centroides del modelo usando raíces ciclotómicas
  Para garantizar la coherencia atencional y evitar la deriva semántica inicial.

  Escenario: Carga de plantilla ciclotómica al nacer
    Dado que no existe un modelo pre-entrenado en la ruta destino
    Y el archivo "models/core/algebraic_codebook.json" contiene los centroides de Q(zeta_16)
    Cuando ejecuto el subcomando "gaje-cli --init models/silver_adult.gaje --preset silver_adult"
    Entonces el cargador nativo debe crear el archivo de base de datos
    Y todas las capas GenomicLinear del modelo deben tener sus centroides inicializados con la plantilla escalada
    Y el MSE de cuantización inicial respecto a la distribución teórica debe ser menor a 0.00005
```

---

## 3. Plan de Implementación Técnica (TDD)

1. **Paso Red (Fallo):**
   * Escribir un test unitario en `tests/unit/test_algebraic_init.py` que intente inicializar un modelo sin `algebraic_codebook.json` y verifique que el cargador falle o regrese los valores gaussianos tradicionales.
   * Verificar que se detecte una anomalía si los centroides no respetan la simetría de punto-reflexión.

2. **Paso Green (Paso):**
   * Modificar `init_born_genomic_model` en `src/io/loader.rs` para precargar y forzar la inyección de `algebraic_c` si está presente.
   * Ajustar la función de generación en `src/compute/math.rs` para realizar la asignación de bits Gray Code sobre la plantilla ciclotómica en lugar de intervalos uniformes.

3. **Refactor:**
   * Optimizar la velocidad de asignación bitwise utilizando registros SIMD en Rust.
   * Asegurar que el cargador libere memoria correctamente tras la transacción de inicialización en la base de datos `redb`.
