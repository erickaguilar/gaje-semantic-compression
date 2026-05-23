# 🗣️ BDD: Behavior-Driven Development en GAJE-Flow

El desarrollo basado en comportamiento (BDD) complementa nuestra suite de TDD asegurando que el motor genómico cumpla con las expectativas funcionales y de rendimiento desde una perspectiva de usuario.

## 1. Estructura de Escenarios (Gherkin)

Utilizamos el formato **Given-When-Then** para definir el comportamiento esperado.

### Ejemplo: Inferencia de Alta Velocidad
*   **Given (Dado):** Un motor genómico inicializado con arquitectura SoA y pesos de 2 bits.
*   **When (Cuando):** Se procesa un flujo de entrada de 10,000 eventos neuromórficos.
*   **Then (Entonces):** El tiempo de procesamiento total debe ser inferior a 50ms (SIMD NEON optimizado).

### Ejemplo: Estabilidad de Memoria
*   **Given (Dado):** Un sistema con recursos limitados (entorno móvil/edge).
*   **When (Cuando):** Se realiza una clonación de identidad de gran escala.
*   **Then (Entonces):** El consumo de memoria f32 no debe exceder los límites predefinidos en el `forward`.

## 2. Relación con TDD
Mientras que el **TDD** asegura que el código sea correcto técnicamente (test unitarios), el **BDD** asegura que el sistema se comporte como se espera en escenarios del mundo real (test de integración y aceptación).

## 3. Implementación en el Repositorio
Los escenarios definidos aquí deben ser validados mediante los tests de integración ubicados en `tests/integration/`.
