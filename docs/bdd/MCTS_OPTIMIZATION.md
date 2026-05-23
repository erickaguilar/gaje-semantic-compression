# 🗣️ BDD: Escenario de Optimización MCTS

Este escenario define cómo debe comportarse el motor MCTS frente a un problema de cuantización complejo.

**Escenario: Superar la optimización Monte Carlo clásica**
*   **Given (Dado):** Un conjunto de pesos f32 con una distribución asimétrica y un motor MCTS configurado con $c_{puct} = 1.41$.
*   **When (Cuando):** Se ejecuta la búsqueda de centroides durante 10,000 iteraciones (mismo presupuesto que el script de Python).
*   **Then (Entonces):** El Error Cuadrático Medio (MSE) final debe ser al menos un 15% menor que el obtenido por la búsqueda aleatoria pura.
*   **And (Y):** El motor debe ser capaz de evaluar al menos 50,000 nodos por segundo en un solo núcleo.
