# 🧠 Investigación: Topología de Centroides y Grafos Semánticos (Fase 4.0)

## 1. Visión General
La Fase 4.0 propone una evolución disruptiva en el motor GAJE-Flow: pasar de centroides como valores estáticos de cuantización a **Centroides como Nodos de un Grafo Relacional**. Esta arquitectura permite que el modelo "recuerde" la estructura de los datos encontrados y facilite el traslado de conocimiento entre dominios (ej. de inglés a español o de lógica a lenguaje).

## 2. Hallazgos Teóricos
*   **Aislamiento de Señal:** Los centroides actuales son ciegos al contexto. Al transformarlos en grafos, cada centroide adquiere una "vecindad semántica", reduciendo la perplejidad al pre-activar nodos relacionados.
*   **Isomorfismo Semántico:** La estructura del lenguaje (su topología) es similar entre idiomas. Un grafo de centroides bien entrenado puede ser "trasladado" de un genoma a otro, actuando como un mapa de inteligencia pre-existente.
*   **Resistencia al Olvido:** Las aristas del grafo actúan como anclas (Anchors) que protegen las relaciones aprendidas durante el entrenamiento por resonancia.

## 3. Requerimientos para la Fase 4.0
Para implementar esta visión en Rust, se identifican los siguientes componentes necesarios:

### A. Matriz de Adyacencia Genómica (GAM)
*   **Estructura:** Una tabla de pesos $[C \times C]$ donde $C$ es el número de centroides.
*   **Función:** Almacena la fuerza de la relación (frecuencia de co-activación) entre estados de 2-bits.

### B. Motor de Inferencia por Propagación de Grafo
*   **Mecanismo:** El forward neuromórfico debe incluir un paso de "Pre-Spike Activation" basado en los vínculos del grafo.
*   **Optimización:** Uso de kernels de Rust para multiplicar la activación actual por la matriz de adyacencia en tiempo real (< 1ms).

### C. Algoritmo de Traslado (Transfer Learning)
*   **Proceso:** Capacidad de exportar la topología del grafo de un modelo maduro e inyectarla en un "Embrión" mediante `gaje-cli --inject-topology`.

## 4. Plan de Acción
1.  **Prototipado (Semana 1):** Definir la estructura `CentroidGraph` en `src/core/topology.rs`.
2.  **Integración (Semana 2):** Modificar `RustGenomicBlock` para consultar la matriz de adyacencia durante el forward.
3.  **Validación (Semana 3):** Re-ejecutar el test "Needle in a Haystack" para verificar si la topología del grafo ayuda a recuperar la aguja semántica.

---
*Documento de visión técnica para la evolución del Micro-Genoma GAJE.*
