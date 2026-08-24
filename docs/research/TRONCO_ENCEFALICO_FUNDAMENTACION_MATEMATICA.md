# 🧬 Fundamentación Matemática y Neuroanatómica del Tronco Encefálico en GAJE HELIX

> **Autor:** Erick Aguilar / GAJE Architecture Working Group
> **Fecha:** 22 de Agosto de 2026
> **Módulo:** GAJE-WASM (`src/wasm.rs`, `src/compute/island.rs`, `src/io/gmem.rs`)
> **Estado:** Documento Canónico de Arquitectura e Investigación

---

## 1. Introducción y Tesis Epistemológica

En la arquitectura **GAJE HELIX**, el **Tronco Encefálico (*Brainstem*)** designa formalmente la **capa de control involuntario, homeostático y sensoriomotor** que media de forma determinista entre los pesos congelados de la red neuronal profunda (el *Córtex*) y el entorno físico/digital (la *Periferia*).

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                     PERIFERIA (ENTORNO)                         │
 │     Eventos DOM, WebSockets, Streams de Audio, Señales REST      │
 └───────────────────────────────┬──────────────────────────────────┘
                                 │ Vía Aferente (Sensación)
                                 ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │                 TRONCO ENCEFÁLICO WASM (GAJE)                    │
 │                                                                  │
 │  1. Vías Aferentes:  Proyección Semántica a Espacio de Hilbert   │
 │                      Ingesta y Resonancia RAG Multi-Nicho        │
 │                                                                  │
 │  2. Sistema Reticular: Homeostasis y Ciclo de Sueño (Consolidación)│
 │                       Poda Sináptica de Redundancia (τ ≥ 0.95)   │
 │                       Persistencia Soberana Zero-Copy (.gmem v2) │
 │                                                                  │
 │  3. Vías Eferentes:  Muestreo Lagrangiano, Filtro de Penalización │
 │                      Emisión Motora Estructurada (Tool Calling)  │
 └───────────────────────────────▲──────────────────────────────────┘
                                 │ Vía Eferente (Acción / Logits)
                                 ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │                     CÓRTEX (RED NEURONAL)                        │
 │            Pesos Genómicos Congelados (.flat / Qwen / SmolLM)    │
 └──────────────────────────────────────────────────────────────────┘
```

### 1.1 Tesis de Necesidad Anatómica
Un Modelo de Lenguaje (LLM) aislado carece de:
1. **Homeostasis de Memoria:** No puede decidir cuándo consolidar o podar recuerdos sin reentrenamiento.
2. **Filtro Sensorial Autónomo:** No puede modular la resonancia de estímulos externos antes de su procesamiento.
3. **Conducta Involuntaria:** No puede ejecutar tareas de mantenimiento vital en segundo plano (background ticks) mientras no se le consulta.

El **Tronco Encefálico WASM** resuelve esta carencia operando como un **sistema nervioso autónomo en memoria lineal WebAssembly**.

---

## 2. Formulación Matemática de las Vías Aferentes (Sensación Ascendente)

Las vías aferentes reciben estímulos no estructurados del entorno $\mathcal{X}$ y los proyectan a un espacio euclídeo/hilbertiano $d$-dimensional $\mathcal{H}^d = \mathbb{R}^d$.

### 2.1 Proyección de N-Gramas a Espacio de Hilbert $\mathcal{H}^d$
Sea una secuencia textual sensorial $x \in \mathcal{X}$ dividida en una tupla de palabras normalizadas $\mathcal{W}(x) = (w_1, w_2, \dots, w_m)$. Se define el operador de proyección determinista $\phi: \mathcal{X} \to \mathbb{R}^d$:

$$\mathbf{u}(x) = \sum_{i=1}^m \mathbf{e}_{h(w_i)}$$

donde $h: \mathcal{W} \to \{0, 1, \dots, d-1\}$ es una función de dispersión hash FNV-1a criptográficamente balanceada:

$$h(w) = \left( \left( \bigoplus_{b \in w} b \right) \cdot p_{\text{prime}} \right) \pmod d$$

El vector sensorial aferente normalizado $\mathbf{v}(x) \in \mathbb{S}^{d-1}$ se obtiene aplicando la normalización $L_2$:

$$\mathbf{v}(x) = \frac{\mathbf{u}(x)}{\|\mathbf{u}(x)\|_2 + \epsilon} = \frac{\mathbf{u}(x)}{\sqrt{\sum_{j=1}^d u_j(x)^2 + \epsilon}}$$

### 2.2 Resonancia Semántica Multi-Nicho
La memoria del tronco encefálico está estratificada en $N$ nichos especializados:
- $\mathcal{M}_{\text{epi}}$: Memoria Episódica (hechos recientes y cronológicos).
- $\mathcal{M}_{\text{doc}}$: Memoria Documental (conocimiento semántico consolidado).
- $\mathcal{M}_{\text{conv}}$: Memoria Conversacional (diálogo activo).

Sea un vector de pesos de nicho $\mathbf{w} = (w_{\text{epi}}, w_{\text{doc}}, w_{\text{conv}}) \in \Delta^2$ tal que $\sum_{n} w_n = 1$ y $w_n \ge 0$. Para un estímulo sensorial de consulta $q$ con vector $\mathbf{v}_q$, el operador de resonancia $\mathcal{R}_k(q)$ recupera el conjunto de los $k$ recuerdos más resonantes:

$$\mathcal{R}_k(q) = \operatorname{arg\,top}_k \left( \bigcup_{n \in \{\text{epi, doc, conv}\}} \left\{ \left( e_i^{(n)}, w_n \cdot \mathcal{S}_C(\mathbf{v}_q, \mathbf{v}_i^{(n)}) \right) \right\} \right)$$

donde $\mathcal{S}_C(\mathbf{a}, \mathbf{b})$ es la similitud coseno vectorial pura:

$$\mathcal{S}_C(\mathbf{a}, \mathbf{b}) = \mathbf{a} \cdot \mathbf{b} = \sum_{j=1}^d a_j b_j \quad (\text{dado que } \|\mathbf{a}\|_2 = \|\mathbf{b}\|_2 = 1)$$

---

## 3. Formulación Matemática del Ciclo Autonómico (Sueño, Homeostasis y Poda)

Durante la fase de inactividad o en intervalos temporales periódicos (background Web Worker), el tronco encefálico ejecuta el **Ciclo de Sueño Autonómico**.

### 3.1 Consolidación y Transferencia de Memorias
Sean los conjuntos de memorias volátiles $\mathcal{M}_{\text{vol}} = \mathcal{M}_{\text{epi}} \cup \mathcal{M}_{\text{conv}}$ y la memoria consolidada de largo plazo $\mathcal{M}_{\text{doc}}$. El operador de consolidación $\mathcal{C}_{\tau}$ evalúa cada entrada $e \in \mathcal{M}_{\text{vol}}$ frente al corpus documental preexistente:

$$\mathcal{M}_{\text{doc}}^{(t+1)} = \mathcal{M}_{\text{doc}}^{(t)} \cup \left\{ e \in \mathcal{M}_{\text{vol}}^{(t)} \;\middle|\; \max_{d \in \mathcal{M}_{\text{doc}}^{(t)}} \mathcal{S}_C(\mathbf{v}_e, \mathbf{v}_d) < \tau_{\text{dedup}} \right\}$$

donde $\tau_{\text{dedup}} \in [0.90, 0.99]$ es el umbral de redundancia semántica (por defecto $\tau_{\text{dedup}} = 0.95$).

### 3.2 Operador de Poda Sináptica (*Synaptic Pruning*)
El operador de poda elimina las conexiones y recuerdos colineales que superan el umbral de redundancia:

$$\mathcal{P}_{\tau}(\mathcal{M}) = \left\{ x_i \in \mathcal{M} \;\middle|\; \forall j < i, \; \mathcal{S}_C(\mathbf{v}_i, \mathbf{v}_j) < \tau \right\}$$

**Teorema de Bounded Memory Footprint:**
Si la esfera unitaria $\mathbb{S}^{d-1}$ se recubre con esferas de radio angular $\theta = \arccos(\tau_{\text{dedup}})$, el número máximo de recuerdos almacenables en $\mathcal{M}_{\text{doc}}$ está acotado superiormente por el número de empaquetamiento esférico (*spherical packing number*):

$$|\mathcal{M}_{\text{doc}}| \le \mathcal{N}(\mathbb{S}^{d-1}, \theta) \approx \left( \frac{c}{\theta} \right)^d$$

Esto garantiza que la memoria no crezca indefinidamente en el navegador, preservando la estabilidad de la RAM lineal de WebAssembly.

---

## 4. Formulación Matemática de las Vías Eferentes (Control Motor y Actuación)

Las vías eferentes transforman los logits producidos por el Córtex en acciones motoras y llamadas a herramientas (*tool calls*) conformes al esquema de la periferia.

### 4.1 Muestreo Condicionado y Modulación de Temperatura
Dado un vector de logits no normalizados $\mathbf{z} \in \mathbb{R}^{|V|}$ emitido por el modelo en el paso autorregresivo $t$, el tronco encefálico aplica modulación por temperatura $\mathcal{T}$ y penalización de repetición $\rho(v)$:

$$P(y_t = v \mid y_{<t}) = \frac{\exp\left( \frac{z_v}{\mathcal{T} \cdot \rho(v)} \right)}{\sum_{u \in V} \exp\left( \frac{z_u}{\mathcal{T} \cdot \rho(u)} \right)}$$

donde:
$$\rho(v) = \begin{cases} \alpha_{\text{rep}} & \text{si } v \in y_{<t} \text{ y } z_v > 0 \\ \frac{1}{\alpha_{\text{rep}}} & \text{si } v \in y_{<t} \text{ y } z_v \le 0 \\ 1.0 & \text{en otro caso} \end{cases}$$

### 4.2 Actuación Motora Estructurada
La función de decisión motora proyecta la secuencia de tokens generada al conjunto de herramientas válidas $\mathcal{T}_{\text{tools}}$:

$$\mathbf{a} = \operatorname{Actuate}(y_{1:T}, \mathcal{S}_{\text{tools}}) = \begin{cases} \operatorname{JSON\_Decode}(y_{1:T}) & \text{si } \operatorname{ValidateSchema}(y_{1:T}, \mathcal{S}_{\text{tools}}) = \text{True} \\ \text{null} & \text{en caso de fallo sintáctico} \end{cases}$$

---

## 5. Tabla Comparativa: Neuroanatomía Biológica vs. GAJE HELIX WASM

| Estructura Biológica | Función Fisiológica Humana | Equivalente en GAJE HELIX WASM | Archivo / Función en Código |
|:---|:---|:---|:---|
| **Corteza Cerebral (*Cortex*)** | Razonamiento de alto nivel, lenguaje y patrones declarativos complejos. | Red Neuronal LLM con pesos estáticos congelados. | [`src/nn/llm.rs`](file:///home/erickaguilar/Documentos/gaje-semantic-compression/src/nn/llm.rs) (`GenomicLLM`) |
| **Tracto Espinotalámico (Vías Aferentes)** | Conducción, filtrado y normalización de estímulos periféricos al encéfalo. | Proyección hash de texto e ingesta en islas de memoria. | [`src/wasm.rs`](file:///home/erickaguilar/Documentos/gaje-semantic-compression/src/wasm.rs#L58-L100) (`ingest_sensory`, `text_to_embedding`) |
| **Tálamo / Resonancia Asociativa** | Selección de señales relevantes para alimentar la atención consciente. | Búsqueda por Similitud Coseno multi-nicho ponderada. | [`src/wasm.rs`](file:///home/erickaguilar/Documentos/gaje-semantic-compression/src/wasm.rs#L102-L135) (`retrieve_context`) |
| **Formación Reticular / Ciclo Sueño** | Homeostasis, consolidación nocturna de recuerdos y poda sináptica. | Transferencia de memoria episódica a documental y poda $\ge 0.95$. | [`src/wasm.rs`](file:///home/erickaguilar/Documentos/gaje-semantic-compression/src/wasm.rs#L137-L148) (`autonomic_sleep_cycle`) |
| **Cuerpo Estriado / Vías Corticoespinales** | Conversión de la intención cognitiva en señales motoras ejecutables. | Formateo y validación de Tool Calls estructurados en JSON. | [`src/wasm.rs`](file:///home/erickaguilar/Documentos/gaje-semantic-compression/src/wasm.rs#L222-L235) (`actuate`) |
| **Lóbulo Temporal Medial (Memoria de Hábito)** | Almacenamiento persistente determinista de memorias no volátiles. | Archivos binarios zero-copy `.gmem` v2 en IndexedDB / OPFS. | [`src/io/gmem.rs`](file:///home/erickaguilar/Documentos/gaje-semantic-compression/src/io/gmem.rs) (`GmemMemoryIndex`) |

---

## 6. Garantías de Complejidad y Rendimiento

1. **Ingesta Sensorial ($O(m + d)$):** Ingesta determinista donde $m$ es la longitud del texto y $d$ es la dimensión del embedding (típicamente $d=576$ o $896$).
2. **Resonancia Semántica ($O(K \cdot d)$):** Búsqueda vectorial SIMD lineal sobre las $K$ entradas de las islas.
3. **Poda Sináptica en Ciclo de Sueño ($O(E \cdot D \cdot d)$):** Ejecutada en background dentro de un Web Worker dedicado, sin bloquear el hilo de renderizado del navegador.
4. **Paridad Bit a Bit:** Determinismo idéntico certificado entre la ejecución nativa en CPU/Rust y la ejecución en el motor WebAssembly.

---

## 7. Referencias Teóricas

1. **Friston, K.** (2010). *The free-energy principle: a unified brain theory?* Nature Reviews Neuroscience, 11(2), 127-138.
2. **Tononi, G., & Cirelli, C.** (2014). *Sleep and the price of plasticity: from synaptic and cellular homeostasis to memory consolidation and integration.* Neuron, 81(1), 12-34.
3. **Hebb, D. O.** (1949). *The Organization of Behavior: A Neuropsychological Theory.* John Wiley & Sons.
4. **Kandel, E. R., et al.** (2021). *Principles of Neural Science (6th Edition).* McGraw Hill.
5. **GAJE Helix Master Plan:** [`docs/plans/WASM_BRAINSTEM_PLAN.md`](file:///home/erickaguilar/Documentos/gaje-semantic-compression/docs/plans/WASM_BRAINSTEM_PLAN.md).
