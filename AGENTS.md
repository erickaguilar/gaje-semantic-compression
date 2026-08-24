# 🧬 GAJE Semantic Compression — Protocolo Global de Desarrollo y Guía para Agentes

Este archivo define la **descripción global del proyecto**, la arquitectura del repositorio y los **estándares operativos y técnicos obligatorios** tanto para desarrolladores como para agentes de Inteligencia Artificial (Antigravity, Claude, Gemini, Copilot, etc.).

---

## 1. Descripción Global del Proyecto

**GAJE (Genetic Adaptive Joint Embedding / DNA Semantic Compression)** es un framework híbrido de compresión semántica y memoria genética neuronal. Su objetivo es comprimir y recuperar representaciones semánticas densas a nivel genómico/neuronal con latencias ultrabajas, combinando algoritmos genéticos, memoria mmap zero-copy e inhibición lateral K-WTA.

### Componentes Clave:
* **Núcleo Nativo (Rust):** Implementación de máxima eficiencia para compresión/descompresión, búsqueda vectorial, mmap y kernels matemáticos.
* **Capa de Investigación (Python):** Interfaz, embeddings, integraciones con PyTorch/HuggingFace y herramientas de experimentación.
* **Interfaz Web (Web UI & Sistema de Temas):** Panel interactivo con chat, telemetría HUD, visualizador de arquitectura y documentación.

---

## 2. Mapa y Arquitectura del Repositorio

* **`/src`**: Núcleo nativo en Rust (librería `gaje-core`, CLI `gaje-cli`, kernels SIMD, estructuras de datos).
* **`/python`**: Módulo Python `gaje` y bindings hacia el núcleo nativo.
* **`/examples/ui/web_ui`**:
  * `index.html`: Chat interactivo y HUD de telemetría de compresión.
  * `docs.html`: Centro de documentación interactiva.
  * `architecture.html`: Grafo y visualización del sistema.
  * `server.py`: Servidor backend FastAPI para la Web UI.
  * `static/`: Estilos (`css/base.css`, `css/chat.css`), scripts e iconos del sprite Y2K (`static/icons/y2k/sprite.svg`).
* **`/docs`**: Documentación clasificada (`guides/`, `plans/`, `bdd/`, `reports/`, `meta/`, `research/`).
* **`/tests`**: Pruebas unitarias, de integración y de métricas (`pytest`, `cargo test`).
* **`/benchmarks`**: Suites de evaluación de latencia, throughput y entropía.
* **`/data`**: Modelos, datasets, logs y artefactos genómicos.

---

## 3. Reglas de Oro para Desarrolladores y Agentes de IA

### A. Soberanía del Núcleo y Gestión de Memoria (Rust & Python)
1. **Memoria Eficiente:** Prohibidas las pre-asignaciones masivas innecesarias de tensores en loops críticos. Priorizar punteros, referencias y memoria compartida zero-copy (`Arc<Vec<u8>>`, `mmap`).
2. **Soberanía Nativa:** Funcionalidades de alto rendimiento o herramientas CLI administrativas deben implementarse como comandos en `gaje-cli` (Rust), evitando scripts monolíticos descartables.
3. **Sin Colisiones de Módulos:** En `python/gaje/`, no crear carpetas que colisionen con extensiones nativas binarias (`_impl`).

### B. Sistema de Diseño Y2K & Dual-Theme

#### 1. Definición y Filosofía de los Temas:
* **`y2k-dark = 'HIG-APPLE'` (Tema Oscuro por Defecto):**
  * **Concepto:** Intersección entre los materiales oscuros de Apple Human Interface Guidelines (HIG) y el futurismo ciberpunk de la era Y2K (Web 1.0). Crea una atmósfera de consola de investigación genómica avanzada y terminal de alta fidelidad con cero fatiga visual.
  * **Fondo y Paneles:** Fondo negro absoluto (`#000000`), paneles en carbón translúcido (`#1c1c1e`, `#2c2c2e`) y `backdrop-filter: blur(20px)` (Glassmorphism de grado OS).
  * **Acentos Neón & ADN:** Azul Eléctrico (`#0a84ff`), Violeta (`#5e5ce6` / `#a78bfa`), Cian terminal (`#22d3ee`), Rosa Neón (`#f472b6`) y Verde Matrix (`#30d158`). Bases nitrogenadas de ADN: A (`#ff453a`), C (`#0a84ff`), G (`#30d158`), T (`#ffd60a`).
  * **Capas Retro:** Scanlines CRT generadas por CSS (`repeating-linear-gradient`), sheen diagonal reflectivo (115°) y cursor de terminal interactivo con parpadeo continuo (`brand-underscore`).
* **`y2k-light = 'SCANDINAVIAN-DESIGN'` (Tema Claro vía `[data-theme="light"]`):**
  * **Concepto:** Basado en los principios del diseño nórdico/escandinavo (funcionalismo democrático, minimalismo cálido *hygge*, conexión con la naturaleza y maximización de la luz), fusionado con el concepto de **Cuaderno de Campo & Aprendizaje Continuo (Lab Research Notebook)**. Refleja visualmente que el modelo es un organismo en constante aprendizaje y consolidación de memoria semántica.
  * **Geometría Cuadrada (0px radius):** En `y2k-light`, toda la interfaz (fichas de notas, burbujas, botones, contenedores, dropdowns y badges) es estrictamente rectangular con esquinas a 90 grados (`border-radius: 0px`), evocando fichas de archivo técnico y cuadernos de laboratorio de ingeniería.
  * **Fondo, Papel y Cuaderno:** Fondo marfil suave / pergamino limpio (`#f6f5f3` / `#edebe9`) con sutil cuadrícula punteada (*dot-grid* 24px) y línea guía de margen ámbar/carmesí; paneles y tarjetas de respuesta en blanco marfil estructurado (`#ffffff`) como fichas de notas de laboratorio encuadernadas.
  * **Acentos Orgánicos & ADN:** Verde Bosque / Jade Profundo (`#2c5234`), Pizarra (`#2c3539`), Ámbar botánico de notas (`#b45309`) y Carmesí (`#b91c1c`). Módulo de razonamiento desplegable (*thought disclosure*) estilizado como un memorando de investigación (*Field Notes*).
  * **UX & Ergonomía:** Tipografía geométrica limpia (*Inter* / *Plus Jakarta Sans*), micro-bordes neutros de alta legibilidad, iconografía adaptada con filtros invertidos de alto contraste y cero fatiga visual en entornos diurnos.

#### 2. Reglas Técnicas de Frontend para Agentes y Desarrolladores:
* **Regla de Oro de Overflow:** `overflow: hidden` está **estrictamente prohibido** en `.y2k-header` para evitar recortar menús desplegables y tooltips flotantes.
* **Jerarquía Z-Index:**
  * `z-index: 1` y `2`: Efectos de scanlines CRT y sheen de vidrio (`::before` / `::after`).
  * `z-index: 3`: Contenido principal de la barra `.wrap`.
  * `z-index: 200`: Dropdown de menú (`.y2k-menu-dropdown`) y modales (`.y2k-apple-modal`).
* **Botones Bevel 3D:** Mantener el efecto Web 1.0 con sombras interiores (`box-shadow: inset 1px 1px 0 rgba(255,255,255,.18), inset -1px -1px 0 rgba(0,0,0,.35)`) que se invierten al presionar (`:active`).

### C. Verdad Empírica y Certificación
1. **Compilación no equivale a éxito semántico:** Que el código compile no certifica la precisión de compresión. Las validaciones de Perplejidad (PPL) y distancia semántica deben verificarse formalmente.
2. **Ciclo de Desarrollo:** Diseñar bajo SDD (especificaciones) -> BDD (escenarios *Given-When-Then*) -> TDD (tests unitarios/integración).

---

## 4. Comandos Frecuentes de Desarrollo

```bash
# Compilar núcleo nativo en Rust
cargo build --release

# Ejecutar suite de pruebas de Rust
cargo test

# Ejecutar tests de Python
pytest tests/

# Iniciar servidor de la Web UI
python -m uvicorn examples.ui.web_ui.server:app --reload --port 8000
```

---

## 5. Estándar de Commits
Usar **Conventional Commits**:
* `feat(modulo):` Nueva característica
* `fix(modulo):` Corrección de errores
* `perf(modulo):` Optimización de rendimiento
* `docs(modulo):` Actualizaciones en documentación
* `style(ui):` Ajustes visuales o de diseño (Y2K / CSS)
* `refactor(modulo):` Refactorización de código sin cambio de comportamiento
