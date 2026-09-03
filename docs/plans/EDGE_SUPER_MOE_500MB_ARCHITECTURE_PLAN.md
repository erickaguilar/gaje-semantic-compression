# 🧬 Plan Arquitectónico: Super-MoE Edge 500 MB (`gaje_prime_moe_500m.flat`)

> **Fecha:** 3 de Septiembre de 2026  
> **Versión:** `GAJE Helix v1.8.0 / Super-MoE Edge Platform`  
> **Estado:** 📝 `PROPUESTA DE INGENIERÍA Y ESPECIFICACIÓN ARQUITECTÓNICA`  
> **Ámbitos:** Mixture of Experts (MoE 8-Expertos) · Enrutamiento Top-2 Zero-Copy · Memoria Congénita Extensa · Despliegue Local Soberano  
> **Módulos Directos:** `src/compute/graph.rs`, `src/io/flat_header.rs`, `src/compute/island.rs`, `src/nn/block/`

---

## 1. 🎯 Visión y Objetivos

El **Super-MoE Edge de 500 MB** representa el punto de equilibrio óptimo (*sweet spot*) entre **compresión extrema, capacidad cognitiva multimodal y eficiencia en silicio de borde (móviles, tablets, navegadores y microcomputadores)**.

### Objetivos Principales:
1. **Capacidad Paramétrica Virtual de 2.2B con Cómputo de 350M:**
   Banco de **8 expertos especializados** donde cada token activa únicamente los **2 expertos más resonantes (Top-2)**.
2. **Vocabulario Humano Extendido (16K tokens):**
   Soporte nativo y sin fragmentación excesiva para español, inglés, código de programación y estructuración de datos JSON/BSON.
3. **Memoria Congénita Integrada de 25 MB (`.gmem` v2):**
   Almacén fáctico, episódico y conversacional indexado en 576 dimensiones con arranque en frío submilisegundo ($< 0.15\text{ ms}$).
4. **Throughput de Inferencia:**
   $\ge 25 - 40\text{ tok/s}$ en un solo hilo de CPU y $\ge 80 - 120\text{ tok/s}$ con aceleración WebGPU/Vulkan.

---

## 2. 📦 Presupuesto y Distribución de Memoria (< 500 MB)

```
┌────────────────────────────────────────────────────────────────────────┐
│        PAQUETE SUPER-MOE EDGE: gaje_prime_moe_500m.flat (~487 MB)      │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Tokenizador GTOK 16K + Embeddings FP32 compartidos:        ~48.0 MB │
│ 2. Atención Base Compartida (16 capas, D=576, H=8):            ~42.5 MB │
│ 3. Router de Fase Cuántico / Soft-Gating Top-2:                 ~3.5 MB │
│ 4. Hipocampo Congénito Extendido (.gmem 3 nichos fácticos):    ~25.0 MB │
│ 5. BANCO DE 8 EXPERTOS FFN (SwiGLU Q4_0 / BF2 Híbrido):                │
│    • Exp 1: Diálogo, Personalidad y Empatía                    ~40.0 MB │
│    • Exp 2: Lógica Formal y Matemáticas                        ~40.0 MB │
│    • Exp 3: Generación de Código y APIs                        ~40.0 MB │
│    • Exp 4: Tool-Calling y Estructuración JSON/BSON            ~40.0 MB │
│    • Exp 5: Ciencia, Geografía y Hechos del Mundo              ~40.0 MB │
│    • Exp 6: Traducción y Adaptación Cultural                   ~40.0 MB │
│    • Exp 7: Memoria y Búsqueda en Documentos Locales           ~40.0 MB │
│    • Exp 8: Crítico de Seguridad, Paridad y Anti-Loops         ~40.0 MB │
│ 6. Cabeza de Salida Compartida (lm_head FP32):                 ~48.0 MB │
├────────────────────────────────────────────────────────────────────────┤
│ 📊 TAMAÑO TOTAL DEL PAQUETE COMPLETO:                         ~487.0 MB │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 🏛️ Arquitectura del Flujo de Inferencia MoE Zero-Copy

```
                          [ Token de Entrada x_t ]
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │    Embeddings Compartidos   │
                      │    (16K Vocabulario FP32)   │
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │   16 Capas Atención Base    │
                      │     (D=576, RoPE Split)     │
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │   Router Gating (Softmax)   │
                      │   Calcula Top-2 Expertos    │
                      └──────────────┬──────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼ (Peso g_a)                            ▼ (Peso g_b)
      ┌─────────────────────┐                 ┌─────────────────────┐
      │  Experto A (FFN_a)  │                 │  Experto B (FFN_b)  │
      │  Offset &db[ptr_a]  │                 │  Offset &db[ptr_b]  │
      └──────────┬──────────┘                 └──────────┬──────────┘
                 │                                       │
                 └───────────────────┬───────────────────┘
                                     │
                                     ▼  y = g_a · FFN_a(x) + g_b · FFN_b(x)
                      ┌─────────────────────────────┐
                      │   Interferencia .gmem RAG   │
                      │   (Inyección <0.12 ms)      │
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │   lm_head FP32 Compartido   │
                      │   (Argmax / Sampler Rust)   │
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                            [ Siguiente Token ]
```

---

## 4. ⚡ Innovaciones Clave del Diseño

1. **Mapeo Virtual Unificado (`FlatHeaderV2` con Tabla de Expertos):**
   * El archivo `.flat` contiene un índice de desplazamiento de 64 bytes por experto.
   * El cambio de un experto a otro no genera asignaciones en el montículo (*heap allocations*), solo lectura de punteros de memoria ya mapeados (`mmap`).
2. **Desacoplamiento Modular de Especialidades:**
   * El experto en código (Exp 3) puede refinarse mediante destilación o afinamiento DNI sin alterar los pesos del experto en diálogo (Exp 1), impidiendo el olvido catastrófico cruzado.
3. **Escala de Cómputo Dinámica:**
   * Para consultas sencillas (saludos, respuestas cortas), el Router puede colapsar a **Top-1 activo**, reduciendo el consumo de cómputo a menos de **180M de parámetros efectivos**.

---

## 5. 📅 Fases de Implementación Técnica

| Fase | Duración | Entregables | Criterio de Éxito |
| :--- | :---: | :--- | :--- |
| **Fase 1: Descriptor MoE en Cabecera** | 2 días | Extensión en `src/io/flat_header.rs` (`MoeDescriptor`, offsets de 8 expertos) | Validación estructural con `gaje-cli audit` |
| **Fase 2: Bloque SwiGLU MoE en Rust** | 3 días | Implementación de `GenomicMoeLayer` en `src/nn/block/moe.rs` | Enrutamiento Top-2 en $< 5\text{ µs}$ |
| **Fase 3: Tokenizador GTOK 16K** | 2 días | Compilación de `data/gtok_human_16k.bin` con vocabulario enriquecido | Preservación de sintaxis en español y código |
| **Fase 4: Exportador y Empaquetado** | 3 días | Script `scripts/export/export_moe_500m.py` | Generación del archivo `.flat` $\le 490\text{ MB}$ |
| **Fase 5: Certificación y Benchmarks** | 2 días | Evaluación completa de PPL, coherencia y velocidad | PPL $< 6.5$, Throughput $> 25\text{ tok/s}$ |

---

## 6. 🧪 Escenarios BDD de Verificación

```gherkin
Característica: Super-MoE Edge de 500 MB
  Como motor nativo GAJE Helix
  Quiero orquestar un modelo MoE de 8 expertos en un binario de 500 MB
  Para brindar alta coherencia multimodelo en hardware edge sin saturar la RAM

  Escenario: Enrutamiento Top-2 sin copias de memoria
    Dado el archivo "models/production/gaje_prime_moe_500m.flat" mapeado en memoria
    Cuando se procesa un token con el bloque GenomicMoeLayer
    Entonces el router selecciona exactamente los 2 expertos con mayor probabilidad
    Y la ejecución FFN accede directamente a los offsets binarios sin asignaciones en el heap
    Y el tiempo de despacho del router es inferior a 10 microsegundos

  Escenario: Preservación de la huella en disco y RAM
    Dado el empaquetado del superorganismo MoE
    Cuando se audita con "gaje-cli audit"
    Entonces el tamaño total en disco es menor a 500 MB
    Y el consumo de memoria física viva (RSS) no excede 520 MB durante la inferencia
    Y el modelo genera respuestas coherentes tanto en conversación como en código
```
