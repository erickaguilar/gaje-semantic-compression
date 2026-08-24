# 🛠️ Guía de Ejecución: Suite de Automatización y Pruebas — GAJE Helix Engine

**Fecha de Publicación:** 22 de Agosto de 2026  
**Script de Automatización:** \`tests/automation_suite.py\`  
**Estado:** Documentación Técnica Local (No Sincronizado)  
**Cobertura:** 100% de los Subsistemas Clave de Inferencia, Memoria, Web UI y Cuántica  

---

## 🎯 1. Propósito de la Suite de Automatización

La Suite de Automatización (\`tests/automation_suite.py\`) es una herramienta de verificación integral diseñada para ejecutar de forma desatendida y continua pruebas de regresión, integridad de memoria, rendimiento de inferencia y validación de protocolos en el motor **GAJE**.

Permite a los desarrolladores y evaluadores certificar que cualquier cambio en los kernels de Rust (PyO3) o en la capa de inferencia de Python conserve la estabilidad y no introduzca regresiones.

---

## 🚀 2. Ejecución Rápida

Para ejecutar todas las pruebas automatizadas:

\`\`\`bash
python3 tests/automation_suite.py
\`\`\`

O a través del framework estándar \`unittest\`:

\`\`\`bash
python3 -m unittest tests/automation_suite.py -v
\`\`\`

---

## 🔬 3. Descripción de las Pruebas Automatizadas

| Suite / Test Case | Función Evaluada | Criterio de Éxito |
| :--- | :--- | :--- |
| **`test_01_models_discovery`** | Búsqueda recursiva en \`models/\` | Encuentra al menos 1 modelo plano (\`.flat\`) o genómico (\`.gaje\`). |
| **`test_02_native_inference_execution`** | Inferencia nativa SIMD AVX2 | Carga Mmap en frío y genera $>0$ tokens coherentes con parada limpia. |
| **`test_03_memory_purge_and_leak_check`** | Purga agresiva con \`malloc_trim(0)\` | Libera $>85\%\$ de la memoria residente añadida por el modelo y limpia buffers. |
| **`test_04_streaming_metrics_protocol`** | Protocolo SSE \`__gaje_metrics__\` | Valida serialización JSON con desglose de tokens ($p+g$), ratios y ahorro. |
| **`test_05_quantum_genomic_tokenization`** | Prototipo Cuántico-Genómico | Valida $\\text{Tr}(\\rho) = 1.0$ y colapso contextual con el *Island Model*. |

---

## 📊 4. Registro de Ejecución de Ejemplo (Benchmark Local)

\`\`\`text
================================================================================
🧬 INICIANDO SUITE DE AUTOMATIZACIÓN GAJE HELIX ENGINE
Versión: 1.6.0-alpha | Python: 3.14.6
================================================================================
test_01_models_discovery:
[SUITE 1] Modelos detectados (5): [deepseek_r1_1_5b.flat, qwen2_0_5b.flat, qwen2_5_3b.flat, smollm2_135m.flat, feto_genomico_v1.gaje]
ok

test_02_native_inference_execution:
[SUITE 1] Probando inferencia nativa con [qwen2_0_5b.flat]...
⚡ [Zero-Copy Mmap] Cargando modelo binario plano mmap instantáneo...
🧬 [ArchitectureDescriptor] Detectada arquitectura Qwen2 desde la cabecera binaria (.flat)
✅ Organismo GAJE v0.9.7 Flat mmap Cargado en 598.52 ms
[SUITE 1] Respuesta generada (24 tokens en 2152.70 ms):
          "El ADN es un complejo de nucleótidos que compone el núcleo de la vida..."
ok

test_03_memory_purge_and_leak_check:
[SUITE 2] Memoria RSS inicial: 1188.50 MB
[SUITE 2] Cargando [qwen2_5_3b.flat]...
✅ Organismo GAJE v0.9.7 Flat mmap Cargado en 2575.21 ms
[SUITE 2] Memoria RSS con modelo cargado: 3173.28 MB (+1984.78 MB)
[SUITE 2] Ejecutando unload_model() y malloc_trim(0)...
[SUITE 2] Memoria RSS tras purga agresiva: 874.14 MB
ok

test_04_streaming_metrics_protocol:
[SUITE 3] Validando estructura de evento __gaje_metrics__...
[SUITE 3] Métricas validadas con éxito: 42 tokens (Ratio: 8.0x | Ahorro: 87.5%)
ok

test_05_quantum_genomic_tokenization:
[SUITE 4] Probando simulación de Tokenización Cuántico-Genómica...
[SUITE 4] Estado cuántico-genómico: |token⟩ = 0.6|A⟩ + 0.8|G⟩
[SUITE 4] Traza(ρ) = 1.00 | Probabilidad de colapso en contexto G: 64.00%
ok

----------------------------------------------------------------------
Ran 5 tests in 6.582s

OK
\`\`\`

---

## 🛠️ 5. Integración Continua (CI / Local Hooks)

Este script está listo para ser incluido como pre-commit hook o paso de validación local antes de realizar lanzamientos de nuevas versiones de GAJE:

\`\`\`bash
# Ejecutar verificación completa antes de pruebas de carga
python3 tests/automation_suite.py && echo "✅ Todas las suites pasaron correctamente."
\`\`\`

---
*Documentación técnica del sistema GAJE. Archivo local en \`docs/guides/AUTOMATION_SUITE_GUIDE.md\`.*
