# 🧪 Reporte de Error: El Abismo de Vocabulario (Falla de Inferencia)

**Fecha:** 24 de mayo de 2026
**Estatus:** Crítico - Resuelto por re-escalado.
**Síntoma:** `Error: "Token id 28019 out of bounds"` durante el chat.

## 1. El Problema: Desincronización Estructural
Al ejecutar el Paso 5, el motor de inferencia colapsó. La causa es una discrepancia entre la arquitectura del genoma y el tokenizador integrado:
- **Genoma:** Inicializado con un `vocab_size` de **16,384** (límite del micro-organismo SMG-1).
- **Tokenizador:** El archivo `tokenizer.json` nativo (3.4 MB) utiliza un vocabulario de **49,152** tokens.
- **Resultado:** Cuando el usuario escribe palabras como "España" (ID 28,019), el motor busca una fila de pesos que no existe en el ADN del genoma, provocando un pánico de memoria.

## 2. Diagnóstico de la Meta de 4 MB
Este error revela por qué el archivo `.gaje` pesaba 23 MB en lugar de 4 MB:
1.  Un vocabulario de 49k tokens en 2 bits ocupa ~5 MB solo en la capa de embeddings.
2.  La arquitectura de 8 bloques añade ~15 MB adicionales.
3.  **Conclusión:** Para alcanzar los 4 MB reales, no podemos usar el tokenizador de SmolLM2 (49k). Necesitamos un **Micro-Tokenizador** de 16k tokens o aceptar un tamaño de ~12-15 MB para el Gold Embryo v1.

## 3. Solución Aplicada
Se ha decidido re-inicializar el Gold Embryo con `vocab_size: 49152` para garantizar la viabilidad funcional inmediata, sacrificando temporalmente la meta de almacenamiento estricta a favor de la coherencia semántica.

---
*Este hallazgo recalibra las expectativas de tamaño para la v1.0: la inteligencia requiere un léxico mínimo.*
