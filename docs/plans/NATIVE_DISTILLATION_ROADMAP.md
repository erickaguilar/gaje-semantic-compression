# 🗺️ Roadmap: Crianza Nativa y Destilación Soberana

Este documento detalla los pasos exactos para el desarrollo del modelo GAJE tras el pivote estratégico de junio de 2026. Abandonamos la importación masiva en favor de la **Crianza Progresiva**.

## Paso 1: Concepción del Estudiante (Born-Genomic)
En lugar de importar pesos, creamos un cuerpo vacío optimizado para el dispositivo objetivo.
*   **Acción:** `gaje-cli --init models/born/student_v1.gaje --preset silver_fetus`
*   **Razón:** Evita heredar el ruido de cuantización de un modelo maestro grande.

## Paso 2: La Gran Destilación (Nivel 2)
Entrenar al estudiante usando un "Consejo de Profesores" (Modelos GGUF potentes) que corrigen sus pesos de 2 bits en tiempo real.
*   **Acción:** Ejecutar `micro-distiller` con el `Mosaic Dataset` (420MB).
*   **Parámetros Críticos:**
    *   `--epochs 50`: Para asegurar que la gramática se asiente.
    *   `--lr 0.005`: Tasa de aprendizaje baja para no romper las anclas.
*   **Meta:** Alcanzar un PPL < 15.0.

## Paso 3: Afinación de Resonancia (Nivel 1)
Una vez que el modelo sabe hablar (L2), probamos su capacidad de memoria masiva.
*   **Acción:** Ejecutar `scripts/benchmarks/needle_haystack.py`.
*   **Meta:** 100% de precisión en recuperación de datos en 128k tokens.

## Paso 4: Ingesta DNI (Nivel 3)
Inyectar conocimiento específico (documentación técnica, libros, leyes) directamente en el genoma sin re-entrenar.
*   **Acción:** `gaje-cli ingest --model <path> --file <conoce.txt>`

## Paso 5: Certificación y Despliegue (L4 y L5)
Validación final de consumo energético y soberanía nativa.
*   **Acción:** Firma del `CERTIFICATION_REPORT_V1.5.md`.

---
*Este roadmap es el único camino validado para alcanzar un modelo de 10MB con inteligencia real en Android.*
