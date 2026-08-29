# GGUF Writer — Plan de implementación

> Estado: **pendiente de implementar**. Plan de exploración acordado; aún no hay código.
> Contexto: tras refactorizar `io/gguf.rs` en submódulos, surge la idea del inverso del lector.

## 1. Objetivo

Implementar `GGUFWriter`, el inverso de `io/gguf/reader.rs`, capaz de serializar un
archivo GGUF en formato binario (v2/v3): cabecera, metadatos key-value, información
de tensores y datos crudos con alineación correcta.

## 2. Ubicación

- Nuevo submódulo: `src/io/gguf/writer.rs`.
- Re-exportado desde `src/io/gguf/mod.rs`:
  ```rust
  pub mod writer;
  pub use crate::io::gguf::writer::*;
  ```
- Reutiliza los tipos existentes de `src/io/gguf/types.rs` (`GGUFValue`, `GGMLType`,
  `GGUFTensorInfo`) **sin modificarlos**.

## 3. Brecha actual en `types.rs`

El reader **descarta** valores que el writer necesita reconstruir fielmente:

| Dato | Estado actual | Necesidad del writer |
|---|---|---|
| Alineación de tensores | Solo `general.alignment` | Aplicar padding a 32 bytes entre tensores y conocer `tensor_count`/`metadata_kv_count` |
| Offset de cada tensor | Implícito (lo lee el reader) | Calcular con `data_offset` al escribir |
| Tipos soportados | `F32`/`F16`/`Q8_0` | Restringirse a esos tres (o fallar); ver §8 para ampliación Q4_K |

> **Vinculación con soporte DeepSeek:** el reader actualmente solo acepta
> `F32`/`F16`/`Q8_0` (`src/io/gguf/reader.rs`). DeepScaleR/DeepSeek se distribuyen
> habitualmente en `Q4_K`/`Q4_K_M`, por lo que ampliar `GGMLType` y el layout de
> bloques K-quant es requisito para el plan
> [`docs/deepseek-gemma-support-plan.md`](deepseek-gemma-support-plan.md). El writer
> hereda esa misma restricción: escribir `Q4_K` exige implementar el formato de
> bloque K-quant (subgrupos + cuantización por canal).

## 4. Estructura propuesta

```rust
pub struct GGUFWriter {
    metadata: Vec<(String, GGUFValue)>,     // orden estable (HashMap pierde orden)
    tensors: Vec<GGUFTensorInfo>,           // orden de escritura
    tensor_data: Vec<Vec<u8>>,              // bytes crudos por tensor, en orden
}
```

Métodos:

- `new() -> Self`
- `add_metadata(key: impl Into<String>, value: GGUFValue)`
- `add_tensor(name, shape, tensor_type, data: Vec<u8>) -> Result<()>` (valida shape/tipo soportado)
- `write(&self, w: impl Write) -> std::io::Result<()>`:
  1. Magic `b"GGUF"` + version `3` + `tensor_count` + `metadata_kv_count`
  2. Metadatos (key, type id, value) — lógica inversa de `read_value`
  3. Tensor infos (name, n_dims, shape, type id, offset **calculado**)
  4. Datos con padding a 32 bytes desde `data_offset` (según `general.alignment`)

## 5. Complementos necesarios

- `GGUFValue::value_type() -> u32` (inverso de `read_value`).
- Helpers `write_value`, `write_string`, `write_u32`, `write_u64` (inverso de las
  privadas del reader).
- `tensor_size_bytes` (ya existe en el reader como privado) — duplicarlo como
  `pub(crate)` en el writer para cálculo de offsets.
- Cálculo de offset por tensor acumulando tamaño + padding de alineación.

## 6. Cuantización (fuera del alcance inicial)

- Escribir F32 es trivial (reinterpretar `&[f32]` → bytes).
- F16/Q8_0 requieren quantizar; el código ya existe en `compute/quantize/*`
  (`Q4_0Block`/`Q8_0Block` en `io/header/blocks.rs`).
- Se puede enganchar luego sin cambiar la API del writer: basta pasar `Vec<u8>`
  ya cuantizado.

## 7. Verificación

- `cargo check` + `cargo check --features python` (sin warnings).
- Test roundtrip en `gguf/writer.rs` (o `gguf/tests.rs`):
  1. Leer un modelo con `GGUFReader`.
  2. Escribir con `GGUFWriter`.
  3. Releer el resultado y comparar `GGUFValue`s y bytes de tensores.

## 8. Ampliación de tipos (Q4_K) para DeepSeek

El layout K-quant de GGUF divide cada bloque (p.ej. 256 elementos) en **subgrupos**
(16) con una escala `f16` por subgrupo, más una escala/offset de bloque y pesos
redistribuidos de 6 bits/8 bits. Ampliación propuesta:

- En `gguf/types.rs`: añadir variantes a `GGMLType` (`Q4_K`, `Q4_K_M`, etc.) con su
  `tensor_type_id` real (`gguf` llama.cpp).
- En `gguf/reader.rs`: mapear los `type_id` de K-quant y calcular
  `tensor_size_bytes` según el layout de bloques/subgrupos.
- En `io/header/blocks.rs` (opcional, formato `.gaje`): definir `Q4_KBlock` para
  persistir el mismo layout; de lo contrario, re-cuantizar a Q8_0/Q4_0 al importar.
- Reutilizable por el writer: la cuantización K-quant por subgrupo es la función
  inversa de la dequantización, así que implementarla una sola vez.

> **Decisión abierta:** importar DeepSeek directamente en `Q4_K` (ampliar el reader)
> vs. re-cuantizar desde `F16` a `Q8_0`/`Q4_0` al importar (evita tocar K-quant).
> La primera preserva fidelidad; la segunda es menos esfuerzo y reusa lo existente.

## 9. Tareas abiertas

- [ ] Implementar `writer.rs` (esqueleto sin cuantización).
- [ ] Añadir `GGUFValue::value_type()` y helpers de escritura.
- [ ] Test de roundtrip read → write → read.
- [ ] Revisar si conviene exponer un helper público de quantización para F16/Q8_0.
- [ ] (Depende de soporte DeepSeek) decidir: ampliar `GGMLType` a `Q4_K` o re-cuantizar
      desde `F16` → `Q8_0`/`Q4_0` al importar.
