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
| Tipos soportados | `F32`/`F16`/`Q8_0` | Restringirse a esos tres (o fallar) |

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

## 8. Tareas abiertas

- [ ] Implementar `writer.rs` (esqueleto sin cuantización).
- [ ] Añadir `GGUFValue::value_type()` y helpers de escritura.
- [ ] Test de roundtrip read → write → read.
- [ ] Revisar si conviene exponer un helper público de quantización para F16/Q8_0.
