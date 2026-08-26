# Plan: Soporte de arquitectura DeepSeek (MLA/MoE) y Google Gemma

> **Estado:** propuesto (no implementado).
> **Contexto:** `gaje-cli` hoy solo soporta arquitecturas **GQA densas** (Llama,
> Qwen2/Qwen2.5, SmolLM). DeepScaleR/DeepSeek usan **MLA + MoE**; Gemma usa
> **GQA con RoPE interleaved** y (Gemma2) FFN alternado. Este plan dimensiona el
> trabajo para habilitar ambas familias.

## 1. Diagnóstico del estado actual (verificado)

### 1.1 Lo que ya existe
- `io/arch.rs` ya enumera `ModelFamily::Gemma = 5` y detecta `gemma` por nombre
  (`arch.rs:127-129`); `io/header/flat.rs` mapea `arch_family` y aplica
  `rope_base=10000`, `rope_style="split"`, `ffn_act="geglu"` para Gemma.
- `nn/attention.rs` implementa GQA + RoPE con dos estilos de rotación: `"split"`
  (Llama/Qwen) e interleaved (caso por defecto), con `rope_base` configurable.
- `nn/block` construye FFN denso `gate/up/down` (SwiGLU/GeGLU) por bloque.
- `gguf_loader` sabe leer `F32/F16/Q8_0` y mapea arquitecturas `qwen2`/`llama`.

### 1.2 Lo que NO existe (brechas por familia)
| Capacidad | DeepSeek (MLA/MoE) | Gemma |
|---|---|---|
| Atención MLA (latent compresión KV) | **No** | n/a (usa GQA) |
| MoE (ruteo de expertos + gate) | **No** | n/a (denso) |
| RoPE interleaved (GPT-NeoX) | n/a | Parcial (style por defecto, sin validar) |
| `ffn_act` alternado por bloque (Gemma2) | n/a | No |
| Carga de tensores `attn_qkv`/`ffn_experts`/`gate_inp` | **No** | Tensores estándar ya ok |
| Rama `arch=="deepseek"` en `infer_config` | **No** | `gemma` cae en default (no `split`) |

## 2. Decisiones de arquitectura propuestas

### 2.1 Modelo de datos: extender el bloque, no romper GQA
Mantener `RustGenomicBlock` denso-GQA intacto y **añadir variantes opcionales**:

```rust
// nn/attention.rs
pub enum AttentionKind {
    GQA(GenomicAttention),      // existente
    Mla(MlaAttention),          // nuevo: DeepSeek
}

// nn/block — campos opcionales MoE
pub struct RustGenomicBlock {
    // ... campos GQA densos existentes ...
    pub moe: Option<MoeRouter>,          // nuevo: DeepSeek (gate + top-k expertos)
    pub ffn_act_by_block: Option<Vec<String>>, // nuevo: Gemma2 alternado
}
```

Convenio **bit-depth mixto** del loader aplica por igual; solo cambia qué tensores
se leen y cómo se ejecutan.

### 2.2 Nueva capa `MlaAttention` (DeepSeek)
- Compresión latente: pesos `kv_lora_a` (compresión) y `kv_lora_b` (reproyección),
  más `q_lora` — mapea a `attn_qkv_a`/`attn_qkv_b` en GGUF DeepSeek.
- **Cache latente** en vez de cache KV completa: `c_kv` comprimido por token +
  `rope_freqs` (DeepSeek usa RoPE sobre la latente, no sobre k/v).
- `attn_output` sigue siendo una proyección lineal (ya soportada).

### 2.3 Nueva capa `MoeRouter` (DeepSeek)
- FFN con `n_experts` y `n_activated` (top-k).
- Tensores GGUF DeepSeek: `ffn_gate_inp` (gate), `ffn_gate_exps`/`ffn_up_exps`/
  `ffn_down_exps` (expertos, shape `[n_experts, ...]`).
- Ruteo: gate → top-k expertos → suma ponderada por router logits (softmax).

### 2.4 Gemma
- RoPE: usar el estilo **interleaved** (GPT-NeoX, ya el default) explícitamente y
  **validarlo**; corregir `infer_config` para que `arch=="gemma"` no caiga en el
  default silencioso.
- GeGLU: `ffn_act=="geglu"` (con error function `erf`) — ya soportado por nombre,
  verificar que la ejecución lo aplique.
- Gemma2 (opcional): aplicar `ffn_act` distinto por bloque impar/par.

## 3. Fases y esfuerzo

### Fase 1 — DeepSeek: atención MLA (ALTO, ~1-2 semanas)
- [ ] Definir `MlaAttention` con `kv_lora_a/b`, `q_lora`, cache latente.
- [ ] Implementar RoPE sobre latente (DeepSeek: `rope_freqs` por head_dim, interleaved).
- [ ] Integrar en `GenomicAttention`/`forward_attention_core` como `AttentionKind::Mla`.
- [ ] Unit test de paridad MLA vs referencia HF (logits por token).

### Fase 2 — DeepSeek: MoE (ALTO, ~1-2 semanas)
- [ ] `MoeRouter`: gate logits → top-k → suma ponderada de expertos.
- [ ] Almacenamiento de expertos con bit-depth mixto reutilizando `GenomicLinear`.
- [ ] Integrar en `RustGenomicBlock` (campo `moe: Option<MoeRouter>`).
- [ ] Unit test de paridad MoE (routing + combinación).

### Fase 3 — Loader GGUF DeepSeek (MEDIO, ~2-4 días)
- [ ] `infer_config`: rama `arch=="deepseek"` (rope_style interleaved, rope_base, eps).
- [ ] Cargar tensores MLA (`attn_qkv_a/b`) y MoE (`ffn_gate_inp`, `ffn_*_exps`).
- [ ] Verificar `F16`/`Q8_0`; DeepScaleR suele venir `Q4_K_M` → documentar necesidad
      de GGUF F16/BF16 o ampliar tipos.

### Fase 4 — Gemma (MEDIO, ~3-5 días)
- [ ] `infer_config`: `arch=="gemma"` → `rope_style="interleaved"`, validar GeGLU.
- [ ] Verificar carga de tensores Gemma estándar (GQA ya soportado).
- [ ] Test de paridad Gemma vs referencia HF.
- [ ] (Opcional) Gemma2: `ffn_act` alternado por bloque.

### Fase 5 — E2E y compresión (BAJO, ~1-2 días)
- [ ] Importar GGUF → `.gaje` de cada familia y generar.
- [ ] `cargo test --lib` + `cargo check --features python`.
- [ ] Benchmark de compresión (bits/peso, size) por familia.

## 4. Criterio de "hecho" (DoD)
- [ ] `gaje-cli --import <deepseek.gguf> --output model.gaje` funciona E2E y genera texto coherente.
- [ ] Paridad de logits con HF dentro de tolerancia definida (p.ej. cos-sim > 0.9 o PPL comparable).
- [ ] Idem para Gemma/Gemma2.
- [ ] Compresión mixta (4-bit attn / 2-bit FFN) aplicada a ambas familias.
- [ ] Suite Rust: tests de MLA, MoE, Gemma RoPE/GeGLU + 26 tests existentes en verde.

## 5. Riesgos y mitigaciones
- **DeepScaleR GGUF en Q4_K_M:** no es soportado por el reader (solo F32/F16/Q8_0).
  Mitigación: fase 3 documenta y, si es bloqueante, ampliar `GGMLType` (Q4_K) — esfuerzo extra.
- **Paridad MLA/MoE es numéricamente sensible:** top-k + softmax del router pueden
  divergir. Mitigación: tests de paridad por capa con umbrales, no solo E2E.
- **Memoria:** MoE con muchos expertos es pesado; el cache latente de MLA reduce KV.
  Mitigación: reutilizar `GenomicLinear` con bit-depth mixto y medir en fase 5.
- **Riesgo de no romper GQA:** las variantes son opcionales (`Option`/`enum`), el
  flujo denso actual queda intacto y cubierto por los 26 tests.

## 6. Anexo: mapa tensores GGUF
| Familia | Atención | FFN | RoPE |
|---|---|---|---|
| Llama/Qwen2.5 (actual) | `attn_q/k/v`, `attn_output` | `ffn_gate/up/down` | split |
| DeepSeek (objetivo) | `attn_qkv_a`, `attn_qkv_b` | `ffn_gate_inp`, `ffn_{gate,up,down}_exps` | interleaved (latente) |
| Gemma (objetivo) | `attn_q/k/v`, `attn_output` | `ffn_gate/up/down` (geglu) | interleaved |
