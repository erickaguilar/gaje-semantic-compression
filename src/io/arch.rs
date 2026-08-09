use serde::{Deserialize, Serialize};

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass(eq, eq_int))]
pub enum ModelFamily {
    Legacy = 0,
    Llama = 1,
    SmolLM = 2,
    Qwen2 = 3,
    Qwen2_5 = 4,
    Gemma = 5,
    Unknown = 6,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
pub struct ArchitectureDescriptor {
    pub family: ModelFamily,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_blocks: usize,
    pub head_dim: usize,
    pub rope_base: f32,
    pub rope_style: String,    // "split" or "interleaved"
    pub ffn_act: String,       // "swiglu", "silu", etc.
    pub qk_permute: bool,      // whether Q/K weights need to be unpermutated
    pub chat_template: String, // chatML, llama, etc.
}

#[cfg_attr(feature = "python", pymethods)]
impl ArchitectureDescriptor {
    #[cfg(feature = "python")]
    #[new]
    #[pyo3(signature = (family=ModelFamily::Legacy, n_embd=0, n_head=0, n_head_kv=0, n_blocks=0, head_dim=0, rope_base=10000.0, rope_style="split".to_string(), ffn_act="swiglu".to_string(), qk_permute=false, chat_template="standard".to_string()))]
    pub fn py_new(
        family: ModelFamily,
        n_embd: usize,
        n_head: usize,
        n_head_kv: usize,
        n_blocks: usize,
        head_dim: usize,
        rope_base: f32,
        rope_style: String,
        ffn_act: String,
        qk_permute: bool,
        chat_template: String,
    ) -> Self {
        let actual_head_dim = if head_dim == 0 && n_head > 0 {
            n_embd / n_head
        } else {
            head_dim
        };
        Self {
            family,
            n_embd,
            n_head,
            n_head_kv,
            n_blocks,
            head_dim: actual_head_dim,
            rope_base,
            rope_style,
            ffn_act,
            qk_permute,
            chat_template,
        }
    }

    #[cfg(feature = "python")]
    #[staticmethod]
    pub fn infer_from_name(
        name: String,
        n_embd: usize,
        n_head: usize,
        n_head_kv: usize,
        n_blocks: usize,
    ) -> Self {
        Self::infer_from_name_core(&name, n_embd, n_head, n_head_kv, n_blocks)
    }
}

impl ArchitectureDescriptor {
    pub fn infer_from_name_core(
        name: &str,
        n_embd: usize,
        n_head: usize,
        n_head_kv: usize,
        n_blocks: usize,
    ) -> Self {
        let name_lower = name.to_lowercase();
        let (family, rope_base, qk_permute, chat_template, rope_style) =
            if name_lower.contains("qwen2.5") {
                (
                    ModelFamily::Qwen2_5,
                    1000000.0,
                    false,
                    "chatml".to_string(),
                    "split".to_string(),
                )
            } else if name_lower.contains("qwen2") {
                (
                    ModelFamily::Qwen2,
                    1000000.0,
                    false,
                    "chatml".to_string(),
                    "split".to_string(),
                )
            } else if name_lower.contains("smollm2") {
                (
                    ModelFamily::SmolLM,
                    100000.0,
                    true,
                    "chatml".to_string(),
                    "split".to_string(),
                )
            } else if name_lower.contains("smollm") {
                (
                    ModelFamily::SmolLM,
                    100000.0,
                    true,
                    "chatml".to_string(),
                    "split".to_string(),
                )
            } else if name_lower.contains("gemma") {
                (
                    ModelFamily::Gemma,
                    10000.0,
                    false,
                    "gemma".to_string(),
                    "split".to_string(),
                )
            } else if name_lower.contains("llama") {
                (
                    ModelFamily::Llama,
                    10000.0,
                    true,
                    "llama".to_string(),
                    "split".to_string(),
                )
            } else {
                (
                    ModelFamily::Unknown,
                    10000.0,
                    false,
                    "standard".to_string(),
                    "split".to_string(),
                )
            };

        let head_dim = if n_head > 0 { n_embd / n_head } else { 0 };

        Self {
            family,
            n_embd,
            n_head,
            n_head_kv,
            n_blocks,
            head_dim,
            rope_base,
            rope_style,
            ffn_act: "swiglu".to_string(),
            qk_permute,
            chat_template,
        }
    }
}
