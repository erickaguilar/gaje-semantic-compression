use serde::{Deserialize, Serialize};

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// # ⚓ Configuración del Organismo: Anclas de Estabilidad
///
/// Define la estructura del modelo, incluyendo la densidad de las **Anclas de Estabilidad**.
///
/// ## El Alma de Acero (Inner Threads):
/// En la arquitectura Silver Adult, las anclas son hilos de alta precisión (F16) que corren
/// por el núcleo del toroide semántico. Actúan como pozos gravitatorios que guían la
/// coherencia de los hilos genómicos de 2 bits, evitando la deriva semántica y
/// permitiendo una identidad estable.
#[cfg_attr(feature = "python", pyclass(get_all))]
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct ArchConfig {
    #[serde(default = "default_name")]
    pub name: String,
    #[serde(default = "default_version")]
    pub version: String,
    #[serde(default = "default_tokenizer")]
    pub tokenizer_id: String,
    #[serde(default = "default_rope_base")]
    pub rope_base: f32,
    #[serde(default = "default_ffn_act")]
    pub ffn_act: String,
    #[serde(default = "default_false")]
    pub use_genomic_norm: bool,
    #[serde(default = "default_rope_style")]
    pub rope_style: String,
    #[serde(default = "default_anchor_threshold")]
    pub anchor_threshold: f32,
    #[serde(default = "default_anchor_threshold")]
    pub ffn_anchor_threshold: f32,
    #[serde(default = "default_rna_threshold")]
    pub rna_threshold: f32,
    #[serde(default = "default_false")]
    pub unpermute_weights: bool,
    #[serde(default = "default_false")]
    pub apply_smollm_rope_patch: bool,
    #[serde(default = "default_false")]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_dni")]
    pub dni: String,
    #[serde(default = "default_state")]
    pub state: String,
}

#[cfg_attr(feature = "python", pymethods)]
impl ArchConfig {
    #[cfg(feature = "python")]
    #[new]
    #[pyo3(signature = (name = "GAJE-Model".to_string(), version = "1.0.0-alpha".to_string(), tokenizer_id = "gpt2".to_string(), rope_base = 10000.0, ffn_act = "swiglu".to_string(), use_genomic_norm = false, rope_style = "split".to_string(), anchor_threshold = 0.1, ffn_anchor_threshold = 0.1, rna_threshold = 0.5, unpermute_weights = false, apply_smollm_rope_patch = false, dni = "".to_string(), state = "stable".to_string()))]
    pub fn py_new(
        name: String,
        version: String,
        tokenizer_id: String,
        rope_base: f32,
        ffn_act: String,
        use_genomic_norm: bool,
        rope_style: String,
        anchor_threshold: f32,
        ffn_anchor_threshold: f32,
        rna_threshold: f32,
        unpermute_weights: bool,
        apply_smollm_rope_patch: bool,
        dni: String,
        state: String,
    ) -> Self {
        let actual_dni = if dni.is_empty() { default_dni() } else { dni };
        ArchConfig {
            name,
            version,
            tokenizer_id,
            rope_base,
            ffn_act,
            use_genomic_norm,
            rope_style,
            anchor_threshold,
            ffn_anchor_threshold,
            rna_threshold,
            tie_word_embeddings: false,
            unpermute_weights,
            apply_smollm_rope_patch,
            dni: actual_dni,
            state,
        }
    }
}

fn default_name() -> String {
    "GAJE-Model".to_string()
}
fn default_version() -> String {
    "1.0.0-alpha".to_string()
}
fn default_tokenizer() -> String {
    "gpt2".to_string()
}
fn default_rope_base() -> f32 {
    10000.0
}
fn default_ffn_act() -> String {
    "swiglu".to_string()
}
fn default_false() -> bool {
    false
}
fn default_rope_style() -> String {
    "split".to_string()
}
fn default_anchor_threshold() -> f32 {
    0.1
}
fn default_rna_threshold() -> f32 {
    0.5
}
fn default_state() -> String {
    "stable".to_string()
}

pub(crate) fn default_dni() -> String {
    use chrono::Utc;
    use rand::Rng;
    let now = Utc::now().format("%Y%m%d%H%M%S").to_string();
    let rand_id: u32 = rand::thread_rng().gen();
    format!("GAJE-DNI-{}-{:08X}", now, rand_id)
}

pub fn py_new_dni() -> String {
    default_dni()
}

#[cfg_attr(feature = "python", pyclass(get_all))]
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct ModelConfig {
    pub config: ArchConfig,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_blocks: usize,
    pub vocab_size: Option<usize>,
    pub eps: f32,
}

#[cfg_attr(feature = "python", pymethods)]
impl ModelConfig {
    #[cfg(feature = "python")]
    #[new]
    #[pyo3(signature = (config, n_embd, n_head, n_head_kv, n_blocks, vocab_size=None, eps=1e-6))]
    pub fn py_new(
        config: ArchConfig,
        n_embd: usize,
        n_head: usize,
        n_head_kv: usize,
        n_blocks: usize,
        vocab_size: Option<usize>,
        eps: f32,
    ) -> Self {
        ModelConfig {
            config,
            n_embd,
            n_head,
            n_head_kv,
            n_blocks,
            vocab_size,
            eps,
        }
    }
}
