//! # 🧠 GAJE-WASM: El Motor como Tronco Encefálico (In-Browser Runtime)
//!
//! Este módulo expone la interfaz `wasm-bindgen` para ejecutar modelos genómicos planos (.flat)
//! directamente dentro de navegadores web (Web Workers / Main Thread) sin servidores de fondo.

use crate::core::gtok::GtokNativeTokenizer;
use crate::io::config::ModelConfig;
use crate::io::flat_reader::GajeFlatFileReader;
use crate::nn::llm::GenomicLLM;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct GajeWasmEngine {
    llm: GenomicLLM,
    config: ModelConfig,
    tokenizer: Option<GtokNativeTokenizer>,
}

#[wasm_bindgen]
impl GajeWasmEngine {
    /// Inicializa las tablas de cómputo matemático globales para WASM.
    #[wasm_bindgen]
    pub fn init_engine() {
        console_error_panic_hook::set_once();
        unsafe {
            crate::compute::kernels::init_shuffle_table();
        }
    }

    /// Carga el organismo genómico .flat directamente desde un ArrayBuffer / Uint8Array en JS.
    #[wasm_bindgen]
    pub fn load_from_bytes(bytes: &[u8]) -> Result<GajeWasmEngine, JsValue> {
        Self::init_engine();

        let reader = GajeFlatFileReader::from_bytes(bytes.to_vec())
            .map_err(|e| JsValue::from_str(&format!("Error leyendo formato .flat: {}", e)))?;

        let config = reader.load_config().map_err(|e| {
            JsValue::from_str(&format!("Error leyendo metadatos de configuración: {}", e))
        })?;

        let tokenizer = reader.get_embedded_gtok();

        let llm = reader
            .load_genomic()
            .map_err(|e| JsValue::from_str(&format!("Error instanciando GenomicLLM: {}", e)))?;

        Ok(Self {
            llm,
            config,
            tokenizer,
        })
    }

    /// Tokeniza un texto a un arreglo de IDs de tokens en JS.
    #[wasm_bindgen]
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, JsValue> {
        if let Some(ref tok) = self.tokenizer {
            Ok(tok.encode(text))
        } else {
            Err(JsValue::from_str(
                "El modelo no contiene un tokenizador GTOK incrustado",
            ))
        }
    }

    /// Decodifica una secuencia de IDs de tokens a string.
    #[wasm_bindgen]
    pub fn decode(&self, ids: &[u32]) -> Result<String, JsValue> {
        if let Some(ref tok) = self.tokenizer {
            Ok(tok.decode(ids))
        } else {
            Err(JsValue::from_str(
                "El modelo no contiene un tokenizador GTOK incrustado",
            ))
        }
    }

    /// Generación completa autorregresiva en Rust nativo sobre WASM.
    #[wasm_bindgen]
    pub fn generate(
        &mut self,
        prompt_ids: &[u32],
        max_tokens: usize,
        temperature: f32,
        repetition_penalty: f32,
        stop_ids: &[u32],
    ) -> Result<Vec<u32>, JsValue> {
        let prompt_usize: Vec<usize> = prompt_ids.iter().map(|&x| x as usize).collect();
        let stop_usize: Vec<usize> = stop_ids.iter().map(|&x| x as usize).collect();

        let gen_usize = self
            .llm
            .generate_native_core(
                prompt_usize,
                max_tokens,
                temperature,
                repetition_penalty,
                stop_usize,
            )
            .map_err(|e| {
                JsValue::from_str(&format!("Error en generación autorregresiva: {}", e))
            })?;

        Ok(gen_usize.into_iter().map(|x| x as u32).collect())
    }

    /// Chat end-to-end: recibe texto del usuario y retorna la respuesta generada.
    #[wasm_bindgen]
    pub fn chat(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        repetition_penalty: f32,
    ) -> Result<String, JsValue> {
        let (prompt_ids, stop_ids) = {
            let tok = self.tokenizer.as_ref().ok_or_else(|| {
                JsValue::from_str("Tokenizador GTOK no disponible en el modelo cargado")
            })?;

            let p_ids = tok.encode(prompt);
            if p_ids.is_empty() {
                return Ok(String::new());
            }

            let mut s_ids = vec![2];
            if let Some(&im_end) = tok.token_to_id.get("<|im_end|>") {
                s_ids.push(im_end);
            }
            if let Some(&eos) = tok.token_to_id.get("<|endoftext|>") {
                s_ids.push(eos);
            }
            if let Some(&eot) = tok.token_to_id.get("<end_of_turn>") {
                s_ids.push(eot);
            }
            (p_ids, s_ids)
        };

        let gen_ids = self.generate(
            &prompt_ids,
            max_tokens,
            temperature,
            repetition_penalty,
            &stop_ids,
        )?;

        let full_text = if let Some(ref tok) = self.tokenizer {
            tok.decode(&gen_ids)
        } else {
            String::new()
        };

        let cleaned = full_text
            .split("<|im_end|>")
            .next()
            .unwrap_or("")
            .split("<|endoftext|>")
            .next()
            .unwrap_or("")
            .split("<end_of_turn>")
            .next()
            .unwrap_or("")
            .trim()
            .to_string();

        Ok(cleaned)
    }

    /// Retorna información arquitectónica del modelo como objeto JSON.
    #[wasm_bindgen]
    pub fn get_model_info(&self) -> String {
        serde_json::to_string(&serde_json::json!({
            "n_embd": self.config.n_embd,
            "n_head": self.config.n_head,
            "n_head_kv": self.config.n_head_kv,
            "n_layer": self.config.n_blocks,
            "vocab_size": self.config.vocab_size,
            "has_quantum_embeddings": self.llm.quantum_embeddings.is_some(),
            "has_gtok": self.tokenizer.is_some(),
        }))
        .unwrap_or_else(|_| "{}".to_string())
    }

    /// Limpia el estado interno de KV Cache para reiniciar la conversación.
    #[wasm_bindgen]
    pub fn reset_cache(&mut self) {
        self.llm.clear_cache_core();
    }
}
