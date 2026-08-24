//! # 🧠 GAJE-WASM: El Motor como Tronco Encefálico (In-Browser Runtime)
//!
//! Este módulo expone la interfaz `wasm-bindgen` para ejecutar modelos genómicos planos (.flat)
//! y la memoria soberana Island Model (.gmem v2) directamente dentro del navegador web
//! (Web Workers / Main Thread), integrando el ciclo sensorio-motor completo y consolidación
//! autonómica (ciclo de sueño en background) sin intermediación de servidores externos.

use crate::compute::island::IslandOrchestrator;
use crate::core::gtok::GtokNativeTokenizer;
use crate::io::config::ModelConfig;
use crate::io::flat_reader::GajeFlatFileReader;
use crate::io::gmem::GmemMemoryIndex;
use crate::nn::llm::GenomicLLM;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct GajeWasmEngine {
    llm: GenomicLLM,
    config: ModelConfig,
    tokenizer: Option<GtokNativeTokenizer>,
    memory: IslandOrchestrator,
}

/// Genera una representación vectorial determinista normalizada a partir de palabras y n-gramas de texto.
fn text_to_embedding(text: &str, dim: usize) -> Vec<f32> {
    let mut vec = vec![0.0f32; dim];
    let words: Vec<&str> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .collect();

    if words.is_empty() {
        return vec;
    }

    for word in words {
        let mut h: u64 = 0xcbf29ce484222325;
        for &b in word.to_lowercase().as_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        let idx = (h as usize) % dim;
        vec[idx] += 1.0;
    }

    let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-6 {
        for v in &mut vec {
            *v /= norm;
        }
    }
    vec
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

        let memory = IslandOrchestrator::new(config.n_embd as u32);

        Ok(Self {
            llm,
            config,
            tokenizer,
            memory,
        })
    }

    // =========================================================================
    // 1. VÍAS AFERENTES (SENSORIAL): Ingesta & Resonancia Semántica en WASM
    // =========================================================================

    /// Ingesta sensorial: registra un nuevo recuerdo en el nicho de memoria Island especificado.
    #[wasm_bindgen]
    pub fn ingest_sensory(
        &mut self,
        text: &str,
        vector: &[f32],
        niche: &str,
        custom_id: Option<u64>,
    ) -> Result<u64, JsValue> {
        let dim = self.config.n_embd;
        let v = if vector.len() == dim {
            vector.to_vec()
        } else {
            text_to_embedding(text, dim)
        };

        let next_id = custom_id.unwrap_or_else(|| {
            let count = self.memory.episodic.entries.len()
                + self.memory.documental.entries.len()
                + self.memory.conversational.entries.len();
            (count as u64) + 1
        });

        match niche.to_lowercase().as_str() {
            "episodic" | "episodica" | "episódica" => {
                self.memory.episodic.add_entry(next_id, v, text.to_string());
            }
            "documental" | "doc" => {
                self.memory
                    .documental
                    .add_entry(next_id, v, text.to_string());
            }
            "conversational" | "conversacion" | "conversación" => {
                self.memory
                    .conversational
                    .add_entry(next_id, v, text.to_string());
            }
            _ => {
                return Err(JsValue::from_str(
                    "Nicho desconocido. Opciones válidas: episodic, documental, conversational",
                ));
            }
        }

        Ok(next_id)
    }

    /// Recupera los contextos más resonantes en la memoria Island como objeto JSON.
    #[wasm_bindgen]
    pub fn retrieve_context(
        &self,
        query_text: &str,
        query_vector: &[f32],
        top_k: usize,
    ) -> Result<String, JsValue> {
        let dim = self.config.n_embd;
        let q_vec = if query_vector.len() == dim {
            query_vector.to_vec()
        } else {
            text_to_embedding(query_text, dim)
        };

        let results = self.memory.retrieve_context(&q_vec, top_k);
        let items: Vec<serde_json::Value> = results
            .into_iter()
            .map(|r| {
                serde_json::json!({
                    "id": r.id,
                    "text": r.text,
                    "similarity": r.similarity,
                    "niche": r.niche.as_str()
                })
            })
            .collect();

        serde_json::to_string(&items)
            .map_err(|e| JsValue::from_str(&format!("Error serializando retrieval: {}", e)))
    }

    // =========================================================================
    // 2. CICLO AUTONÓMICO: Sueño, Consolidación y Poda Semántica en WASM
    // =========================================================================

    /// Ejecuta el ciclo autonómico de consolidación de memoria (sueño biológico en background).
    #[wasm_bindgen]
    pub fn autonomic_sleep_cycle(&mut self, dedup_threshold: f32) -> Result<String, JsValue> {
        let stats = self.memory.consolidate_memory(dedup_threshold);
        serde_json::to_string(&stats)
            .map_err(|e| JsValue::from_str(&format!("Error en ciclo de sueño: {}", e)))
    }

    /// Exporta la memoria de un nicho a formato binario .gmem v2 para persistencia en IndexedDB/OPFS.
    #[wasm_bindgen]
    pub fn export_gmem_island(&self, niche: &str) -> Result<Vec<u8>, JsValue> {
        let idx = match niche.to_lowercase().as_str() {
            "episodic" | "episodica" | "episódica" => &self.memory.episodic,
            "documental" | "doc" => &self.memory.documental,
            "conversational" | "conversacion" | "conversación" => &self.memory.conversational,
            _ => {
                return Err(JsValue::from_str(
                    "Nicho desconocido. Opciones: episodic, documental, conversational",
                ));
            }
        };
        Ok(idx.save_to_bytes())
    }

    /// Importa la memoria de un nicho desde bytes binarios .gmem v2 recuperados de IndexedDB/OPFS.
    #[wasm_bindgen]
    pub fn import_gmem_island(&mut self, niche: &str, bytes: &[u8]) -> Result<(), JsValue> {
        let idx = GmemMemoryIndex::load_from_bytes(bytes)
            .map_err(|e| JsValue::from_str(&format!("Error importando .gmem: {}", e)))?;

        match niche.to_lowercase().as_str() {
            "episodic" | "episodica" | "episódica" => self.memory.episodic = idx,
            "documental" | "doc" => self.memory.documental = idx,
            "conversational" | "conversacion" | "conversación" => self.memory.conversational = idx,
            _ => {
                return Err(JsValue::from_str(
                    "Nicho desconocido. Opciones: episodic, documental, conversational",
                ));
            }
        }
        Ok(())
    }

    /// Retorna estadísticas en tiempo real del estrato de memoria Island.
    #[wasm_bindgen]
    pub fn get_memory_stats(&self) -> String {
        serde_json::to_string(&serde_json::json!({
            "dim": self.memory.dim,
            "niche_weights": self.memory.niche_weights,
            "episodic_entries": self.memory.episodic.entries.len(),
            "documental_entries": self.memory.documental.entries.len(),
            "conversational_entries": self.memory.conversational.entries.len(),
            "documental_consolidated": self.memory.documental.is_consolidated(),
            "total_entries": self.memory.episodic.entries.len() + self.memory.documental.entries.len() + self.memory.conversational.entries.len(),
        }))
        .unwrap_or_else(|_| "{}".to_string())
    }

    // =========================================================================
    // 3. VÍAS EFERENTES (MOTORA) & CHAT CONTEXTUAL INTEGRADO
    // =========================================================================

    /// Chat end-to-end con inyección automática de memoria asociativa e ingesta de turno.
    #[wasm_bindgen]
    pub fn chat_with_memory(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        repetition_penalty: f32,
        inject_rag: bool,
    ) -> Result<String, JsValue> {
        let (prompt_ids, stop_ids) = {
            let tok = self.tokenizer.as_ref().ok_or_else(|| {
                JsValue::from_str("Tokenizador GTOK no disponible en el modelo cargado")
            })?;

            let mut relevant_context = String::new();
            if inject_rag {
                let q_vec = text_to_embedding(prompt, self.config.n_embd);
                let contexts = self.memory.retrieve_context(&q_vec, 2);
                let relevant_snippets: Vec<String> = contexts
                    .iter()
                    .filter(|c| c.similarity >= 0.50)
                    .map(|c| format!("- {}", c.text))
                    .collect();

                if !relevant_snippets.is_empty() {
                    relevant_context = format!(
                        "Información de memoria recuperada:\n{}\n\n",
                        relevant_snippets.join("\n")
                    );
                }
            }

            let formatted = if tok.token_to_id.contains_key("<|im_start|>") {
                if !relevant_context.is_empty() {
                    format!(
                        "<|im_start|>system\n{}Responde al usuario de manera precisa y directa.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                        relevant_context, prompt
                    )
                } else {
                    format!(
                        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                        prompt
                    )
                }
            } else if tok.token_to_id.contains_key("<|user|>") {
                if !relevant_context.is_empty() {
                    format!(
                        "<|system|>\n{}<|end|>\n<|user|>\n{}<|end|>\n<|assistant|>\n",
                        relevant_context, prompt
                    )
                } else {
                    format!("<|user|>\n{}<|end|>\n<|assistant|>\n", prompt)
                }
            } else if !relevant_context.is_empty() {
                format!("{}{}", relevant_context, prompt)
            } else {
                prompt.to_string()
            };

            let p_ids = tok.encode(&formatted);
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
            .trim_end_matches("<|im_end|>")
            .trim_end_matches("<|endoftext|>")
            .trim_end_matches("<end_of_turn>")
            .trim();

        let response = cleaned.to_string();

        // Auto-ingesta del turno conversacional
        let conv_record = format!("U: {} | A: {}", prompt, response);
        let _ = self.ingest_sensory(&conv_record, &[], "conversational", None);

        Ok(response)
    }

    /// Emite decisiones motoras estructuradas (Tool Calling / Actuadores).
    #[wasm_bindgen]
    pub fn actuate(&mut self, prompt: &str, tools_schema_json: &str) -> Result<String, JsValue> {
        let system_instruction = format!(
            "Eres un agente motor que debe responder exclusivamente en JSON según este esquema de herramientas:\n{}\n\nEntrada del usuario: {}",
            tools_schema_json, prompt
        );
        self.chat(&system_instruction, 128, 0.2, 1.1)
    }

    // =========================================================================
    // 4. MÉTODOS BASE DE GENERACIÓN Y TOKENIZACIÓN
    // =========================================================================

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

            let formatted = if tok.token_to_id.contains_key("<|im_start|>") {
                format!(
                    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                    prompt
                )
            } else if tok.token_to_id.contains_key("<|user|>") {
                format!("<|user|>\n{}<|end|>\n<|assistant|>\n", prompt)
            } else {
                prompt.to_string()
            };

            let p_ids = tok.encode(&formatted);
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
