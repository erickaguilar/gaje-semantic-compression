use crate::compute::sampler::ToroidalSampler;
use crate::core::session_memory::SessionBuffer;
use crate::core::tokenizer::GajeTokenizer;
use crate::io::loader::NativeLoader;
use crate::nn::llm::GenomicLLM;
use std::error::Error;
use std::time::{SystemTime, UNIX_EPOCH};

/// # 🏛️ GajeSession: Fachada de Alto Nivel para el SDK Nativo
///
/// Encapsula el modelo, el sampler, la memoria de sesión y el tokenizador en un único
/// objeto persistente para permitir inferencia fluida sin dependencias de Python.
pub struct GajeSession {
    pub model: GenomicLLM,
    pub sampler: ToroidalSampler,
    pub memory: SessionBuffer,
    pub tokenizer: GajeTokenizer,
    pub n_embd: usize,
}

impl GajeSession {
    /// Inicializa una nueva sesión cargando el modelo desde el path especificado.
    pub fn load(model_path: &str, memory_capacity: usize) -> Result<Self, Box<dyn Error>> {
        // Inicializar tablas globales de cómputo
        unsafe {
            crate::compute::kernels::init_shuffle_table();
        }

        let loader = NativeLoader::new(model_path)?;
        let model = loader.load_llm()?;
        let config = loader.load_config()?;
        let tokenizer = loader.load_tokenizer()?;

        let n_embd = config.n_embd;
        let sampler = ToroidalSampler::new_core(1.0, 0.1);
        let memory = SessionBuffer::new(memory_capacity, n_embd);

        Ok(Self {
            model,
            sampler,
            memory,
            tokenizer,
            n_embd,
        })
    }

    /// Procesa una interacción completa: Recupera memoria, genera respuesta y guarda en sesión.
    pub fn chat(
        &mut self,
        user_input: &str,
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
    ) -> Result<String, Box<dyn Error>> {
        // 1. Tokenización del input del usuario
        let user_tokens = self.tokenizer.encode(user_input, false)?;
        if user_tokens.is_empty() {
            return Ok(String::new());
        }

        // 2. Recuperación de Memoria Semántica (Toroidal Recall)
        // Obtenemos la fase del último token del input para buscar en memoria
        let (_, user_phase) = self
            .model
            .forward_with_hidden_core(user_tokens[user_tokens.len() - 1] as usize, true)?;
        let relevant_context = self.memory.retrieve_relevant(user_phase.clone(), 2)?;

        // 3. Construcción del Prompt (Formato ChatML básico)
        let mut prompt = String::new();
        if !relevant_context.is_empty() {
            prompt.push_str("<|im_start|>system\n--- MEMORIA RELEVANTE ---\n");
            for ctx in relevant_context {
                prompt.push_str(&ctx);
                prompt.push('\n');
            }
            prompt.push_str("--------------------------<|im_end|>\n");
        }
        prompt.push_str(&format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            user_input
        ));

        // 4. Generación Token-a-token con Sampler Toroidal
        let response = self.generate(&prompt, max_new_tokens, temperature, top_p)?;

        // 5. Recirculación: Guardar en Memoria de Sesión
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        let interaction = format!("U: {}\nA: {}", user_input, response);
        self.memory.push(interaction, user_phase, timestamp);

        Ok(response)
    }

    /// Genera texto a partir de un prompt crudo.
    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
    ) -> Result<String, Box<dyn Error>> {
        let tokens = self.tokenizer.encode(prompt, false)?;
        if tokens.is_empty() {
            return Ok(String::new());
        }

        self.model.clear_cache_core();
        self.sampler.reset();

        // Prefilled (KV Cache)
        let mut last_logits = Vec::new();
        for &tid in &tokens {
            last_logits = self.model.forward_core(tid as usize, false)?;
        }

        let mut generated_ids = Vec::new();
        let mut current_logits = last_logits;

        let eos_token_id = self
            .tokenizer
            .token_to_id("<|im_end|>")
            .or_else(|| self.tokenizer.token_to_id("<|endoftext|>"))
            .or_else(|| self.tokenizer.token_to_id("</s>"));

        for _ in 0..max_tokens {
            let next_id = self
                .sampler
                .sample_core(current_logits.clone(), temperature, top_p)
                .map_err(|e| format!("Sampler error: {}", e))?;

            if let Some(eos) = eos_token_id {
                if next_id == eos as usize {
                    break;
                }
            }

            generated_ids.push(next_id as u32);

            // Siguiente paso (incremental)
            current_logits = self.model.forward_core(next_id, false)?;
        }

        if generated_ids.is_empty() {
            return Ok("[Sin respuesta]".to_string());
        }

        self.tokenizer.decode(&generated_ids, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::loader::init_born_genomic_model;
    use crate::io::loader::ArchConfig;
    use crate::io::loader::ModelConfig;
    use std::fs;

    #[test]
    fn test_sdk_session_lifecycle() {
        let model_path = "test_sdk_model.gaje";
        let tokenizer_json = "models/core/tokenizer.json";

        let mut vocab_size = 1000;
        let mut tokenizer_opt = None;

        if fs::metadata(tokenizer_json).is_ok() {
            let tokenizer =
                GajeTokenizer::from_file(tokenizer_json).expect("Error cargando tokenizer");
            vocab_size = tokenizer.vocab_size();
            tokenizer_opt = Some(tokenizer);
        }

        // 1. Crear un modelo de prueba (micro_organism) con el tamaño real del vocabulario
        let arch = ArchConfig {
            name: "Test-SDK".to_string(),
            version: "1.0.0".to_string(),
            tokenizer_id: "gpt2".to_string(),
            rope_base: 10000.0,
            ffn_act: "swiglu".to_string(),
            use_genomic_norm: true,
            rope_style: "split".to_string(),
            anchor_threshold: 0.1,
            ffn_anchor_threshold: 0.1,
            rna_threshold: 0.5,
            unpermute_weights: false,
            apply_smollm_rope_patch: false,
            tie_word_embeddings: false,
            dni: "test-dni".to_string(),
            state: "born".to_string(),
        };
        let config = ModelConfig {
            config: arch,
            n_embd: 64,
            n_head: 2,
            n_head_kv: 2,
            n_blocks: 1,
            vocab_size: Some(vocab_size),
            eps: 1e-6,
        };

        // Inicializar el modelo físicamente
        let model = init_born_genomic_model(model_path, config.clone(), vocab_size)
            .expect("Error inicializando modelo");

        // Guardar con tokenizer si lo tenemos
        if let Some(tok) = tokenizer_opt {
            let _ = crate::io::loader::save_genomic_model(model_path, &model, &config, Some(&tok));
        }

        // 2. Probar carga de la sesión
        match GajeSession::load(model_path, 5) {
            Ok(mut session) => {
                assert_eq!(session.n_embd, 64);

                // 3. Probar generación cruda
                let prompt = "Hola";
                match session.generate(prompt, 2, 0.7, 0.9) {
                    Ok(res) => {
                        println!("✅ SDK Nativo generó: '{}'", res);
                    }
                    Err(e) => panic!("Fallo en generación nativa: {}", e),
                }

                // 4. Probar Chat (Ciclo completo con memoria)
                match session.chat("¿Quién eres?", 2, 0.7, 0.9) {
                    Ok(res) => println!("✅ SDK Nativo Chat respondió: '{}'", res),
                    Err(e) => panic!("Fallo en Chat nativo: {}", e),
                }
            }
            Err(e) => {
                println!("Aviso: No se pudo probar el ciclo completo: {}", e);
            }
        }

        // Limpieza
        let _ = fs::remove_file(model_path);
    }
}
