use crate::nn::llm::GenomicLLM;
use crate::compute::sampler::ToroidalSampler;
use crate::core::session_memory::SessionBuffer;
use crate::core::tokenizer::GajeTokenizer;
use crate::io::loader::NativeLoader;
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
    pub fn chat(&mut self, user_input: &str, max_new_tokens: usize, temperature: f32, top_p: f32) -> Result<String, Box<dyn Error>> {
        // 1. Tokenización del input del usuario
        let user_tokens = self.tokenizer.encode(user_input, false)?;
        if user_tokens.is_empty() {
            return Ok(String::new());
        }

        // 2. Recuperación de Memoria Semántica (Toroidal Recall)
        // Obtenemos la fase del último token del input para buscar en memoria
        let (_, user_phase) = self.model.forward_with_hidden_core(user_tokens[user_tokens.len() - 1] as usize, true)?;
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
        prompt.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", user_input));

        // 4. Generación Token-a-token con Sampler Toroidal
        let response = self.generate(&prompt, max_new_tokens, temperature, top_p)?;

        // 5. Recirculación: Guardar en Memoria de Sesión
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        let interaction = format!("U: {}\nA: {}", user_input, response);
        self.memory.push(interaction, user_phase, timestamp);

        Ok(response)
    }

    /// Genera texto a partir de un prompt crudo.
    pub fn generate(&mut self, prompt: &str, max_tokens: usize, temperature: f32, top_p: f32) -> Result<String, Box<dyn Error>> {
        let tokens = self.tokenizer.encode(prompt, false)?;
        if tokens.is_empty() { return Ok(String::new()); }

        self.model.clear_cache_core();
        self.sampler.reset();

        // Prefilled (KV Cache)
        let mut last_logits = Vec::new();
        for &tid in &tokens {
            last_logits = self.model.forward_core(tid as usize, false)?;
        }

        let mut generated_ids = Vec::new();
        let mut current_logits = last_logits;
        
        let eos_token_id = self.tokenizer.token_to_id("<|im_end|>")
            .or_else(|| self.tokenizer.token_to_id("<|endoftext|>"))
            .or_else(|| self.tokenizer.token_to_id("</s>"));

        for _ in 0..max_tokens {
            let next_id = self.sampler.sample_core(current_logits, temperature, top_p)
                .map_err(|e| format!("Sampler error: {}", e))?;
            
            if let Some(eos) = eos_token_id {
                if next_id == eos as usize { break; }
            }

            generated_ids.push(next_id as u32);
            
            // Siguiente paso (incremental)
            current_logits = self.model.forward_core(next_id, false)?;
        }

        self.tokenizer.decode(&generated_ids, true)
    }
}
