use crate::core::tokenizer::GajeTokenizer;
use crate::nn::linear::GenomicOperable;
use crate::nn::llm::GenomicLLM;
use serde::Deserialize;
use serde_json::json;
use std::collections::HashSet;
use std::io::{Cursor, Read};
use std::sync::mpsc::{channel, Receiver};
use std::thread;
use std::time::Instant;
use tiny_http::{Header, Request, Response, StatusCode};

#[derive(Deserialize, Debug)]
pub struct ChatMessage {
    pub role: Option<String>,
    pub content: Option<String>,
    pub message: Option<String>,
}

#[derive(Deserialize, Debug)]
pub struct ChatRequest {
    pub message: Option<String>,
    pub model: Option<String>,
    pub history: Option<Vec<ChatMessage>>,
    pub system_prompt: Option<String>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
}

pub struct ChannelReader {
    receiver: Receiver<Vec<u8>>,
    current_chunk: Cursor<Vec<u8>>,
}

impl ChannelReader {
    pub fn new(receiver: Receiver<Vec<u8>>) -> Self {
        Self {
            receiver,
            current_chunk: Cursor::new(Vec::new()),
        }
    }
}

impl Read for ChannelReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.current_chunk.read(buf)?;
        if n > 0 {
            return Ok(n);
        }
        match self.receiver.recv() {
            Ok(data) => {
                self.current_chunk = Cursor::new(data);
                self.current_chunk.read(buf)
            }
            Err(_) => Ok(0), // Fin del stream cuando se cierra el Sender
        }
    }
}

pub fn clean_special_tokens(text: &str) -> String {
    text.replace("<|im_end|>", "")
        .replace("<|im_start|>", "")
        .replace("<|endoftext|>", "")
        .replace("<|eot_id|>", "")
        .replace("<|start_header_id|>", "")
        .replace("<|end_header_id|>", "")
        .replace("<start_of_turn>", "")
        .replace("<end_of_turn>", "")
        .replace("</s>", "")
}

pub fn detect_chat_template_from_tokenizer(tokenizer: &GajeTokenizer) -> crate::core::gtok::ChatTemplate {
    if let Some(gtok) = tokenizer.gtok() {
        gtok.detect_chat_template()
    } else if tokenizer.token_to_id("<|im_start|>").is_some() {
        crate::core::gtok::ChatTemplate::ChatML
    } else if tokenizer.token_to_id("<|start_header_id|>").is_some() {
        crate::core::gtok::ChatTemplate::Llama3
    } else if tokenizer.token_to_id("<start_of_turn>").is_some() {
        crate::core::gtok::ChatTemplate::Gemma
    } else if tokenizer.token_to_id("[INST]").is_some() {
        crate::core::gtok::ChatTemplate::Llama2
    } else {
        crate::core::gtok::ChatTemplate::Classic
    }
}

pub fn format_chat_prompt_from_template(
    template: crate::core::gtok::ChatTemplate,
    effective_sys_prompt: &str,
    user_msg: &str,
    history: Option<&[ChatMessage]>,
) -> String {
    let mut full_prompt = String::new();
    match template {
        crate::core::gtok::ChatTemplate::ChatML => {
            if !effective_sys_prompt.is_empty() {
                full_prompt.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", effective_sys_prompt));
            }
            if let Some(hist) = history {
                for msg in hist.iter().rev().take(6).rev() {
                    let role = msg.role.as_deref().unwrap_or("user");
                    let content = msg
                        .content
                        .as_deref()
                        .or(msg.message.as_deref())
                        .unwrap_or("");
                    full_prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, content));
                }
            }
            full_prompt.push_str(&format!(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                user_msg.trim()
            ));
        }
        crate::core::gtok::ChatTemplate::Llama3 => {
            full_prompt.push_str("<|begin_of_text|>");
            if !effective_sys_prompt.is_empty() {
                full_prompt.push_str(&format!(
                    "<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>",
                    effective_sys_prompt
                ));
            }
            if let Some(hist) = history {
                for msg in hist.iter().rev().take(6).rev() {
                    let role = msg.role.as_deref().unwrap_or("user");
                    let content = msg
                        .content
                        .as_deref()
                        .or(msg.message.as_deref())
                        .unwrap_or("");
                    full_prompt.push_str(&format!(
                        "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                        role, content
                    ));
                }
            }
            full_prompt.push_str(&format!(
                "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                user_msg.trim()
            ));
        }
        crate::core::gtok::ChatTemplate::Llama2 => {
            full_prompt.push_str("<s>[INST] ");
            if !effective_sys_prompt.is_empty() {
                full_prompt.push_str(&format!("<<SYS>>\n{}\n<</SYS>>\n\n", effective_sys_prompt));
            }
            if let Some(hist) = history {
                for msg in hist.iter().rev().take(6).rev() {
                    let role = msg.role.as_deref().unwrap_or("user");
                    let content = msg
                        .content
                        .as_deref()
                        .or(msg.message.as_deref())
                        .unwrap_or("");
                    if role == "assistant" {
                        full_prompt.push_str(&format!("{} </s><s>[INST] ", content));
                    } else {
                        full_prompt.push_str(&format!("{} [/INST] ", content));
                    }
                }
            }
            full_prompt.push_str(&format!("{} [/INST] ", user_msg.trim()));
        }
        crate::core::gtok::ChatTemplate::Gemma => {
            if let Some(hist) = history {
                for msg in hist.iter().rev().take(6).rev() {
                    let role = if msg.role.as_deref() == Some("assistant") { "model" } else { "user" };
                    let content = msg
                        .content
                        .as_deref()
                        .or(msg.message.as_deref())
                        .unwrap_or("");
                    full_prompt.push_str(&format!("<start_of_turn>{}\n{}<end_of_turn>\n", role, content));
                }
            }
            if !effective_sys_prompt.is_empty() {
                full_prompt.push_str(&format!(
                    "<start_of_turn>user\n{}\n\n{}<end_of_turn>\n<start_of_turn>model\n",
                    effective_sys_prompt,
                    user_msg.trim()
                ));
            } else {
                full_prompt.push_str(&format!(
                    "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",
                    user_msg.trim()
                ));
            }
        }
        _ => {
            if !effective_sys_prompt.is_empty() {
                full_prompt.push_str(&format!("System: {}\n\n", effective_sys_prompt));
            }
            if let Some(hist) = history {
                for msg in hist.iter().rev().take(6).rev() {
                    let role = if msg.role.as_deref() == Some("assistant") { "Assistant" } else { "User" };
                    let content = msg
                        .content
                        .as_deref()
                        .or(msg.message.as_deref())
                        .unwrap_or("");
                    full_prompt.push_str(&format!("{}: {}\n", role, content));
                }
            }
            full_prompt.push_str(&format!("User: {}\nAssistant: ", user_msg.trim()));
        }
    }
    full_prompt
}

pub fn handle_chat_stream_request(
    mut request: Request,
    llm: &mut GenomicLLM,
    tokenizer: &GajeTokenizer,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut body = String::new();
    request.as_reader().read_to_string(&mut body)?;

    let chat_req: ChatRequest = serde_json::from_str(&body).unwrap_or(ChatRequest {
        message: Some(body.clone()),
        model: None,
        history: None,
        system_prompt: None,
        max_tokens: Some(256),
        temperature: Some(0.4),
        top_p: Some(0.9),
        repetition_penalty: Some(1.15),
    });

    let user_msg = chat_req.message.as_deref().unwrap_or("");
    let sys_prompt = chat_req.system_prompt.as_deref().unwrap_or(
        "Tu nombre es GAJE. Eres un asistente de inteligencia artificial avanzado, servicial, conciso y preciso.",
    );

    // 1. Detección Canónica de Plantilla de Diálogo (Zero Heuristics - Regla de Oro A.4)
    let template = detect_chat_template_from_tokenizer(tokenizer);

    // Búsqueda de memoria episódica/hipocampal
    let model_name_str = chat_req.model.as_deref().unwrap_or("");
    let mut context_prefix = String::new();
    if let Some(orch) = crate::compute::island::IslandOrchestrator::try_load_paired_memory(
        model_name_str,
        llm.dim() as u32,
    ) {
        let q_vec =
            crate::compute::island::IslandOrchestrator::vector_from_text(user_msg, llm.dim());
        let matches = orch.retrieve_context(&q_vec, 2);
        let relevant: Vec<_> = matches
            .into_iter()
            .filter(|m| m.similarity >= 0.50)
            .collect();
        if !relevant.is_empty() {
            let facts_str = relevant
                .iter()
                .map(|m| m.text.as_str())
                .collect::<Vec<_>>()
                .join(" | ");
            context_prefix = format!("[Conocimiento Hipocampal: {}]\n", facts_str);
        }
    }

    let effective_sys_prompt = if !context_prefix.is_empty() {
        if !sys_prompt.is_empty() {
            format!("{}\n\n{}", context_prefix.trim(), sys_prompt.trim())
        } else {
            context_prefix.trim().to_string()
        }
    } else {
        sys_prompt.trim().to_string()
    };

    let full_prompt = format_chat_prompt_from_template(
        template,
        &effective_sys_prompt,
        user_msg,
        chat_req.history.as_deref(),
    );

    let max_tokens = chat_req.max_tokens.unwrap_or(256);
    let temperature = chat_req.temperature.unwrap_or(0.3);
    let rep_penalty = chat_req.repetition_penalty.unwrap_or(1.15);

    let prompt_tokens_u32 = tokenizer
        .encode(&full_prompt, false)
        .map_err(|e| e.to_string())?;
    let prompt_tokens: Vec<usize> = prompt_tokens_u32.into_iter().map(|t| t as usize).collect();

    if prompt_tokens.is_empty() {
        let err_json = serde_json::json!({ "error": "Prompt vacío" });
        let resp = Response::from_string(err_json.to_string()).with_status_code(StatusCode(400));
        let _ = request.respond(resp);
        return Ok(());
    }

    let (tx, rx) = channel::<Vec<u8>>();
    let reader = ChannelReader::new(rx);

    let response = Response::new(
        StatusCode(200),
        vec![
            Header::from_bytes(&b"Content-Type"[..], &b"text/event-stream"[..]).unwrap(),
            Header::from_bytes(&b"Cache-Control"[..], &b"no-cache"[..]).unwrap(),
            Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap(),
        ],
        reader,
        None,
        None,
    );

    // Iniciar hilo HTTP para responder inmediatamente con cabeceras SSE y chunked encoding
    let respond_handle = thread::spawn(move || {
        let _ = request.respond(response);
    });

    let start_time = Instant::now();
    let n_prompt = prompt_tokens.len();

    // 1. Prefill de tokens en el KV-cache
    llm.clear_cache_core();
    for i in 0..n_prompt - 1 {
        let _ = llm.forward_blocks_only(prompt_tokens[i]);
    }

    let mut last_logits = match llm.forward_core(prompt_tokens[n_prompt - 1], false) {
        Ok(l) => l,
        Err(e) => {
            let err_event = format!("data: {}\n\n", json!({"error": e.to_string()}));
            let _ = tx.send(err_event.into_bytes());
            let _ = tx.send(b"data: [DONE]\n\n".to_vec());
            drop(tx);
            let _ = respond_handle.join();
            return Ok(());
        }
    };

    let eos_ids: HashSet<usize> = {
        let mut stops = tokenizer.get_stop_tokens();
        if stops.is_empty() {
            stops = vec![0, 2, 151643, 151644, 151645];
        }
        stops.into_iter().map(|t| t as usize).collect()
    };
    let mut generated_tokens = Vec::new();
    let mut generated_text = String::new();
    let mut seen_tokens: HashSet<usize> = HashSet::new();

    // 2. Bucle Generativo Token-por-Token en Tiempo Real
    for _ in 0..max_tokens {
        if last_logits.is_empty() {
            break;
        }

        let mut logits = last_logits.clone();

        // Penalización por repetición
        if rep_penalty > 1.0 {
            for &t in &seen_tokens {
                if t < logits.len() {
                    if logits[t] < 0.0 {
                        logits[t] *= rep_penalty;
                    } else {
                        logits[t] /= rep_penalty;
                    }
                }
            }
        }

        let next_token = if temperature <= 0.01 {
            logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        } else {
            crate::compute::sampling::sample_min_p(&logits, temperature, 0.05).unwrap_or(0)
        };

        if eos_ids.contains(&next_token) {
            break;
        }

        generated_tokens.push(next_token);
        seen_tokens.insert(next_token);

        let full_curr = tokenizer
            .decode(
                &generated_tokens
                    .iter()
                    .map(|&t| t as u32)
                    .collect::<Vec<_>>(),
                false,
            )
            .unwrap_or_default();
        let clean_full = clean_special_tokens(&full_curr);

        if clean_full.len() > generated_text.len() {
            let piece = &clean_full[generated_text.len()..];
            let sse_event = format!("data: {}\n\n", json!(piece));
            generated_text = clean_full;
            if tx.send(sse_event.into_bytes()).is_err() {
                // Cliente desconectado / abortado
                break;
            }
        }

        // Siguiente paso autorregresivo
        match llm.forward_core(next_token, false) {
            Ok(next_l) => last_logits = next_l,
            Err(_) => break,
        }
    }

    let elapsed_s = start_time.elapsed().as_secs_f64();
    let tok_count = generated_tokens.len();
    let tps = tok_count as f64 / elapsed_s.max(0.001);
    let bit_depth = llm.embeddings.weight_db.bit_depth();
    let compression_ratio = match bit_depth {
        2 => "16.0x (Q2_0 2-Bits)",
        4 => "8.0x (Q4_0 Zero-Copy)",
        8 => "4.0x (Q8_0 Native)",
        _ => "4.0x (Zero-Copy Mmap)",
    };

    let metrics_event = format!(
        "data: {}\n\n",
        json!({
            "__gaje_metrics__": {
                "tokens_per_sec": tps,
                "generated_tokens": tok_count,
                "prompt_tokens": prompt_tokens.len(),
                "total_tokens": prompt_tokens.len() + tok_count,
                "latency_ms": elapsed_s * 1000.0,
                "compression_ratio": compression_ratio
            },
            "dna": "GGCCCCCGCCCGCCGCCGCGGCGCGGGCCCGTCGGGGCGCGCCCCGGCGGCCGGCGGGGCCCCCCCCCGCCCCGCGCCCGCCGGGGCGGGCGCGGCGGCCAGCGGGCCCGGGGGCCGGGCGGGCGCGC"
        })
    );
    let _ = tx.send(metrics_event.into_bytes());
    let _ = tx.send(b"data: [DONE]\n\n".to_vec());

    drop(tx);
    let _ = respond_handle.join();
    Ok(())
}
