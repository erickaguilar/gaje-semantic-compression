use crate::compute::sampler::ToroidalSampler;
use crate::core::tokenizer::GajeTokenizer;
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

    let user_msg = chat_req.message.unwrap_or_default();
    let sys_prompt = chat_req.system_prompt.unwrap_or_else(|| {
        "Eres GAJE AI, un asistente genómico soberano, conciso y de alto rendimiento.".to_string()
    });

    let mut full_prompt = format!("<|im_start|>system\n{}<|im_end|>\n", sys_prompt);
    if let Some(hist) = chat_req.history {
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
        user_msg
    ));

    let max_tokens = chat_req.max_tokens.unwrap_or(256);
    let temperature = chat_req.temperature.unwrap_or(0.4);
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

    let eos_ids: HashSet<usize> = [0, 2, 151643, 151644, 151645].iter().cloned().collect();
    let mut generated_tokens = Vec::new();
    let mut generated_text = String::new();
    let mut seen_tokens: HashSet<usize> = HashSet::new();

    let mut sampler = ToroidalSampler::new_core(0.5, 0.1);

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
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        } else {
            sampler
                .sample_core(logits.clone(), temperature, chat_req.top_p.unwrap_or(0.9))
                .unwrap_or(0)
        };

        if eos_ids.contains(&next_token) {
            break;
        }

        generated_tokens.push(next_token);
        seen_tokens.insert(next_token);

        let piece = tokenizer
            .decode(&[next_token as u32], true)
            .unwrap_or_default();
        let clean_piece = piece
            .replace("<|im_end|>", "")
            .replace("<|im_start|>", "")
            .replace("<|endoftext|>", "");

        if !clean_piece.is_empty() {
            generated_text.push_str(&clean_piece);
            let sse_event = format!("data: {}\n\n", json!(clean_piece));
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

    let metrics_event = format!(
        "data: {}\n\n",
        json!({
            "__gaje_metrics__": {
                "tokens_per_sec": tps,
                "generated_tokens": tok_count,
                "prompt_tokens": prompt_tokens.len(),
                "total_tokens": prompt_tokens.len() + tok_count,
                "latency_ms": elapsed_s * 1000.0,
                "compression_ratio": "4.0x (Q4_0 Zero-Copy)"
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
