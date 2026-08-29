use crate::core::tokenizer::GajeTokenizer;
use crate::nn::llm::GenomicLLM;
use serde::Deserialize;
use serde_json::json;
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
            Err(_) => Ok(0), // Fin del stream (Sender cerrado)
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
            let content = msg.content.as_deref().or(msg.message.as_deref()).unwrap_or("");
            full_prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, content));
        }
    }
    full_prompt.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", user_msg));

    let max_tokens = chat_req.max_tokens.unwrap_or(256);
    let temperature = chat_req.temperature.unwrap_or(0.4);
    let rep_penalty = chat_req.repetition_penalty.unwrap_or(1.15);

    let prompt_tokens_u32 = tokenizer.encode(&full_prompt, false).map_err(|e| e.to_string())?;
    let prompt_tokens: Vec<usize> = prompt_tokens_u32.into_iter().map(|t| t as usize).collect();

    let (tx, rx) = channel::<Vec<u8>>();
    let reader = ChannelReader::new(rx);

    let mut response = Response::new(
        StatusCode(200),
        vec![
            Header::from_bytes(&b"Content-Type"[..], &b"text/event-stream"[..]).unwrap(),
            Header::from_bytes(&b"Cache-Control"[..], &b"no-cache"[..]).unwrap(),
            Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap(),
        ],
        reader,
        None, // Chunked transfer encoding
        None,
    );

    // Iniciar hilo de inferencia
    let eos_ids = vec![2, 0];
    let start_time = Instant::now();

    let gen_res = llm.generate_native_core(
        prompt_tokens.clone(),
        max_tokens,
        temperature,
        rep_penalty,
        eos_ids,
    );

    // Respondemos la petición conectando el lector de canal
    let respond_handle = thread::spawn(move || {
        let _ = request.respond(response);
    });

    match gen_res {
        Ok(tokens) => {
            let mut generated_text = String::new();
            let mut tok_count = 0;

            for t in tokens {
                tok_count += 1;
                let piece = tokenizer.decode(&[t as u32], true).unwrap_or_default();
                let clean_piece = piece
                    .replace("<|im_end|>", "")
                    .replace("<|im_start|>", "")
                    .replace("<|endoftext|>", "");

                if !clean_piece.is_empty() {
                    generated_text.push_str(&clean_piece);
                    let sse_event = format!("data: {}\n\n", json!(clean_piece));
                    let _ = tx.send(sse_event.into_bytes());
                }
            }

            let elapsed_s = start_time.elapsed().as_secs_f64();
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
        }
        Err(e) => {
            let err_event = format!("data: {}\n\n", json!({"error": e.to_string()}));
            let _ = tx.send(err_event.into_bytes());
            let _ = tx.send(b"data: [DONE]\n\n".to_vec());
        }
    }

    drop(tx); // Cierra el stream
    let _ = respond_handle.join();
    Ok(())
}
