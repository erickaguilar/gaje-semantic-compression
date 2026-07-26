import os
import numpy as np
from transformers import AutoTokenizer
from gaje.nn.stabilized import GenomicLLM


def test_fp32_baseline():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(
        project_root, "data", "models", "qwen2-0_5b-instruct-fp16.gguf"
    )
    model_id = "Qwen/Qwen2-0.5B-Instruct"

    print("[*] Cargando Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"[*] Cargando modelo GAJE FP32 puro desde '{model_path}'...")
    gaje_llm = GenomicLLM(model_path)

    messages = [
        {
            "role": "user",
            "content": "Responde únicamente con una palabra: capital de Francia",
        }
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(f"\n[Formatted Chat Prompt]:\n{repr(prompt)}")

    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    curr_ids = list(input_ids)

    print("[*] Fase de Prefill...")
    logits = None
    for p_idx, tok_id in enumerate(input_ids):
        clear_cache = p_idx == 0
        logits = np.array(gaje_llm.rust_llm.forward(tok_id, clear_cache))

    print(
        "[*] Generando 10 tokens con motor nativo Rust GAJE (KV Cache Persistente)..."
    )
    generated_tokens = []

    for gen_step in range(10):
        next_tok = int(np.argmax(logits))
        generated_tokens.append(next_tok)
        curr_ids.append(next_tok)

        token_str = tokenizer.decode([next_tok])
        print(f"Token {gen_step + 1:02d}: ID={next_tok:<6d} ('{token_str}')")

        if next_tok in [tokenizer.eos_token_id, 151643]:
            break

        # Pasar ÚNICAMENTE el nuevo token generado manteniendo la KV Cache en Rust
        logits = np.array(gaje_llm.rust_llm.forward(next_tok, False))

    full_text = tokenizer.decode(curr_ids)
    print(f"\n[Resultado Generado]: '{full_text}'")


if __name__ == "__main__":
    test_fp32_baseline()
