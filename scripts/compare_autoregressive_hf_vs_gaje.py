import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.nn.stabilized import GenomicLLM

model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
hf_model.eval()

gaje_path = os.path.join(PROJECT_ROOT, "models", "production", "smollm2_4bit.gaje")
print(f"🧬 Cargando GAJE 4-bit desde: {gaje_path}...")
gaje_llm = GenomicLLM.load_genomic(gaje_path)
gaje_llm.rust_llm.set_k_wta_ratio(0.0)

test_questions = [
    "¿Cuál es la capital de Francia?",
    "Count from 1 to 5.",
    "What is the largest planet in our solar system?",
]

print("\n======================================================")
print("🔬 COMPARATIVA AUTORREGRESIVA (PyTorch FP32 vs GAJE 4-bit)")
print("======================================================")

CHATML_TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\\n'}}{% endif %}"
tokenizer.chat_template = CHATML_TEMPLATE

for q in test_questions:
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": q}],
        tokenize=False,
        add_generation_prompt=True,
    )

    # 1. PyTorch FP32 Greedy Generation
    input_ids_hf = tokenizer.encode(formatted_prompt, return_tensors="pt")
    with torch.no_grad():
        hf_gen_ids = hf_model.generate(
            input_ids_hf,
            max_new_tokens=25,
            do_sample=False,
            eos_token_id=tokenizer.encode("<|im_end|>", add_special_tokens=False)[0],
        )[0][input_ids_hf.shape[1] :]
    hf_gen_text = tokenizer.decode(hf_gen_ids, skip_special_tokens=False)

    # 2. GAJE 4-bit Greedy Generation
    gaje_tokens = []
    gaje_llm.tokenizer.chat_template = CHATML_TEMPLATE
    for tok_text in gaje_llm.generate(
        formatted_prompt, max_new_tokens=25, temperature=0.0
    ):
        if "<|im_end|>" in tok_text:
            break
        gaje_tokens.append(tok_text)
    gaje_gen_text = "".join(gaje_tokens)

    print(f"\n❓ PREGUNTA: '{q}'")
    print(f"  - PyTorch FP32: '{hf_gen_text.strip()}'")
    print(f"  - GAJE 4-bit:   '{gaje_gen_text.strip()}'")
