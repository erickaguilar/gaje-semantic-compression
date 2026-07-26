import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.nn.stabilized import GenomicLLM

model_path = os.path.join(PROJECT_ROOT, "models", "production", "smollm2_4bit.gaje")
print(f"🧬 Probando Generación con Template ChatML explícito: {model_path}...")

llm = GenomicLLM.load_genomic(model_path)
llm.rust_llm.set_k_wta_ratio(0.0)

# Asignar chat_template si no está presente
CHATML_TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\\n'}}{% endif %}"
llm.tokenizer.chat_template = CHATML_TEMPLATE

questions = [
    "¿Cuál es la capital de Francia?",
    "Count from 1 to 5.",
]

for q in questions:
    formatted = llm.tokenizer.apply_chat_template(
        [{"role": "user", "content": q}],
        tokenize=False,
        add_generation_prompt=True,
    )
    print(f"\n--- PREGUNTA: '{q}' ---")
    print(f"ChatML Formatted Prompt:\n{repr(formatted)}")
    print("Respuesta: ", end="", flush=True)
    for tok in llm.generate(formatted, max_new_tokens=40, temperature=0.0):
        print(tok, end="", flush=True)
    print()
