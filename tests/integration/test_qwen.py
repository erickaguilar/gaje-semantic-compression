import gaje.core._impl as dna

print(f"Module file: {dna.__file__}")
from gaje.nn.stabilized import GenomicLLM

model_path = "models/gguf/qwen2-0_5b-q8_0.gguf"
llm = GenomicLLM(model_path, num_blocks=5)

print("\n--- TEST GENERATION ---")
prompt = "<|im_start|>user\nWho are you?<|im_end|>\n<|im_start|>assistant\n"
for token in llm.generate(prompt, max_new_tokens=20, temperature=0.3):
    print(token, end="", flush=True)
print("\n")
