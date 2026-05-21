import sys
import os
sys.path.append(os.path.abspath("python"))
import gguf
from gaje.nn.stabilized import GenomicLLM

llm = GenomicLLM("models/gguf/smollm2-135m-q8_0.gguf", num_blocks=10)
print("Generando con 10 bloques (CON permutación)...")
prompt = "<|im_start|>user\nHello, what is your name?<|im_end|>\n<|im_start|>assistant\n"
for token in llm.generate(prompt, max_new_tokens=30, temperature=0.7, repetition_penalty=1.1):
    print(token, end="", flush=True)
print("\n")
