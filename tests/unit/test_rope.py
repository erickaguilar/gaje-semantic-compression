import sys
import os

sys.path.insert(0, os.path.abspath("python"))
import gaje.utils.quantization as q

q.unpermute_to_interleaved = lambda w, *args: w

from gaje.nn.stabilized import GenomicLLM

llm = GenomicLLM("models/gguf/smollm2-135m-q8_0.gguf", num_blocks=10)
print(f"Usando RoPE Base: {llm.rope_base} | SIN PERMUTACION")

prompt = "<|im_start|>user\nWho are you?<|im_end|>\n<|im_start|>assistant\n"
for token in llm.generate(prompt, max_new_tokens=30, temperature=0.7):
    print(token, end="", flush=True)
print("\n")
