import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.nn.stabilized import GenomicLLM  # noqa: E402


def test_gen():
    flat_path = os.path.join(
        PROJECT_ROOT, "models", "production", "qwen2_0_5b_4bit.gaje.flat"
    )
    print(f"[*] Cargando {flat_path}...")
    llm = GenomicLLM.load_genomic(flat_path)
    llm.rust_llm.set_k_wta_ratio(0.0)

    prompt = "<|im_start|>user\n¿Cuál es la capital de Francia?<|im_end|>\n<|im_start|>assistant\n"
    print(f"[*] Generando para: {prompt!r}")

    out_text = ""
    for tok in llm.generate(
        prompt, max_new_tokens=40, temperature=0.3, repetition_penalty=1.1
    ):
        out_text += tok
        print(tok, end="", flush=True)
    print("\n------------------------------------")
    print("Respuesta completa:", out_text)


if __name__ == "__main__":
    test_gen()
