import time
from gaje.nn.stabilized import GenomicLLM


def run_inference():
    model_path = "models/production/qwen2_0_5b_q4_0_q8_0_embd.gaje.flat"
    print(f"Loading hybrid Q4_0 + Q8_0 model from {model_path}...")

    start_load = time.time()
    # Load model via wrapper
    model = GenomicLLM.load_genomic(model_path)
    load_time = (time.time() - start_load) * 1000
    print(f"Model loaded in {load_time:.2f} ms")

    prompt = "<|im_start|>user\nTell me a short story about an astronaut who discovered a green portal on Mars.<|im_end|>\n<|im_start|>assistant\n"
    print(f"\nPrompt: {prompt.strip()}")

    # Configure sampler params
    # 0.5B needs repetition penalty and a small temperature for clean output
    temp = 0.3
    rep_penalty = 1.15
    max_tokens = 150

    print("Generating response (streaming):")
    start_gen = time.time()
    response_list = []
    for tok in model.generate(prompt, max_tokens, temp, rep_penalty):
        print(tok, end="", flush=True)
        response_list.append(tok)
    print()  # new line after stream
    gen_time = time.time() - start_gen
    response = "".join(response_list)

    print(f"\nFull Response: {response}")

    # Compute throughput
    tokens = len(response_list)  # Exact token count from generator
    tok_s = tokens / gen_time
    print(
        f"\nThroughput: ~{tok_s:.2f} tok/s (generated {tokens} words in {gen_time:.2f} s)"
    )


if __name__ == "__main__":
    run_inference()
