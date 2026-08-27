import os
import sys
import time
import json
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

# If we are in the subprocess:
if "--run-single" in sys.argv:
    # Import inside subprocess to avoid importing heavy libraries in the manager process
    from gaje.nn.stabilized import GenomicLLM  # noqa: E402

    model_name = sys.argv[2]
    model_path = sys.argv[3]
    output_temp_file = sys.argv[4]

    prompts = [
        "¿Cuál es la capital de Francia?",
        "太阳系中最大的行星是哪一颗？",
        "Count from 1 to 5",
        "Un padre tiene el triple de la edad de su hijo. Dentro de 12 años, tendrá el doble. ¿Qué edad tienen ambos actualmente? Explica el procedimiento paso a paso.",
        "Write a Python function that takes a string and returns the first non-repeating character.",
        "¿Quién eres?",
        "Explain the physics of why water boils when heated.",
    ]

    if not os.path.exists(model_path):
        sys.exit(1)

    start_load = time.time()
    llm = GenomicLLM.load_genomic(model_path)
    load_time = time.time() - start_load

    llm.rust_llm.set_k_wta_ratio(0.0)

    eos_ids = [151643, 151645]
    if (
        hasattr(llm.tokenizer, "eos_token_id")
        and llm.tokenizer.eos_token_id is not None
    ):
        eos_ids.append(llm.tokenizer.eos_token_id)

    results = []

    for idx, prompt in enumerate(prompts):
        formatted = f"<|im_start|>system\nYou are a helpful and precise assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        tokens = llm.tokenizer.encode(formatted, add_special_tokens=False)
        if hasattr(tokens, "ids"):
            tokens = tokens.ids

        llm.rust_llm.clear_cache_py()

        start_gen = time.time()
        gen_ids = llm.rust_llm.generate_native_py(tokens, 250, 0.0, 1.0, eos_ids)
        gen_duration = time.time() - start_gen

        response = llm.tokenizer.decode(gen_ids)
        cleaned = (
            response.split("<|im_start|>")[0]
            .split("<|im_end|>")[0]
            .split("<|endoftext|>")[0]
            .strip()
        )

        n_tokens = len(gen_ids)
        tok_per_sec = n_tokens / gen_duration if gen_duration > 0 else 0.0

        results.append(
            {
                "prompt": prompt,
                "response": cleaned,
                "gen_time_sec": gen_duration,
                "tokens_generated": n_tokens,
                "tokens_per_sec": tok_per_sec,
            }
        )

    res_dict = {
        "model_name": model_name,
        "load_time_sec": load_time,
        "results": results,
    }

    with open(output_temp_file, "w", encoding="utf-8") as f:
        json.dump(res_dict, f, ensure_ascii=False)

    sys.exit(0)


# Else, we are in the main manager process:
def main():
    models_to_test = [
        (
            "Qwen2.5-1.5B (Q4_0 weights + FP32 embeddings)",
            os.path.join(
                PROJECT_ROOT, "models", "production", "qwen2_5_1_5b_q4_0.gaje.flat"
            ),
        ),
        (
            "Qwen2.5-1.5B Híbrido (Q4_0 weights + Q8_0 embeddings)",
            os.path.join(
                PROJECT_ROOT,
                "models",
                "production",
                "qwen2_5_1_5b_q4_0_q8_0_embd.gaje.flat",
            ),
        ),
        (
            "Qwen2.5-3B Híbrido (Q4_0 weights + Q8_0 embeddings)",
            os.path.join(
                PROJECT_ROOT,
                "models",
                "production",
                "qwen2_5_3b_q4_0_q8_0_embd.gaje.flat",
            ),
        ),
    ]

    all_benchmarks = {}

    for name, path in models_to_test:
        if not os.path.exists(path):
            print(f"❌ Skipping {name} (path does not exist: {path})")
            continue

        print(f"\n🧬 Spawning isolated benchmark process for: {name}")
        temp_file = os.path.join(PROJECT_ROOT, f"temp_res_{int(time.time())}.json")

        # Run itself as a subprocess
        cmd = [sys.executable, __file__, "--run-single", name, path, temp_file]

        start_proc = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - start_proc

        if proc.returncode == 0 and os.path.exists(temp_file):
            with open(temp_file, "r", encoding="utf-8") as f:
                res_dict = json.load(f)
            all_benchmarks[name] = res_dict

            # Print summary of the results
            print(
                f"✅ Process finished in {duration:.2f}s. Loaded in {res_dict['load_time_sec']:.2f}s."
            )
            for r in res_dict["results"]:
                # Print only first line of response or brief snippet
                snippet = r["response"].split("\n")[0][:80]
                print(f"  - Prompt: '{r['prompt']}'")
                print(
                    f"    Response: {snippet!r} (generated {r['tokens_generated']} tokens at {r['tokens_per_sec']:.2f} tok/s)"
                )

            os.remove(temp_file)
        else:
            print(f"❌ Subprocess failed with exit code {proc.returncode}")
            print(f"Stderr: {proc.stderr}")
            if os.path.exists(temp_file):
                os.remove(temp_file)

    # Save final results
    output_json = os.path.join(
        PROJECT_ROOT, "docs", "reports", "factual_audit_phase_3.3_results.json"
    )
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_benchmarks, f, indent=4, ensure_ascii=False)

    print(f"\n[+] Benchmarks written to: {output_json}")


if __name__ == "__main__":
    main()
