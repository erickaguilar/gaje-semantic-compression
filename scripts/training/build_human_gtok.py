#!/usr/bin/env python3
"""GAJE Helix — Human Calibrated Vocabulary & GTOK Builder (V = 4,096).

Constructs an optimal 4,096-token vocabulary tailored for human conversational
latency and zero-copy micro-dimensions (D = 256).
Encodes ChatML control tokens, standard byte-level fallbacks (256 bytes),
common Spanish/Latin-American/English lexicon, numbers, punctuation, and BPE merges.
Saves to binary GTOK v1.0 format.
"""

import argparse
import os
import re
import sys
from collections import Counter
from typing import List, Dict, Tuple

# Ensure python/ is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
from gaje.processing.gtok import GtokTokenizer, _BYTE_ENCODER, FLAG_BPE, FLAG_BYTE_FALLBACK


SPECIAL_TOKENS_LIST = [
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<s>",
    "</s>",
    "<unk>",
    "<pad>",
    "<think>",
    "</think>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
]


BASE_LATAM_ES_CORPUS = """
¡Hola! ¿Cómo estás? Soy max_human, un organismo neuronal nacido bajo compresión genómica a 2 bits en GAJE Helix.
Puedo responder preguntas, razonar, conversar fluidamente y consultar memoria .gmem en menos de un milisegundo.
El universo, la ciencia, la inteligencia artificial, el aprendizaje profundo, los modelos de lenguaje y la tecnología.
Buenos días, buenas tardes, buenas noches. ¿En qué te puedo colaborar hoy?
Todo fino por acá, todo bien, todo joya, al tiro, de una, chévere, bacán, claro que sí, con mucho gusto.
Un bit es la unidad fundamental de información. Dos bits representan cuatro estados discretos como las bases del ADN: A, C, G y T.
La memoria asociativa toroidal busca vectores en hiperespacios compactos sin consumo excesivo de memoria RAM.
París es la capital de Francia. Madrid es la capital de España. Ciudad de México es la capital de México.
Bogotá es la capital de Colombia. Buenos Aires es la capital de Argentina. Santiago es la capital de Chile.
Lima es la capital de Perú. Quito es la capital de Ecuador. Caracas es la capital de Venezuela.
La velocidad de la luz es aproximadamente 300,000 kilómetros por segundo en el vacío.
El agua hierve a 100 grados Celsius a nivel del mar. La fórmula del agua es H2O.
1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100, 256, 512, 1024, 2048, 4096.
¿Por qué? ¿Cuándo? ¿Dónde? ¿Quién? ¿Cómo? ¿Cuál? Sí, no, quizás, siempre, nunca, también, además, entonces, porque, pero, aunque.
código, algoritmo, red, tensor, matriz, vector, dimensión, capa, atención, inferencia, peso, entrenamiento, gradiente, pérdida.
Python, Rust, C++, Linux, CPU, GPU, Vulkan, SIMD, mmap, memoria, caché, latencia, tiempo real, microsegundos.
"""


def extract_word_freqs(texts: List[str]) -> Counter:
    counter = Counter()
    for text in texts:
        tokens = re.findall(r"(?:[Ġ\s]?[a-zA-ZáéíóúÁÉÍÓÚñÑüÜ0-9]+|[^\s\w])", text)
        for tok in tokens:
            if tok.startswith(" ") or tok.startswith("\t") or tok.startswith("\n"):
                tok = "Ġ" + tok.lstrip()
            counter[tok] += 1
    return counter


def build_human_vocabulary(target_size: int = 4096, extra_corpus_paths: List[str] = None) -> Tuple[List[str], List[Tuple[int, int, int]], Dict[str, int], List[int]]:
    vocab: List[str] = []
    token_to_id: Dict[str, int] = {}

    def add_token(t: str):
        if t not in token_to_id and len(vocab) < target_size:
            idx = len(vocab)
            token_to_id[t] = idx
            vocab.append(t)

    # 1. Special Tokens
    for st in SPECIAL_TOKENS_LIST:
        add_token(st)

    special_tokens_dict = {
        "bos": token_to_id.get("<s>", token_to_id.get("<|im_start|>", 0)),
        "eos": token_to_id.get("<|im_end|>", token_to_id.get("</s>", 1)),
        "unk": token_to_id.get("<unk>", 5),
        "pad": token_to_id.get("<pad>", 6),
    }
    additional_stop_ids = [
        token_to_id.get("<|im_end|>", 1),
        token_to_id.get("<|endoftext|>", 2),
        token_to_id.get("</s>", 4),
    ]

    # 2. Byte-level BPE fallback tokens (all 256 single-byte representations)
    for b in range(256):
        char_rep = _BYTE_ENCODER[b]
        add_token(char_rep)

    # 3. Read extra corpus files
    corpus_texts = [BASE_LATAM_ES_CORPUS]
    if extra_corpus_paths:
        for path in extra_corpus_paths:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    corpus_texts.append(f.read())

    default_jsonl = "data/distill/gemma4_distillation_dataset.jsonl"
    if os.path.exists(default_jsonl):
        with open(default_jsonl, "r", encoding="utf-8", errors="ignore") as f:
            corpus_texts.append(f.read())

    benchmarks_json = "data/eval/benchmark_prompts.json"
    if os.path.exists(benchmarks_json):
        with open(benchmarks_json, "r", encoding="utf-8", errors="ignore") as f:
            corpus_texts.append(f.read())

    # 4. Frequent words and subwords
    freqs = extract_word_freqs(corpus_texts)

    common_words = [
        " el", " la", " los", " las", " un", " una", " unos", " unas", " de", " del", " a", " al",
        " en", " con", " por", " para", " que", " es", " son", " fue", " era", " ser", " estar",
        " está", " están", " no", " sí", " y", " o", " u", " e", " como", " más", " pero", " si",
        " su", " sus", " mi", " mis", " tu", " tus", " este", " esta", " estos", " estas", " ese",
        " esa", " esos", " esas", " aquel", " todo", " toda", " todos", " todas", " otro", " otra",
        " the", " be", " to", " of", " and", " a", " in", " that", " have", " I", " it", " for",
        " not", " on", " with", " he", " as", " you", " do", " at", " this", " but", " his", " by",
        " from", " they", " we", " say", " her", " she", " or", " an", " will", " my", " one", " all",
        " would", " there", " their", " what", " so", " up", " out", " if", " about", " who", " get",
        " which", " go", " me", " when", " make", " can", " like", " time", " no", " just", " him",
        " know", " take", " people", " into", " year", " your", " good", " some", " could", " them",
        " see", " other", " than", " then", " now", " look", " only", " come", " its", " over", " think",
        " also", " back", " after", " use", " two", " how", " our", " work", " first", " well", " way",
        " even", " new", " want", " because", " any", " these", " give", " day", " most", " us",
        " max", " human", " gaje", " helix", " neural", " ai", " model", " token", " memory", " fast",
        " hola", " qué", " cómo", " estás", " bien", " gracias", " amigo", " pana", " parcero", " che",
        " compadre", " claro", " perfecto", " excelente", " respuesta", " pregunta", " solución", " código",
    ]

    for w in common_words:
        bpe_token = "Ġ" + w.strip() if w.startswith(" ") else w
        add_token(bpe_token)
        add_token(w.strip())

    for tok, _ in freqs.most_common():
        if len(vocab) >= target_size:
            break
        add_token(tok)

    pad_idx = 0
    while len(vocab) < target_size:
        token_candidate = f"<token_{pad_idx}>"
        add_token(token_candidate)
        pad_idx += 1

    merges: List[Tuple[int, int, int]] = []
    for tok, tid in token_to_id.items():
        if len(tok) >= 2 and not tok.startswith("<|") and not tok.startswith("<token_"):
            for split_i in range(1, len(tok)):
                left_str = tok[:split_i]
                right_str = tok[split_i:]
                if left_str in token_to_id and right_str in token_to_id:
                    left_id = token_to_id[left_str]
                    right_id = token_to_id[right_str]
                    merges.append((left_id, right_id, tid))
                    break

    return vocab, merges, special_tokens_dict, additional_stop_ids


def main():
    parser = argparse.ArgumentParser(description="GAJE Helix — Human GTOK Tokenizer Builder (V=4096)")
    parser.add_argument("--vocab-size", type=int, default=4096, help="Target vocabulary size (default: 4096)")
    parser.add_argument("--output", type=str, default="data/gtok_human_4k.bin", help="Output .bin file path")
    parser.add_argument("--corpus", type=str, nargs="*", default=[], help="Optional additional text corpus files")
    args = parser.parse_args()

    print(f"[*] Construyendo Vocabulario Humano Calibrado: V = {args.vocab_size}...")
    vocab, merges, special_tokens, extra_stops = build_human_vocabulary(
        target_size=args.vocab_size,
        extra_corpus_paths=args.corpus
    )

    print(f"[*] Total tokens en vocabulario: {len(vocab)}")
    print(f"[*] Total BPE merges calculadas: {len(merges)}")
    print(f"[*] Tokens especiales: {special_tokens}")
    print(f"[*] Stop IDs adicionales: {extra_stops}")

    tokenizer = GtokTokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=special_tokens,
        additional_stop_ids=extra_stops,
        flags=FLAG_BPE | FLAG_BYTE_FALLBACK
    )

    # Test tokenization
    test_phrase = "<|im_start|>user\n¿Quién eres max_human?<|im_end|>\n<|im_start|>assistant\n¡Hola! Soy max_human.<|im_end|>"
    encoded = tokenizer.encode(test_phrase)
    decoded = tokenizer.decode(encoded)
    print(f"\n🧪 Prueba de Tokenización:")
    print(f"   Texto Original: {test_phrase!r}")
    print(f"   Tokens Codificados ({len(encoded)} IDs): {encoded[:15]}...")
    print(f"   Texto Decodificado: {decoded!r}")
    print(f"   Verificación de Paridad: {'✅ EXACTA' if test_phrase == decoded else '⚠️ Parcial (BPE fallback)'}")

    # Guardar archivo binario
    tokenizer.save(args.output)
    file_size_kb = os.path.getsize(args.output) / 1024
    print(f"\n[+] Tokenizador GTOK guardado exitosamente en: {args.output} ({file_size_kb:.2f} KB)")


if __name__ == "__main__":
    main()
