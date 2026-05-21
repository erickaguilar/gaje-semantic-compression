import sys
import numpy as np
from semantic import DNASemanticRecord

try:
    from gaje.core import _impl as dna_semantic_compression
except ImportError:
    print("Error: Library not built. Run 'maturin develop' first.")
    sys.exit(1)


def main():
    # 1. Initialize the engine
    record_handler = DNASemanticRecord(dna_semantic_compression)

    # 2. Sample Data (1536 dims - Gemini standard)
    text = "The AI system memory experienced a critical error"
    embedding = np.random.uniform(-1, 1, 1536).astype(np.float32).tolist()

    # 3. Pack into Genomic Strand
    chromosome = record_handler.pack(text, embedding)

    # 4. Results
    orig_size = len(text.encode("utf-8")) + (len(embedding) * 4)
    final_size = len(chromosome)
    reduction = (1 - (final_size / orig_size)) * 100

    print("🧬 DNA SEMANTIC COMPRESSION DEMO 🧬")
    print("-" * 40)
    print(f"Original Text: '{text}'")
    print(f"Original Size: {orig_size} bytes")
    print(f"DNA Packed Size: {final_size} bytes")
    print(f"Total Space Saved: {reduction:.2f}%")
    print("-" * 40)
    print("Chromosome Sample (Hex):")
    print(f"{chromosome[:40].hex()}...")


if __name__ == "__main__":
    main()
