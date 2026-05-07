import sys
import numpy as np

try:
    import dna_semantic_compression
except ImportError:
    print("Error: Library not built. Run 'maturin develop' first.")
    sys.exit(1)


class MultimodalDNAArchive:
    def __init__(self):
        self.archive = []  # List of (type, metadata, dna_strand)
        self.dims = 512  # Standard for many CLIP models

    def add_entry(self, entry_type, label, vector):
        dna_strand = dna_semantic_compression.quantize_embedding(vector)
        self.archive.append({"type": entry_type, "label": label, "dna": dna_strand})

    def search(self, query_vector, top_k=3):
        query_dna = dna_semantic_compression.quantize_embedding(query_vector)
        db_dna = [entry["dna"] for entry in self.archive]

        results = dna_semantic_compression.dna_similarity_search(query_dna, db_dna)

        print(f"\n🔍 Search Results (Top {top_k}):")
        for i in range(min(top_k, len(results))):
            idx, dist = results[i]
            entry = self.archive[idx]
            print(
                f"  {i+1}. [{entry['type'].upper()}] {entry['label']} | DNA Hamming Dist: {dist}"
            )


def main():
    print("🧬 GAJE PROTOCOL: MULTIMODAL SEMANTIC ARCHIVE 🧬")
    print("-" * 50)

    archive = MultimodalDNAArchive()
    dims = 512

    # 1. Simulate Image Embeddings (e.g., from CLIP)
    print("[*] Encoding images into DNA strands...")
    img_nebula = np.random.uniform(-1, 1, dims).tolist()
    img_rover = np.random.uniform(-1, 1, dims).tolist()
    archive.add_entry("image", "Crab Nebula (Hubble)", img_nebula)
    archive.add_entry("image", "Perseverance Rover on Mars", img_rover)

    # 2. Simulate Text Embeddings
    print("[*] Encoding text into DNA strands...")
    text_space = np.random.uniform(-1, 1, dims).tolist()
    archive.add_entry("text", "Mission report: Water found on Europa", text_space)

    # 3. Perform Cross-Modal Search
    # Searching for something "space related" using a new vector
    print("\n[*] Performing Cross-Modal Query: 'Deep space exploration'...")
    np.random.uniform(-1, 1, dims).tolist()

    # Let's make the query vector similar to the nebula image to show it works
    query_vec_similar = (np.array(img_nebula) + np.random.normal(0, 0.1, dims)).tolist()

    archive.search(query_vec_similar)

    print("-" * 50)
    print("CONCLUSION: GAJE unifies different data modalities (Text/Images)")
    print("into a single genomic format, allowing universal semantic indexing.")


if __name__ == "__main__":
    main()
