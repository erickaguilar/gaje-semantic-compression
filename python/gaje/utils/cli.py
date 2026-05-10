import argparse
import numpy as np
from gaje.core.archive import GAJEArchive
from gaje.core import _impl as engine

def main():
    parser = argparse.ArgumentParser(description="GAJE Protocol CLI Tool")
    subparsers = parser.add_subparsers(dest="command")

    # Info Command
    subparsers.add_parser("info", help="Show system info")

    # Search Command
    search_parser = subparsers.add_parser("search", help="Search in a .gaje archive")
    search_parser.add_argument("file", help="Path to .gaje file")
    search_parser.add_argument(
        "--dims", type=int, default=100, help="Vector dimensions"
    )

    args = parser.parse_args()

    if args.command == "info":
        print("🧬 GAJE Protocol v0.5.0")
        print("Status: Production-Ready (Anchor Cloning Enabled)")

    elif args.command == "search":
        archive = GAJEArchive.load(args.file)
        print(f"[*] Loaded archive with {len(archive.entries)} entries.")
        print(f"[*] Codebook Centroids: {archive.codebook['centroids']}")

        # Simulate a query
        print("\n[!] Simulating search for 'test query'...")
        q_vec = np.random.normal(0, 0.5, args.dims).tolist()
        db_dna = [e[1] for e in archive.entries]

        results = engine.dna_similarity_search_adc(
            q_vec, db_dna, archive.codebook["centroids"]
        )

        for i in range(min(3, len(results))):
            idx, dist = results[i]
            print(f"  {i+1}. {archive.entries[idx][0]} | Dist: {dist:.4f}")


if __name__ == "__main__":
    main()
