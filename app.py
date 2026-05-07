import gradio as gr
import numpy as np
import dna_semantic_compression
from sentence_transformers import SentenceTransformer

# Load a lightweight model for the demo
print("Loading sentence-transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

def compress_demo(text):
    if not text:
        return "Please enter some text.", "", "", ""
    
    # 1. Get real embedding (384 dimensions for all-MiniLM-L6-v2)
    embedding = model.encode(text).tolist()
    dims = len(embedding)
    
    # 2. Quantize using our Rust Engine
    # Note: Our Rust engine expects 4 dimensions per byte (2-bit per dim)
    dna_strand = dna_semantic_compression.quantize_embedding(embedding)
    
    # 3. Calculate metrics
    orig_size_bytes = dims * 4 # float32
    dna_size_bytes = len(dna_strand)
    reduction = (1 - (dna_size_bytes / orig_size_bytes)) * 100
    
    # 4. Format Output
    dna_hex = dna_strand.hex()
    
    # Mapping based on our Rust implementation:
    # 00: A, 01: C, 11: G, 10: T
    mapping = {0b00: "A", 0b01: "C", 0b11: "G", 0b10: "T"}
    
    bases = []
    # We pack 4 dims per byte. Bits: [D1 D1 D2 D2 D3 D3 D4 D4]
    for byte in dna_strand[:40]: # Show first 40 bytes as bases
        for shift in [6, 4, 2, 0]:
            val = (byte >> shift) & 0b11
            bases.append(mapping[val])
    
    dna_visual = "".join(bases) + "..."
    
    metrics = (
        f"Original Size: {orig_size_bytes} bytes ({dims}-dim float32)\n"
        f"DNA Size: {dna_size_bytes} bytes\n"
        f"Compression Ratio: {orig_size_bytes/dna_size_bytes:.1f}x\n"
        f"Space Saved: {reduction:.2f}%"
    )
    
    return dna_visual, dna_hex[:64] + "...", metrics

# UI Design
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧬 GAJE: DNA Semantic Protocol")
    gr.Markdown("""
    ### Ultra-High Density Semantic Compression
    This demo transforms real AI embeddings (from `Sentence-Transformers`) into 2-bit genomic strands. 
    It reduces the memory footprint by **93.75%** while preserving semantic structure.
    """)
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Input Text", placeholder="Enter a sentence to compress...", lines=3)
            btn = gr.Button("Encode to DNA", variant="primary")
        
        with gr.Column():
            output_dna = gr.Textbox(label="Genomic Strand (Bases)", interactive=False)
            output_hex = gr.Textbox(label="Packed Binary (Hex)", interactive=False)
            output_metrics = gr.Textbox(label="Compression Metrics", interactive=False)

    gr.Examples(
        examples=[
            "The moon base reported a stable oxygen supply.",
            "Artificial Intelligence is the bridge between species.",
            "Searching for water signatures in the lunar south pole.",
            "Biological DNA is the ultimate storage medium."
        ],
        inputs=input_text
    )
    
    gr.Markdown("""
    ### Technical Note
    - **Engine:** Rust (SIMD-ready bit-packing)
    - **Model:** all-MiniLM-L6-v2 (384 dimensions)
    - **Quantization:** 2-bit Genomic ADC (Asymmetric Distance Computation)
    ---
    **Authorship:** Erick Aguilar (Vision) & Gemini (Algorithm) | *Dios cuida de la humanidad.*
    """)
    
    btn.click(compress_demo, inputs=input_text, outputs=[output_dna, output_hex, output_metrics])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
