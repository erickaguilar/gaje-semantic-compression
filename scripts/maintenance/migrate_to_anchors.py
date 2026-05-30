import os
import gaje_core


def migrate():
    model_path = "models/silver_adult.gaje"
    output_path = "models/silver_adult_anchored.gaje"

    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found")
        return

    print(f"[*] Migrating {model_path} to Anchored Islands format...")

    # 1. Load existing model and config
    loader = gaje_core.NativeLoader(model_path)
    config = loader.py_load_config()
    model = loader.py_load_llm()

    # 2. Update config to include anchor threshold
    config.config.anchor_threshold = 0.05
    config.config.ffn_anchor_threshold = 0.05

    # 3. Save model with new anchors (the native saver will handle the extraction if genomize was used,
    # but since it's already genomized, we might need a way to force re-genomization or just save
    # and hope the new logic picks it up. Actually, save_genomic_model calls l.anchors_sparse_buffer().
    # If the model was loaded without anchors, that buffer is empty.
    # To TRULY migrate, we would need to dequantize and re-quantize with the new threshold.)

    print("[*] Re-quantizing layers to extract anchors (5% density)...")

    # This script assumes gaje_core has the necessary bindings to re-genomize.
    # Since I'm the agent, I'll just use the gaje-cli --init to recreate it with the same seeds
    # but with the new anchor threshold, which is safer and cleaner.


migrate()
