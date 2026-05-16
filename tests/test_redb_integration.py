import os
import sys
import numpy as np

# Ensure we use the local package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

from gaje.nn.stabilized import GenomicLLM
from gaje.nn.configs import get_config

def test_save_and_load_db():
    print("🧪 GAJE PROTOCOL: TEST REDB INTEGRATION 🧪")
    
    db_path = "test_model.gaje"
    if os.path.exists(db_path):
        os.remove(db_path)
        
    config = get_config("gaje_native")
    
    # 1. Initialize born-genomic model
    print("[*] Initializing model...")
    model = GenomicLLM(num_blocks=2, config=config)
    
    # Check original value
    orig_centroids = model.blocks[0].attn_layer.q_gen.linear.centroids
    
    # 2. Save model to database
    print(f"[*] Saving model to {db_path}...")
    model.save(db_path)
    
    assert os.path.exists(db_path), "Database file was not created!"
    
    # 3. Load model from database
    print("[*] Loading model from database...")
    loaded_model = GenomicLLM.load_genomic(db_path)
    
    # 4. Verify some metadata
    assert loaded_model.n_blocks == 2
    assert loaded_model.n_embd == 768
    print("[*] Metadata verified successfully.")
    
    # Clean up
    if os.path.exists(db_path):
        os.remove(db_path)
        
    print("✅ INTEGRACIÓN DE BASE DE DATOS K-V COMPLETADA EXITOSAMENTE")

if __name__ == "__main__":
    test_save_and_load_db()
