import os
import sys
import argparse

# Asegurar uso de código local
sys.path.insert(0, os.path.abspath("python"))

from gaje.nn.stabilized import GenomicLLM

def apply_recalibration(model_path, output_path, shift=0.1):
    print(f"[*] Cargando modelo para recalibración: {model_path}")
    llm = GenomicLLM.load_genomic(model_path)
    
    print(f"[*] Aplicando recalibración de fase masiva (shift={shift})...")
    llm.rust_llm.recalibrate_all_centroids(shift)

    print(f"[*] Aplicando Alineación de Vector en Equilibrio (VE) (strength=0.1)...")
    llm.rust_llm.apply_vector_equilibrium_alignment_all(0.1)
        
    print(f"[*] Guardando modelo recalibrado en: {output_path}")
    llm.save(output_path)
    print("✅ Recalibración completada.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/production/silver_adult_steel.gaje")
    parser.add_argument("--output", type=str, default="models/production/silver_adult_calibrated.gaje")
    parser.add_argument("--shift", type=float, default=0.2)
    args = parser.parse_args()
    
    apply_recalibration(args.model, args.output, args.shift)
