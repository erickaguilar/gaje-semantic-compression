import numpy as np
import json
import os

def generate_algebraic_centroids():
    """
    Generates 4 centroids based on the real parts of 16th roots of unity.
    These are algebraic numbers from the cyclotomic field Q(zeta_16).
    Values: cos(7pi/8), cos(5pi/8), cos(3pi/8), cos(pi/8)
    
    This provides a rigid algebraic structure for 2-bit quantization
    as suggested by the OpenAI Unit Distance research.
    """
    # Angles for 4 distinct points in the real projection of Q(zeta_16)
    angles = [7*np.pi/8, 5*np.pi/8, 3*np.pi/8, 1*np.pi/8]
    centroids = [float(np.cos(a)) for a in angles]
    
    # Sort them to map 2-bit states 00, 01, 11, 10
    centroids.sort()
    
    return centroids

def main():
    print("🧬 Generating Algebraic Codebook (Phase 5.0)")
    centroids = generate_algebraic_centroids()
    print(f"[*] Algebraic Centroids: {centroids}")
    
    output_dir = "models/core"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "algebraic_codebook.json")
    
    data = {
        "mode": "global",
        "bits": 2,
        "centroids": centroids,
        "metadata": {
            "source": "OpenAI Unit Distance Disproof Insight",
            "field": "Cyclotomic Q(zeta_16) Projection",
            "symmetry": "Point-Reflective",
            "values_desc": "cos(7pi/8), cos(5pi/8), cos(3pi/8), cos(pi/8)"
        }
    }
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
        
    print(f"[+] Codebook saved to: {output_path}")

if __name__ == "__main__":
    main()
