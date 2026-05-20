import numpy as np
import os
import sys

# Asegurar que usamos el código local de 'python/'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

from gaje.processing.balancer import SignalToNoiseBalancer

def print_heatmap(mask, block_width=32):
    """
    Imprime un mapa de calor visual en la terminal usando bloques ANSI.
    🟩 = 2-bit (Base, Baja Entropía)
    🟨 = 4-bit (Epigenético, Media Entropía)
    🟥 = 6-bit (Triplete, Alta Entropía)
    """
    colors = {
        0: "\033[92m█\033[0m", # Verde
        1: "\033[93m█\033[0m", # Amarillo
        2: "\033[91m█\033[0m"  # Rojo
    }
    
    print("\n" + "=" * (block_width * 2 + 4))
    for i in range(0, len(mask), block_width):
        row = mask[i:i+block_width]
        line = " ".join([colors[val] for val in row])
        print(f"| {line} |")
    print("=" * (block_width * 2 + 4) + "\n")

def main():
    print("🧬 GAJE PROTOCOL: VISUALIZACIÓN DE METABOLISMO (Fase 12)")
    print("=" * 60)
    print("Esta demo ilustra cómo el SignalToNoiseBalancer enruta")
    print("la memoria dinámicamente hacia hebras de mayor precisión")
    print("basándose en la entropía de Shannon (Struct-of-Arrays).\n")

    # 1. Simulación de una capa neuronal (ej. 1024 dimensiones)
    dim = 1024
    print(f"[*] Simulando una matriz de {dim} dimensiones...")
    
    # Generar entropía artificial con algunos picos (outliers)
    base_entropy = np.random.normal(loc=0.5, scale=0.1, size=dim)
    # Inyectar picos (alta entropía)
    outlier_indices = np.random.choice(dim, size=int(dim * 0.05), replace=False)
    base_entropy[outlier_indices] += np.random.normal(loc=1.5, scale=0.5, size=len(outlier_indices))
    
    # 2. Inicializar el Balancer
    balancer = SignalToNoiseBalancer()
    
    # 3. Generar la "Máscara Metabólica"
    # fidelity_level = 0.8 significa que conservará más bits para el top 20%
    print("[*] Aplicando SignalToNoiseBalancer (Nivel de fidelidad: 0.8)...")
    mask = balancer.generate_precision_mask(base_entropy, fidelity_level=0.8)
    
    # 4. Estadísticas
    count_2bit = np.sum(mask == 0)
    count_4bit = np.sum(mask == 1)
    count_6bit = np.sum(mask == 2)
    
    total = len(mask)
    
    print("\n📊 DISTRIBUCIÓN DEL ADN METABÓLICO:")
    print(f"   🟩 Hebras Base (2-bit):       {count_2bit} ({(count_2bit/total)*100:.1f}%) - MatMul Masivo SIMD")
    print(f"   🟨 Hebras Epigenéticas (4-bit): {count_4bit} ({(count_4bit/total)*100:.1f}%) - Corrección de Señal")
    print(f"   🟥 Hebras Tripletes (6-bit):    {count_6bit} ({(count_6bit/total)*100:.1f}%) - Preservación Crítica")
    
    print("\n🧬 MAPA DE CALOR DEL ADN (Fragmento 256 dims):")
    # Mostramos solo un fragmento para que quepa en pantalla
    print_heatmap(mask[:256], block_width=32)

    # 5. Implicación de Memoria SoA
    # 2 bits por dim
    mem_base = count_2bit * 2
    # 4 bits por dim
    mem_epi = count_4bit * 4
    # 6 bits por dim
    mem_tri = count_6bit * 6
    
    total_bits = mem_base + mem_epi + mem_tri
    avg_bits = total_bits / total
    
    print("💾 IMPACTO EN MEMORIA (Struct-of-Arrays):")
    print(f"   - Densidad de compresión efectiva: {avg_bits:.2f} bits/dimensión")
    print(f"   - Al aislar los picos (rojos/amarillos) en vectores dispersos,")
    print(f"     el motor en Rust puede multiplicar la matriz verde a máxima")
    print(f"     velocidad vectorial sin penalización de bifurcación (branching).")
    print("=" * 60)

if __name__ == "__main__":
    main()
