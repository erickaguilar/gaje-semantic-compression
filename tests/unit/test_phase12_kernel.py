import os
import sys
import numpy as np
import pytest

# Asegurar que se usa el paquete local
sys.path.append(os.path.abspath("python"))
from gaje.core import _impl as dna_core

def test_mixed_precision_kernel_consistency():
    """
    Verifica que el kernel de precisión mixta (Phase 12) produzca el mismo resultado
    que la suma manual de las hebras (Base + Epi + Tri).
    """
    print("\n🔬 Testing Unified Mixed-Precision Kernel (Phase 12)...")
    
    # Configuración
    in_features = 16
    out_features = 1
    block_size = 16 # 1 bloque
    stride = 4      # 16 / 4
    
    # 1. Datos base (2-bit)
    weights_base = np.random.normal(0, 0.1, (out_features, in_features)).astype(np.float32)
    thresholds = [-0.1, 0.0, 0.1]
    centroids = [-0.15, -0.05, 0.05, 0.15]
    
    dna_base = b"".join([dna_core.quantize_embedding(row.tolist(), thresholds) for row in weights_base])
    
    # 2. Datos Epigenéticos (4-bit) - Solo para la mitad de las dimensiones
    weights_epi = np.random.normal(0, 0.05, (out_features, in_features)).astype(np.float32)
    epi_centroids = [-0.07, -0.02, 0.02, 0.07]
    
    # Máscara: Primeras 8 dimensiones son 4-bit (1), el resto 2-bit (0)
    # Stride es 4 (cada byte controla 4 dimensiones)
    # Así que los primeros 2 bytes de la máscara serán 1, los otros 2 serán 0
    mask = bytes([1, 1, 0, 0]) 
    
    # Creamos la base de datos epigenética compacta (solo para las dimensiones con mask >= 1)
    # En este caso, el kernel de Rust espera que epi_strands contenga los bytes para las columnas activas.
    # Como mask[0]=1 y mask[1]=1, epi_strands debe tener 2 bytes por fila.
    epi_strands = []
    for row in weights_epi:
        # Solo las dimensiones de los bloques marcados
        # Bloque 0 (dims 0-3), Bloque 1 (dims 4-7)
        row_epi_0 = dna_core.quantize_embedding(row[0:4].tolist(), thresholds)
        row_epi_1 = dna_core.quantize_embedding(row[4:8].tolist(), thresholds)
        epi_strands.append(row_epi_0 + row_epi_1)
    
    epi_database = b"".join(epi_strands)
    
    # 3. Datos de Triplete (6-bit) - Solo para el primer bloque
    weights_tri = np.random.normal(0, 0.02, (out_features, in_features)).astype(np.float32)
    tri_centroids = [-0.03, -0.01, 0.01, 0.03]
    
    # Actualizar máscara: Primer bloque es 6-bit (2), segundo es 4-bit (1), resto 2-bit (0)
    mask = bytes([2, 1, 0, 0])
    
    # tri_strands solo para mask >= 2 (solo el primer byte)
    tri_strands = []
    for row in weights_tri:
        row_tri_0 = dna_core.quantize_embedding(row[0:4].tolist(), thresholds)
        tri_strands.append(row_tri_0)
        
    tri_database = b"".join(tri_strands)
    
    # 4. Entrada
    input_vec = np.random.normal(0, 1.0, in_features).astype(np.float32)
    
    # 5. Ejecutar vía GenomicLinear (que ahora usa el kernel mixto)
    # Nota: GenomicLinear::new procesa las columas activas internamente
    model = dna_core.GenomicLinear(
        dna_base,
        b"", # anchors
        centroids * (out_features), # centroids per row/block
        out_features,
        in_features,
        block_size,
        [], # rmsnorm
        1e-6,
        mask,
        epi_database,
        epi_centroids * (out_features),
        tri_database,
        tri_centroids * (out_features)
    )
    
    output = model.forward(input_vec.tolist())
    
    # 6. Verificación manual (Dequantize + Dot Product)
    def manual_decode(dna, c, n):
        res = []
        for byte in dna:
            for s in range(4):
                shift = (3 - s) * 2
                bits = (byte >> shift) & 0b11
                idx = bits ^ (bits >> 1)
                res.append(c[idx])
        return np.array(res[:n])

    # Reconstrucción de pesos
    w_base_deq = manual_decode(dna_base, centroids, in_features)
    
    # Epi: solo primeras 8 dims (bytes 0 y 1)
    w_epi_deq = np.zeros(in_features)
    w_epi_deq[0:8] = manual_decode(epi_strands[0], epi_centroids, 8)
    
    # Tri: solo primeras 4 dims (byte 0)
    w_tri_deq = np.zeros(in_features)
    w_tri_deq[0:4] = manual_decode(tri_strands[0], tri_centroids, 4)
    
    w_total = w_base_deq + w_epi_deq + w_tri_deq
    expected = np.dot(w_total, input_vec)
    
    print(f"    Expected: {expected:.6f}")
    print(f"    Actual:   {output[0]:.6f}")
    
    assert np.allclose(output[0], expected, atol=1e-5)
    print("✅ Kernel consistency verified!")

if __name__ == "__main__":
    test_mixed_precision_kernel_consistency()
