
import numpy as np
try:
    from gaje.core._impl import GenomicLinear, GenomicAttention
    print("✅ Importación exitosa")
except ImportError as e:
    print(f"❌ Error de importación: {e}")
    exit(1)

# Probar acceso a campos de GenomicLinear (mock data)
try:
    linear = GenomicLinear(
        database=bytes([0]*16),
        anchors_u8=bytes([0]*32),
        centroids=[0.0]*4,
        out_features=1,
        in_features=16,
        block_size=16
    )
    print("✅ GenomicLinear instanciado")
    print(f"   database: {len(linear.database)}")
    print(f"   centroids: {len(linear.centroids)}")
except Exception as e:
    print(f"❌ Error en GenomicLinear: {e}")

# Probar GenomicAttention si es posible
print("\nVerificando campos de GenomicAttention...")
try:
    # Intentar instanciar con valores mínimos si conocemos la firma
    # Por ahora solo listamos atributos disponibles vía dir()
    print(f"Atributos de GenomicAttention: {dir(GenomicAttention)}")
except Exception as e:
    print(f"❌ Error al inspeccionar GenomicAttention: {e}")
