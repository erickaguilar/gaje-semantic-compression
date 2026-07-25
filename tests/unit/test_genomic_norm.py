import numpy as np
import os
import sys

# Asegurar que se usa el paquete local
sys.path.insert(0, os.path.abspath("python"))

from gaje.core import _impl as engine


def test_genomic_norm_evolution():
    print("🔬 Probando 'GenomicNorm Adaptativo' (Evolución de Coherencia)...")

    path = "models/checkpoints/gnorm_test.gaje"
    config = engine.ModelConfig(
        config=engine.ArchConfig(
            name="GNORM-Test",
            tokenizer_id="tokenizer",
            rope_base=1000000.0,
            ffn_act="swiglu",
            use_genomic_norm=True,  # Activamos GenomicNorm
        ),
        n_embd=256,
        n_head=4,
        n_head_kv=4,
        n_blocks=2,
        vocab_size=1000,
        eps=1e-6,
    )

    # 1. Inicialización
    print("[*] Inicializando organismo con GenomicNorm activo...")
    model = engine.init_born_genomic_model(path, config, 1000)

    # 2. Verificar parámetros iniciales
    h_scales = [b.h_scale for b in model.blocks]
    print(f"   - Homeostasis Scales iniciales: {h_scales}")
    assert all(h == 1.0 for h in h_scales), "Initial h_scale should be 1.0"

    # 3. Simular Mutación de Coherencia (Monte Carlo)
    print("[*] Aplicando mutación estocástica a la homeostasis...")
    deltas = model.mutate_all_homeostasis(0.1)
    new_h_scales = [b.h_scale for b in model.blocks]
    print(f"   - Deltas aplicados: {deltas}")
    print(f"   - Nuevas Homeostasis Scales: {new_h_scales}")

    assert any(
        h != 1.0 for h in new_h_scales
    ), "h_scale should have changed after mutation"

    # 4. Inferencia con Estabilización
    print("[*] Probando inferencia estabilizada...")
    logits = model.forward(42, True)
    print(f"   - Logits generados (primeros 5): {logits[:5]}")
    assert not np.isnan(logits).any(), "Logits should not be NaN with GenomicNorm"

    # 5. Rollback
    print("[*] Deshaciendo mutación...")
    model.undo_homeostasis_mutation(deltas)
    final_h_scales = [b.h_scale for b in model.blocks]
    print(f"   - Final Homeostasis Scales: {final_h_scales}")
    assert all(
        abs(h - 1.0) < 1e-6 for h in final_h_scales
    ), "Rollback should restore h_scale to 1.0"

    print("✅ GENOMIC NORM VALIDADO EXITOSAMENTE")


if __name__ == "__main__":
    try:
        test_genomic_norm_evolution()
    except Exception as e:
        print(f"❌ Error durante la validación: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
