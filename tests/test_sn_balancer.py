import numpy as np
from gaje.processing.balancer import SignalToNoiseBalancer
from gaje.nn.stabilized import GenomicLayer

def test_sn_balancer_logic():
    print("🔬 Testeando Lógica del Signal-to-Noise Balancer...")
    balancer = SignalToNoiseBalancer(target_drift=0.001)
    
    # Simular una señal con mucho ruido
    act_orig = np.random.randn(100)
    act_noisy = act_orig + np.random.normal(0, 0.5, 100) # Noise sigma 0.5 -> MSE 0.25 -> Drift ~0.025
    
    initial_t = balancer.current_threshold
    print(f"[*] Threshold inicial: {initial_t:.4f}")
    
    # Ajustar basado en ruido alto
    new_t = balancer.adjust(act_orig, act_noisy)
    print(f"[*] Nuevo threshold (tras ruido alto): {new_t:.4f}")
    
    assert new_t < initial_t, "El balancer debería bajar el threshold ante ruido alto para capturar más anclas."
    
    # Simular señal limpia
    act_clean = act_orig + np.random.normal(0, 0.0001, 100)
    
    for _ in range(5):
        new_t = balancer.adjust(act_orig, act_clean)
        
    print(f"[*] Threshold final (tras señal limpia): {new_t:.4f}")
    assert new_t > initial_t * 0.5, "El balancer debería subir el threshold ante señal limpia."

def test_precision_mask_generation():
    print("\n🔬 Testeando Generación de Máscara de Precisión (Fase 12)...展")
    balancer = SignalToNoiseBalancer()
    
    # Simular entropía por dimensión (768 dims)
    entropy = np.random.uniform(0, 2.0, 768)
    # Hacer algunas dimensiones muy 'importantes' (alta entropía)
    entropy[:10] = 5.0
    
    mask = balancer.generate_precision_mask(entropy, fidelity_level=0.2)
    
    count_2bit = np.sum(mask == 0)
    count_4bit = np.sum(mask == 1)
    count_6bit = np.sum(mask == 2)
    
    print(f"[*] Distribución de Precisión: 2-bit={count_2bit}, 4-bit={count_4bit}, 6-bit={count_6bit}")
    
    assert count_6bit > 0, "Debería haber dimensiones identificadas para 6-bit."
    assert count_2bit > count_6bit, "La mayoría de las dimensiones deberían seguir en 2-bit para eficiencia."
    assert mask[0] == 2, "La dimensión 0 (alta entropía) debería tener precisión máxima (6-bit)."

if __name__ == '__main__':
    test_sn_balancer_logic()
    test_precision_mask_generation()
    print("\n✅ TESTS DE SN BALANCER COMPLETADOS.")
