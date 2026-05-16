import os
import numpy as np
import time
from gaje.nn.stabilized import GenomicLLM

class PerplexityValidator:
    def __init__(self, model_path, num_blocks=4):
        print(f"🧬 Inicializando Organismo para PPL (Bloques: {num_blocks})")
        # Cargamos el modelo usando la lógica estabilizada (chunked)
        self.llm = GenomicLLM(model_path, num_blocks=num_blocks)
        
    def calculate_ppl(self, text):
        """
        Calcula la Perplexity usando inferencia incremental nativa (KV-Cache).
        PPL = exp(-1/N * sum(log P(token_i | tokens_<i)))
        """
        tokens = self.llm.tokenizer.encode(text, add_special_tokens=False)
        n_tokens = len(tokens)
        
        if n_tokens < 2:
            return 0.0

        print(f"[*] Evaluando coherencia sobre {n_tokens} tokens...")
        log_likelihoods = []
        
        # Limpiamos caché antes de empezar
        self.llm.rust_llm.clear_cache()
        
        start_eval = time.time()
        
        # Procesamiento inicial (Primer token)
        logits = self.llm.forward(tokens[0], clear_cache=False)[0]
        
        for i in range(1, n_tokens):
            target_id = tokens[i]
            
            # 1. Softmax sobre los logits del paso anterior
            probs = np.exp(logits - np.max(logits))
            probs /= probs.sum()
            
            # 2. Probabilidad del token real
            prob_target = max(probs[target_id], 1e-10)
            log_likelihoods.append(np.log(prob_target))
            
            # 3. Siguiente paso de inferencia (Incremental)
            logits = self.llm.forward(target_id, clear_cache=False)[0]
            
            if i % 10 == 0:
                print(f"    [~] Progreso: {i}/{n_tokens} tokens evaluados...", flush=True)

        duration = time.time() - start_eval
        avg_log_likelihood = np.mean(log_likelihoods)
        ppl = np.exp(-avg_log_likelihood)
        
        return ppl, duration

def run_ppl_test():
    model_path = "/data/data/com.termux/files/home/models/qwen2-0_5b-q8_0.gguf"
    if not os.path.exists(model_path):
        print("❌ Error: No se encuentra el modelo Qwen2 en la ruta estándar.")
        return

    validator = PerplexityValidator(model_path, num_blocks=4)
    
    test_text = (
        "El protocolo GAJE transforma la información semántica en cadenas de ADN sintético. "
        "Mediante el uso de un motor de búsqueda por grafos y cuantización genómica, "
        "es posible ejecutar modelos de lenguaje en dispositivos con recursos limitados."
    )
    
    ppl_score, duration = validator.calculate_ppl(test_text)
    
    print("\n" + "="*45)
    print(f"📊 REPORTE DE PERPLEXITY (GAJE 2-BIT)")
    print("="*45)
    print(f"✅ Perplexity (PPL):     {ppl_score:.4f}")
    print(f"✅ Tiempo de Evaluación: {duration:.2f} s")
    print(f"✅ Velocidad:           {len(validator.llm.tokenizer.encode(test_text))/duration:.2f} tok/s")
    print("="*45)
    
    if ppl_score < 100:
        print("🔥 ESTADO: EXCELENTE. Alta fidelidad semántica.")
    elif ppl_score < 500:
        print("⚠️ ESTADO: ACEPTABLE. Hay ruido pero el sentido se mantiene.")
    else:
        print("❌ ESTADO: CRÍTICO. Pérdida de coherencia estructural.")

if __name__ == "__main__":
    run_ppl_test()
