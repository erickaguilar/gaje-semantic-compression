import os
import numpy as np
from gaje.core import _impl as dna_semantic_compression
import gguf
import time
from transformers import AutoTokenizer

# Re-usamos la lógica estabilizada del pipeline completo
from gaje.processing.pipeline import GenomicLLM

class PerplexityValidator:
    def __init__(self, model_path, num_blocks=4):
        print(f"🧬 Inicializando Validador de Perplexity (Bloques: {num_blocks})")
        self.llm = GenomicLLM(model_path, num_blocks=num_blocks)
        
    def calculate_ppl(self, text):
        """
        Calcula la Perplexity: exp(-1/N * sum(log P(token_i | tokens_<i)))
        """
        tokens = self.llm.tokenizer.encode(text, add_special_tokens=False)
        n_tokens = len(tokens)
        
        if n_tokens < 2:
            return 0.0

        print(f"[*] Analizando {n_tokens} tokens para PPL...")
        log_likelihoods = []
        
        # Procesar secuencialmente para medir verosimilitud
        # Nota: En una inferencia real usaríamos KV-Cache, aquí aislamos por token
        for i in range(1, n_tokens):
            context = self.llm.tokenizer.decode(tokens[:i])
            target_token_id = tokens[i]
            
            # Obtener logits del modelo genómico
            # Modificamos ligeramente la llamada para obtener logits brutos
            token_ids = self.llm.tokenizer.encode(context, add_special_tokens=False)
            last_id = token_ids[-1]
            
            # 1. Recuperar ADN del Token
            stride = self.llm.embeddings.linear.stride
            start, end = last_id * stride, (last_id + 1) * stride
            dna_strand = self.llm.embeddings.linear.database[start:end]
            x = np.array(dna_semantic_compression.dequantize_embedding(list(dna_strand), self.llm.n_embd, self.llm.embeddings.linear.centroids))
            
            # 2. Inferencia Genómica
            for block in self.llm.blocks:
                x = block.forward(x, pos=i-1)
                
            # 3. Logits Head
            logits = self.llm.lm_head.forward(x)
            
            # 4. Softmax Local
            probs = np.exp(logits - np.max(logits))
            probs /= probs.sum()
            
            # 5. Verosimilitud del token real
            prob_target = probs[target_token_id]
            # Clip para evitar log(0)
            log_likelihoods.append(np.log(max(prob_target, 1e-10)))
            
            if i % 5 == 0:
                print(f"    [~] Progreso: {i}/{n_tokens} tokens evaluados...", flush=True)

        avg_log_likelihood = np.mean(log_likelihoods)
        ppl = np.exp(-avg_log_likelihood)
        return ppl

def run_ppl_test():
    model_path = "/data/data/com.termux/files/home/models/qwen2-0_5b-q8_0.gguf"
    if not os.path.exists(model_path):
        print("❌ Modelo no encontrado.")
        return

    validator = PerplexityValidator(model_path, num_blocks=4)
    
    # Texto de prueba: Un párrafo técnico sobre genética y computación
    test_text = (
        "El Protocolo GAJE utiliza estructuras biológicas para comprimir información semántica. "
        "Al transformar vectores de alta dimensión en cadenas de ADN de dos bits, logramos "
        "una eficiencia sin precedentes en dispositivos móviles con memoria limitada."
    )
    
    print(f"\n📝 Texto de Validación: '{test_text[:50]}...'")
    
    start_time = time.time()
    ppl_score = validator.calculate_ppl(test_text)
    duration = time.time() - start_time
    
    print("\n" + "="*45)
    print(f"📊 REPORTE DE PERPLEXITY (GAJE 2-BIT)")
    print("="*45)
    print(f"✅ Perplexity (PPL):     {ppl_score:.4f}")
    print(f"✅ Tiempo de Evaluación: {duration:.2f} s")
    print(f"✅ Tokens Procesados:    {len(validator.llm.tokenizer.encode(test_text))}")
    print("="*45)
    
    if ppl_score < 50:
        print("🔥 ESTADO: EXCELENTE. El modelo mantiene coherencia gramatical.")
    elif ppl_score < 150:
        print("⚠️ ESTADO: ACEPTABLE. Hay ruido predictivo, pero el sentido se conserva.")
    else:
        print("❌ ESTADO: CRÍTICO. El modelo ha perdido la lógica secuencial.")

if __name__ == "__main__":
    run_ppl_test()
