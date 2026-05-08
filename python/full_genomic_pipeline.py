import os
import numpy as np
import dna_semantic_compression
import gguf
import time
from transformers import AutoTokenizer

class GenomicLayer:
    def __init__(self, name, weights_f32):
        self.name = name
        # Qwen2 GGUF suele tener dimensiones transpuestas en algunos tensores
        # Aseguramos que out_features sea el primer eje para el motor Rust
        self.out_features, self.in_features = weights_f32.shape
        
        print(f"    [+] {name}: {self.in_features} -> {self.out_features}")
        
        # Max-Lloyd 2-bit training
        flat = weights_f32.flatten()
        std, mean = np.std(flat), np.mean(flat)
        self.thresholds = [mean - 0.9816 * std, mean, mean + 0.9816 * std]
        self.centroids = [mean - 1.510 * std, mean - 0.4528 * std, mean + 0.4528 * std, mean + 1.510 * std]
        
        # Quantize and load to Rust
        dna_batch = [
            dna_semantic_compression.quantize_embedding(w.tolist(), self.thresholds)
            for w in weights_f32
        ]
        self.engine = dna_semantic_compression.GajeIndex([], self.centroids)
        self.engine.add_batch(dna_batch)

    def forward(self, x):
        if isinstance(x, np.ndarray): x = x.tolist()
        return np.array(self.engine.genomic_linear_forward(x))

class GenomicTransformerBlock:
    def __init__(self, block_idx, reader):
        self.block_idx = block_idx
        self.layers = {}
        prefix = f"blk.{block_idx}."
        
        for tensor in reader.tensors:
            if tensor.name.startswith(prefix) and "weight" in tensor.name:
                if len(tensor.shape) == 2:
                    short_name = tensor.name.replace(prefix, "").replace(".weight", "")
                    self.layers[short_name] = GenomicLayer(tensor.name, tensor.data.astype(np.float32))
                elif "norm" in tensor.name:
                    short_name = tensor.name.replace(prefix, "").replace(".weight", "")
                    self.layers[short_name] = tensor.data.astype(np.float32)

    def rms_norm(self, x, weight_name, eps=1e-6):
        x = np.array(x)
        if x.size == 0: return x
        scale = self.layers.get(weight_name, 1.0)
        norm = x / np.sqrt(np.mean(x**2) + eps)
        # Asegurar que scale sea un escalar o tenga la misma forma que x
        return norm * scale

    def safe_silu(self, x):
        x = np.clip(x, -20, 20)
        return x * (1.0 / (1.0 + np.exp(-x)))

    def apply_rope(self, x, pos, base=10000.0):
        """
        Aplica Rotary Position Embeddings (RoPE) a un vector de activación.
        """
        dim = len(x)
        # Solo aplicamos a las dimensiones pares/impares para simular la rotación compleja
        res = np.zeros_like(x)
        theta = 1.0 / (base ** (np.arange(0, dim, 2)[:dim//2] / dim))
        m_theta = pos * theta
        
        cos_mt = np.cos(m_theta)
        sin_mt = np.sin(m_theta)
        
        for i in range(0, dim // 2):
            idx = i * 2
            if idx + 1 < dim:
                x0, x1 = x[idx], x[idx+1]
                res[idx] = x0 * cos_mt[i] - x1 * sin_mt[i]
                res[idx+1] = x1 * cos_mt[i] + x0 * sin_mt[i]
        return res

    def forward(self, x, pos=0):
        """
        Inferencia estabilizada con Multi-Query Attention Genómica (Qwen2 style).
        """
        x_in = np.array(x)
        
        # 1. Rama de Atención
        # Qwen2-0.5B usa GQA (Grouped Query Attention) o MQA
        # q: 896, k: 128, v: 128
        q_raw = np.array(self.layers['attn_q'].forward(x_in.tolist()))
        k_raw = np.array(self.layers['attn_k'].forward(x_in.tolist()))
        v_raw = np.array(self.layers['attn_v'].forward(x_in.tolist()))
        
        # RoPE se aplica a Q y K. 
        # Para K (128 dims), aplicamos a las primeras 128 de Q para el dot product
        q_rope = self.apply_rope(q_raw[:128], pos)
        k_rope = self.apply_rope(k_raw[:128], pos)
        
        # Score de atención (simplificado para el token actual)
        score = np.dot(q_rope, k_rope) / np.sqrt(128)
        score = np.clip(score, -10, 10)
        
        # Proyectar impacto (usamos V de 128 y lo expandimos/proyectamos)
        attn_impact_v = v_raw * score
        
        # Qwen2 requiere proyectar de vuelta al espacio hidden (896)
        # Como attn_output es 896x896, necesitamos que la entrada sea 896.
        # En una arquitectura real, esto sumaría todos los heads.
        # Aquí simulamos rellenando con ceros o repitiendo.
        v_expanded = np.zeros(896)
        v_expanded[:128] = attn_impact_v
        
        if 'attn_output' in self.layers:
            attn_out = self.layers['attn_output'].forward(v_expanded.tolist())[:len(x_in)]
        else:
            attn_out = v_expanded[:len(x_in)]
            
        x = self.rms_norm(x_in + attn_out, 'attn_norm')
        
        # 2. FFN (up -> gate -> down)
        up = self.layers['ffn_up'].forward(x)
        gate = self.layers['ffn_gate'].forward(x)
        h = up * self.safe_silu(gate)
        down = self.layers['ffn_down'].forward(h)[:len(x)]
        
        return self.rms_norm(x + down, 'ffn_norm')

class GenomicLLM:
    def __init__(self, model_path, num_blocks=2):
        print(f"🧬 Inicializando LLM Genómico desde: {model_path}")
        self.reader = gguf.GGUFReader(model_path)
        
        # Dimensión oculta de Qwen2-0.5B
        self.hidden_dim = 896
        
        # 1. Tokenizer y Embeddings
        print("[*] Configurando Frontend Genómico...")
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
        embd_tensor = next(t for t in self.reader.tensors if t.name == "token_embd.weight")
        # El tensor es (vocab, hidden) en algunos GGUF, pero gguf.py a veces lo muestra (hidden, vocab)
        # Forzamos la lógica: in=vocab, out=hidden
        w_embd = embd_tensor.data.astype(np.float32)
        if w_embd.shape[0] != self.hidden_dim:
            w_embd = w_embd.T # Asegurar (hidden, vocab)
            
        self.embeddings = GenomicLayer("embeddings", w_embd)
        
        # 2. Transformer Blocks
        self.blocks = []
        for i in range(num_blocks):
            print(f"[*] Genomizando Bloque {i}...")
            self.blocks.append(GenomicTransformerBlock(i, self.reader))
            
        # 3. Output Head (lm_head)
        print("[*] Configurando Backend Genómico (LM Head)...")
        try:
            head_tensor = next(t for t in self.reader.tensors if t.name == "output.weight")
            w_head = head_tensor.data.astype(np.float32)
            if w_head.shape[1] != self.hidden_dim: w_head = w_head.T
        except StopIteration:
            # Weight tying
            w_head = w_embd.T # (vocab, hidden)
            
        self.lm_head = GenomicLayer("lm_head", w_head)

    def generate(self, prompt, max_new_tokens=10):
        print(f"\n📝 Prompt: '{prompt}'")
        print("[*] Generando", end="", flush=True)
        
        generated_text = prompt
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        
        for _ in range(max_new_tokens):
            # Obtener el siguiente token usando ADN Genómico y RoPE
            # La posición es la longitud actual de la secuencia
            pos = len(token_ids)
            next_word = self.generate_step(generated_text, pos=pos, silent=True)
            generated_text += next_word
            token_ids = self.tokenizer.encode(generated_text, add_special_tokens=False)
            print(".", end="", flush=True)
            
        print(" [Hecho]")
        return generated_text

    def generate_step(self, text, pos=0, silent=False):
        # A. Tokenización
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        last_id = token_ids[-1]
        
        # B. Recuperar ADN del Token (Embedding)
        stride = self.embeddings.engine.stride
        start, end = last_id * stride, (last_id + 1) * stride
        dna_strand = self.embeddings.engine.database[start:end]
        
        # Inyectar al espacio de trabajo
        x = np.array(dna_semantic_compression.dequantize_embedding(list(dna_strand), self.hidden_dim, self.embeddings.centroids))
        
        # C. Inferencia por Bloques (ADN -> ADN)
        if not silent: print(f"[*] Procesando ADN a través de {len(self.blocks)} bloques genómicos con RoPE (pos={pos})...")
        for block in self.blocks:
            x = block.forward(x, pos=pos)
            
        # D. Proyección Final (De-Tokenizer Head)
        logits = self.lm_head.forward(x)
        next_token_id = np.argmax(logits)
        
        return self.tokenizer.decode([next_token_id])

def run_full_pipeline():
    model_path = "/data/data/com.termux/files/home/models/qwen2-0_5b-q8_0.gguf"
    if not os.path.exists(model_path):
        print("❌ Modelo no encontrado.")
        return

    # CARGA TOTAL: 24 bloques (El cerebro completo de Qwen2-0.5B)
    llm = GenomicLLM(model_path, num_blocks=24)
    
    prompt = "El ADN es la base de la"
    
    start = time.perf_counter()
    # Generar 20 tokens para ver coherencia real
    full_sentence = llm.generate(prompt, max_new_tokens=20)
    end = time.perf_counter()
    
    print(f"\n✨ Resultado Final (LLM Genómico Completo):")
    print(f"'{full_sentence}'")
    print(f"⏱️ Tiempo total de generación: {(end-start)*1000:.2f} ms")
    
    # RAM Stats
    total_dna_bytes = (
        len(llm.embeddings.engine.database) + 
        sum([sum([len(ly.engine.database) for ly in bl.layers.values()]) for bl in llm.blocks]) +
        len(llm.lm_head.engine.database)
    )
    print(f"\n📊 RAM Genómica en uso: {total_dna_bytes/(1024*1024):.2f} MB")
    print(f"📉 RAM Float32 Equivalente: {(total_dna_bytes/(1024*1024))*16:.2f} MB")

if __name__ == "__main__":
    run_full_pipeline()
