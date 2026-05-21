import numpy as np
import time
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

class GenomicTrainer:
    def __init__(self, model, lr=0.01, use_torch=True):
        self.model = model
        self.lr = lr
        self.use_torch = use_torch and HAS_TORCH
        if self.use_torch:
            print("[*] GenomicTrainer: Usando PyTorch para el cálculo de gradientes.")
        else:
            print("[*] GenomicTrainer: Usando NumPy para el cálculo de gradientes.")
        
    def _tokenize(self, text):
        if not hasattr(self.model, 'tokenizer') or self.model.tokenizer is None:
            return None
            
        # Intentar como tokenizers.Tokenizer (Rust)
        try:
            # Usar argumentos posicionales para evitar problemas con bindings
            encoding = self.model.tokenizer.encode(text)
            if hasattr(encoding, "ids"):
                return encoding.ids
            if hasattr(encoding, "get_ids"):
                return encoding.get_ids()
            return encoding
        except Exception:
            # Intentar como transformers.AutoTokenizer
            try:
                return self.model.tokenizer.encode(text, add_special_tokens=False)
            except Exception:
                return None

    def train_step(self, input_ids, target_ids):
        """
        Executes a single training step using Hybrid Autograd.
        """
        seq_len = len(input_ids)
        self.model.rust_llm.clear_cache()
        loss_total = 0.0
        
        for i in range(seq_len):
            token_id = int(input_ids[i])
            target_id = int(target_ids[i])
            
            logits_raw, h_norm_raw = self.model.rust_llm.forward_with_hidden(token_id, False)
            
            if self.use_torch:
                logits = torch.tensor(logits_raw, requires_grad=True)
                target = torch.tensor([target_id], dtype=torch.long)
                loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), target)
                loss_total += loss.item()
                loss.backward()
                grad_logits = logits.grad.numpy()
            else:
                probs = softmax(np.array(logits_raw))
                loss = -np.log(probs[target_id] + 1e-12)
                loss_total += loss
                grad_logits = probs.copy()
                grad_logits[target_id] -= 1.0
            
            self.model.rust_llm.refine_lm_head(h_norm_raw, grad_logits.tolist(), self.lr)
            
        return loss_total / seq_len

    def fit(self, dataset, epochs=10):
        print(f"[*] Iniciando entrenamiento Born-Genomic ({epochs} épocas)")
        for epoch in range(epochs):
            total_loss = 0
            count = 0
            start_time = time.time()
            
            for text in dataset:
                tokens = self._tokenize(text)
                if tokens is None or len(tokens) < 2:
                    continue
                
                loss = self.train_step(tokens[:-1], tokens[1:])
                total_loss += loss
                count += 1
            
            duration = time.time() - start_time
            if count > 0:
                avg_loss = total_loss / count
                ppl = np.exp(avg_loss) if avg_loss < 20 else 999.9
                print(f"    - Época {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | Tiempo: {duration:.2f}s")
            else:
                print(f"    - Época {epoch+1}/{epochs} | Sin datos válidos")

    def evaluate(self, dataset):
        total_loss = 0
        count = 0
        for text in dataset:
            tokens = self._tokenize(text)
            if tokens is None or len(tokens) < 2:
                continue
            
            loss = self.model.rust_llm.train_on_sequence(tokens, 0.0)
            total_loss += loss
            count += 1
        return total_loss / count if count > 0 else 0
