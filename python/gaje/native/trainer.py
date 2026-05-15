import torch
import torch.nn as nn
import numpy as np
import time
from typing import List, Dict, Any

class GenomicTrainer:
    """
    Hybrid trainer for GAJE-native models.
    Performs forward pass in Rust and uses PyTorch for gradient calculation
    and manual backpropagation to refine genomic centroids.
    """
    def __init__(self, model, lr=1e-4):
        from gaje.nn.stabilized import GenomicLLM
        self.model: GenomicLLM = model
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        print(f"[*] GenomicTrainer initialized (LR: {lr})")

    def _rms_norm_py(self, x, weight, eps=1e-6):
        x = np.array(x)
        rms = np.sqrt(np.mean(x**2) + eps)
        return (x / rms) * np.array(weight)

    def train_step(self, input_ids: List[int], target_ids: List[int]):
        """
        Performs a single training step (Online/Sequence).
        """
        total_loss = 0
        self.model.rust_llm.clear_cache()
        
        # We'll collect gradients and activations per token to update centroids
        # For simplicity in Phase 1, we process tokens sequentially
        for i, (tid, target) in enumerate(zip(input_ids, target_ids)):
            # 1. Forward Pass (Collecting activations)
            # We need to manually traverse to collect inputs for each layer
            
            # Embedding Layer
            h_emb = self.model.rust_llm.embeddings_forward(tid)
            
            # Blocks
            block_inputs = []
            h = h_emb
            pos = i
            for block in self.model.rust_llm.blocks:
                block_inputs.append(h)
                h = block.forward(h, pos)
            
            # Final Norm & Head
            h_final_norm = self._rms_norm_py(h, self.model.rust_llm.output_norm, self.model.rust_llm.eps)
            logits = self.model.rust_llm.lm_head.forward(h_final_norm)
            
            # 2. Loss & Output Gradients
            logits_t = torch.tensor(logits, requires_grad=True, dtype=torch.float32)
            target_t = torch.tensor([target], dtype=torch.long)
            
            loss = self.criterion(logits_t.unsqueeze(0), target_t)
            total_loss += loss.item()
            
            loss.backward()
            
            if logits_t.grad is None:
                print("[!] Error: Gradients not populated for logits_t")
                break
                
            grad_logits = logits_t.grad.numpy()
            
            # 3. Hybrid Backpropagation & Refinement
            # Update LM Head using the new native method to avoid cloning issues
            if i == 0:
                print(f"    [DEBUG] h_final_norm mean: {np.mean(np.abs(h_final_norm)):.6f}")
                print(f"    [DEBUG] grad_logits mean:  {np.mean(np.abs(grad_logits)):.6f}")
            
            self.model.rust_llm.refine_lm_head(h_final_norm.tolist(), grad_logits.tolist(), self.lr)
            
            # For Phase 1 Step 1, we focus on the Output Alignment.
            # Deep backprop through blocks requires dequantizing weights to compute dL/dx.
            # We will implement simplified "Target Propagation" or "Gradient Routing"
            # as specified in the roadmap.
            
            # Simplified Gradient Routing for blocks:
            # We use the grad_logits as a proxy to guide the last block's FFN refinement.
            # (In Step 2 we will implement full chain backprop)
            if len(self.model.rust_llm.blocks) > 0:
                last_block = self.model.rust_llm.blocks[-1]
                # We use a reduced grad for the block to prevent explosion
                block_grad = grad_logits[:len(h)] # Crude approximation for Phase 1 demo
                # last_block.refine_swiglu needs (input_norm, target)
                # But we have grads. For now we call refine_with_grads if exposed.
                # Actually, RustGenomicBlock has refine_swiglu(input_norm, target)
                # We can construct a pseudo-target: target = current_output - lr * grad
                
        return total_loss / len(input_ids)

def run_training_demo():
    from gaje.nn.stabilized import GenomicLLM
    from gaje.nn.configs import get_config
    
    print("\n" + "="*60)
    print("🚀 GAJE BORN-GENOMIC TRAINING: PHASE 1 DEMO")
    print("="*60)
    
    config = get_config("gaje_native")
    # For testing, we use a custom vocab size in the model
    # We'll patch the vocab size for the demo to see fast convergence
    model = GenomicLLM(num_blocks=1, config=config)
    
    # More reasonable LR for the final demo
    trainer = GenomicTrainer(model, lr=1.0) 

    # Tiny dataset: Very simple repetition
    input_ids = [1, 2, 3, 1, 2]
    target_ids = [2, 3, 1, 2, 3]

    print(f"[*] Training on {len(input_ids)} tokens (LR: {trainer.lr})...")

    initial_loss = 0
    for epoch in range(10):
        loss = trainer.train_step(input_ids, target_ids)
        if epoch == 0: initial_loss = loss
        print(f"Epoch {epoch+1:2d} | Loss: {loss:.6f}")
        if loss < 1e-4: break

    print(f"\n[*] Loss improvement: {initial_loss - loss:.6f}")

    # Final Validation
    print("\n[*] Final Validation (Predictions):")
    model.rust_llm.clear_cache()
    for tid, target in zip(input_ids, target_ids):
        logits = model.rust_llm.forward(tid, False)
        pred = np.argmax(logits)
        print(f"    Input: {tid} | Target: {target} | Predicted: {pred} {'✅' if pred == target else '❌'}")

        
    print("="*60)

if __name__ == "__main__":
    run_training_demo()
