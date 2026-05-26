import os
import sys
import json
import time
import numpy as np
from tokenizers import Tokenizer

# Asegurar uso de código local
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

from gaje.core._impl import NativeLoader, save_genomic_model, ArchConfig, ModelConfig, init_born_genomic_model
from gaje.nn.stabilized import GenomicLLM
from gaje.nn.trainer import GenomicTrainer

class ConsensusDistiller:
    """
    Council of Teachers (CoT) Distiller for Silver Fetus phase.
    Orchestrates multiple teacher models to guide a genomic student.
    """
    def __init__(self, student, teachers, student_tokenizer):
        self.student = student
        self.teachers = teachers
        self.tokenizer = student_tokenizer
        print(f"🧠 Council of Teachers inicializado con {len(teachers)} maestros.")

    def get_consensus_sequence_probs(self, text):
        """
        Aggregates logits from all teachers for a full sequence.
        Returns a list of probability distributions (one per token).
        """
        # 1. Tokenizar texto para cada maestro
        all_teacher_probs = []
        
        for t_name, teacher in self.teachers.items():
            tokens = teacher.tokenizer.encode(text, add_special_tokens=False)
            if hasattr(tokens, "ids"): tokens = tokens.ids
            
            # Forward del maestro para toda la secuencia
            logits_seq = teacher.forward(tokens, clear_cache=True)
            
            # Convertir a probabilidades y mapear al estudiante
            mapped_seq_probs = []
            for logits in logits_seq:
                probs = np.exp(logits - np.max(logits))
                probs /= probs.sum()
                
                # Mapeo optimizado (Top 50 es suficiente para destilación)
                student_probs = np.zeros(self.tokenizer.get_vocab_size())
                top_indices = np.argsort(probs)[-50:]
                for idx in top_indices:
                    token_str = teacher.tokenizer.decode([int(idx)])
                    student_id = self.tokenizer.token_to_id(token_str)
                    if student_id is not None:
                        student_probs[student_id] = probs[idx]
                
                # Normalizar mapeo
                s = student_probs.sum()
                if s > 0: student_probs /= s
                else: student_probs = np.ones_like(student_probs) / len(student_probs)
                
                mapped_seq_probs.append(student_probs)
            
            all_teacher_probs.append(mapped_seq_probs)
            
        # Promediar entre maestros (Consenso)
        num_teachers = len(all_teacher_probs)
        seq_len = len(all_teacher_probs[0])
        consensus_seq = []
        
        for i in range(seq_len):
            combined = np.zeros(self.tokenizer.get_vocab_size())
            for t_idx in range(num_teachers):
                combined += all_teacher_probs[t_idx][i]
            consensus_seq.append(combined / num_teachers)
            
        return consensus_seq

    def distill_step(self, text, trainer, lr):
        """
        Executes a distillation step: Student learns from Dataset + Teacher Consensus.
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if hasattr(tokens, "ids"): tokens = tokens.ids
        
        if len(tokens) < 2: return 0.0
        
        # 1. Obtener Consenso del Consejo para TODA la secuencia (O(N))
        teacher_consensus_seq = self.get_consensus_sequence_probs(text)
        
        self.student.rust_llm.clear_cache_py()
        total_loss = 0.0
        
        # Sincronizar longitudes (por si hay diferencias de tokenización mínimas)
        n_steps = min(len(tokens) - 1, len(teacher_consensus_seq))
        
        for i in range(n_steps):
            target_id = tokens[i+1]
            teacher_probs = teacher_consensus_seq[i]
            
            # 2. Inferencia del Estudiante
            logits_s, h_norm = self.student.rust_llm.forward_with_hidden(tokens[i], False)
            
            # 3. Cálculo de Gradiente (Híbrido)
            student_probs = np.exp(logits_s - np.max(logits_s))
            student_probs /= student_probs.sum()
            
            grad_ce = student_probs.copy()
            grad_ce[target_id] -= 1.0
            
            grad_kl = student_probs - teacher_probs
            
            distill_weight = 0.5 # Aumentamos peso de destilación
            grad_final = (1.0 - distill_weight) * grad_ce + distill_weight * grad_kl
            
            # 4. Refinamiento Nativo (LM Head)
            self.student.rust_llm.refine_lm_head(h_norm, grad_final.tolist(), lr)
            
            loss = -np.log(student_probs[target_id] + 1e-12)
            total_loss += loss
            
        return total_loss / n_steps

def main():
    print("🚀 Iniciando Protocolo 'Council of Teachers' (Stage 4 - Relational) 🚀")
    print("-" * 60)

    # 1. Cargar Estudiante (Silver Fetus v1)
    print("[*] Re-inicializando Silver Fetus con varianza...")
    
    name = "SilverFetus-v1"
    vocab_size = 32768
    n_embd = 512
    n_blocks = 12
    tokenizer_path = "models/core/tokenizer.json"

    arch = ArchConfig(name=name, version="0.9.8", tokenizer_id=tokenizer_path, rope_base=1000000.0)
    config = ModelConfig(config=arch, n_embd=n_embd, n_head=8, n_head_kv=8, n_blocks=n_blocks, vocab_size=vocab_size)
    
    student_path = "models/checkpoints/silverfetus-v1/model.gaje"
    rust_llm = init_born_genomic_model(student_path, config, vocab_size)
    
    # --- STAGE 4: INYECTAR TOPOLOGÍA ES ---
    topology_path = "models/core/topology_es.json"
    if os.path.exists(topology_path):
        print(f"[*] Inyectando Topología Relacional (CAM) desde {topology_path}...")
        rust_llm.load_topology(topology_path)
    else:
        print(f"⚠️ Advertencia: No se encontró topología en {topology_path}")

    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    student = GenomicLLM(None, config=config.config, n_embd=config.n_embd, num_blocks=config.n_blocks)
    student.rust_llm = rust_llm
    student.tokenizer = tokenizer
    student.config = config

    # 2. Cargar Consejo de Maestros
    teachers = {}
    print("[*] Cargando Maestro: SmolLM2-135M (259MB)...")
    teachers["smollm"] = GenomicLLM("models/gguf/smollm2-135m-f16.gguf")

    # 3. Preparar Dataset
    dataset_path = "data/datasets/dataset_es_ext.txt"
    with open(dataset_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if len(line.strip()) > 20]
    
    print(f"📊 Dataset cargado: {len(lines)} muestras para destilación.")

    # 4. Bucle de Destilación
    distiller = ConsensusDistiller(student, teachers, tokenizer)
    lr = 0.005
    epochs = 4
    
    print(f"[*] Iniciando Crianza con Guía Topológica ({epochs} épocas)...")
    for epoch in range(epochs):
        start_time = time.time()
        epoch_loss = 0.0
        count = 0
        
        samples = lines[:40]
        for text in samples: 
            loss = distiller.distill_step(text, None, lr)
            epoch_loss += loss
            count += 1
            if count % 10 == 0:
                print(f"    - Muestra {count}/{len(samples)} | Loss: {loss:.4f} | {time.time()-start_time:.1f}s")
        
        duration = time.time() - start_time
        avg_loss = epoch_loss / count if count > 0 else 0
        print(f"✅ Época {epoch+1} completada | Loss Promedio: {avg_loss:.4f} | Total: {duration:.2f}s")

    # 5. Guardar Modelo Destilado
    output_path = "models/checkpoints/silverfetus-distilled-cam.gaje"
    save_genomic_model(output_path, student.rust_llm, config, tokenizer_path)
    print(f"✨ Silver Fetus (Relational) destilado y guardado en {output_path}")

if __name__ == "__main__":
    main()
