import json
import os

def prune_tokenizer(input_path, output_path, target_vocab_size=32768):
    print(f"[*] Pruning tokenizer from {input_path} to {target_vocab_size} tokens...")
    if not os.path.exists(input_path):
        print(f"[!] Error: {input_path} not found.")
        return

    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # 1. Prune vocab
    vocab = data['model']['vocab']
    pruned_vocab = {token: tid for token, tid in vocab.items() if tid < target_vocab_size}
    data['model']['vocab'] = pruned_vocab
    
    # 2. Prune merges
    if 'merges' in data['model']:
        merges = data['model']['merges']
        new_merges = []
        for m in merges:
            if isinstance(m, str):
                parts = m.split(' ')
            else:
                parts = m
                
            if len(parts) != 2:
                continue
                
            p1, p2 = parts
            res = p1 + p2
            if p1 in pruned_vocab and p2 in pruned_vocab and res in pruned_vocab:
                new_merges.append(m)
        
        data['model']['merges'] = new_merges
        print(f"[*] Kept {len(new_merges)} merges out of {len(merges)}.")

    # 3. Prune added_tokens
    if 'added_tokens' in data:
        data['added_tokens'] = [t for t in data['added_tokens'] if t['id'] < target_vocab_size]

    with open(output_path, 'w') as f:
        json.dump(data, f)
    print(f"[+] Pruned tokenizer saved to {output_path}")

if __name__ == "__main__":
    prune_tokenizer('models/core/tokenizer_bak.json', 'models/core/tokenizer.json', 32768)
