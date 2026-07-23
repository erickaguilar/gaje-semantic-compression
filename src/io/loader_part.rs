            let attn_norm = {
                let n = Self::get_tensor_f32(&read_txn, &format!("{}attn_norm", p));
                if n.is_empty() {
                    eprintln!("[Loader Warning] Missing attn_norm for block {}", i);
                    vec![1.0f32; config.n_embd]
                } else {
                    n
                }
            };
