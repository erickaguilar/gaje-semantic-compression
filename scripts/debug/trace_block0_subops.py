import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from gaje.nn.stabilized import GenomicLLM  # noqa: E402
from gaje.core import _impl as dna_semantic_compression  # noqa: E402


def trace_block0_subops():
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    hf_model.eval()

    fp32_path = os.path.join(PROJECT_ROOT, "models", "smollm2-135m-fp32.gaje")
    gaje_llm = GenomicLLM.load_genomic(fp32_path)

    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)

    print("\n=======================================================")
    print("🔬 AUDITORÍA SUB-OPERACIÓN POR SUB-OPERACIÓN EN BLOQUE 00")
    print("=======================================================")

    # Target token is index -1 ('is', token 314) at pos = 4
    input_tensor = torch.tensor([input_ids])

    # Extract PyTorch intermediate tensors in Block 0
    blk0_hf = hf_model.model.layers[0]

    with torch.no_grad():
        emb_all = hf_model.model.embed_tokens(input_tensor)
        _x_input_hf = emb_all[0, -1, :].numpy()  # Embedding input for token 314

        # 1. Attn Norm HF
        x_attn_norm_hf = blk0_hf.input_layernorm(emb_all)[0, -1, :].numpy()

        # 2. Q, K, V HF
        # Attention forward inside HF
        _head_dim = hf_model.config.hidden_size // hf_model.config.num_attention_heads
        q_hf = blk0_hf.self_attn.q_proj(blk0_hf.input_layernorm(emb_all))[
            0, -1, :
        ].numpy()
        k_hf = blk0_hf.self_attn.k_proj(blk0_hf.input_layernorm(emb_all))[
            0, -1, :
        ].numpy()
        v_hf = blk0_hf.self_attn.v_proj(blk0_hf.input_layernorm(emb_all))[
            0, -1, :
        ].numpy()

        # 3. Post Attention HF
        attn_out_tuple = blk0_hf.self_attn(
            hidden_states=blk0_hf.input_layernorm(emb_all),
            position_embeddings=hf_model.model.rotary_emb(
                emb_all, torch.arange(len(input_ids)).unsqueeze(0)
            ),
        )
        attn_out_hf = attn_out_tuple[0]
        attn_proj_hf = blk0_hf.self_attn.o_proj(attn_out_hf)[0, -1, :].numpy()
        x_post_attn_hf = (emb_all + blk0_hf.self_attn.o_proj(attn_out_hf))[
            0, -1, :
        ].numpy()

        x_ffn_norm_hf_t = blk0_hf.post_attention_layernorm(
            emb_all + blk0_hf.self_attn.o_proj(attn_out_hf)
        )
        x_ffn_norm_hf = x_ffn_norm_hf_t[0, -1, :].numpy()

        # 5. FFN Gate, Up, Down HF
        gate_t = blk0_hf.mlp.gate_proj(x_ffn_norm_hf_t)
        up_t = blk0_hf.mlp.up_proj(x_ffn_norm_hf_t)
        gate_hf = gate_t[0, -1, :].numpy()
        up_hf = up_t[0, -1, :].numpy()
        _ffn_out_hf = blk0_hf.mlp.down_proj(blk0_hf.mlp.act_fn(gate_t) * up_t)[
            0, -1, :
        ].numpy()

    # Now run Rust Block 00 subops
    mock_blk0 = gaje_llm.blocks[0]

    # Input x_rust
    x_rust = gaje_llm.embeddings.linear.get_row(input_ids[-1])
    x_rust = np.array(x_rust, dtype=np.float32)

    # 1. Test Input RMSNorm
    attn_norm_w = blk0_hf.input_layernorm.weight.detach().numpy()
    eps = blk0_hf.input_layernorm.variance_epsilon

    # RMSNorm calculation manually: x / sqrt(mean(x^2) + eps) * weight
    rms_val = np.sqrt(np.mean(x_rust**2) + eps)
    x_attn_norm_rust = (x_rust / rms_val) * attn_norm_w

    cos_attn_norm = np.dot(x_attn_norm_hf, x_attn_norm_rust) / (
        np.linalg.norm(x_attn_norm_hf) * np.linalg.norm(x_attn_norm_rust) + 1e-9
    )
    print(
        f"1. Input RMSNorm:   CosSim: {cos_attn_norm:.6f} | HF Norm: {np.linalg.norm(x_attn_norm_hf):.4f} | Rust Norm: {np.linalg.norm(x_attn_norm_rust):.4f}"
    )

    # 2. Test Q Projection
    q_rust = np.array(
        mock_blk0.attn_layer.q_gen.linear.forward(x_attn_norm_rust.tolist(), False)
    )
    cos_q = np.dot(q_hf, q_rust) / (
        np.linalg.norm(q_hf) * np.linalg.norm(q_rust) + 1e-9
    )
    print(
        f"2. Q Projection:     CosSim: {cos_q:.6f} | HF Norm: {np.linalg.norm(q_hf):.4f} | Rust Norm: {np.linalg.norm(q_rust):.4f}"
    )

    # 3. Test K Projection
    k_rust = np.array(
        mock_blk0.attn_layer.k_gen.linear.forward(x_attn_norm_rust.tolist(), False)
    )
    cos_k = np.dot(k_hf, k_rust) / (
        np.linalg.norm(k_hf) * np.linalg.norm(k_rust) + 1e-9
    )
    print(
        f"3. K Projection:     CosSim: {cos_k:.6f} | HF Norm: {np.linalg.norm(k_hf):.4f} | Rust Norm: {np.linalg.norm(k_rust):.4f}"
    )

    # 4. Test V Projection
    v_rust = np.array(
        mock_blk0.attn_layer.v_gen.linear.forward(x_attn_norm_rust.tolist(), False)
    )
    cos_v = np.dot(v_hf, v_rust) / (
        np.linalg.norm(v_hf) * np.linalg.norm(v_rust) + 1e-9
    )
    print(
        f"4. V Projection:     CosSim: {cos_v:.6f} | HF Norm: {np.linalg.norm(v_hf):.4f} | Rust Norm: {np.linalg.norm(v_rust):.4f}"
    )

    # 5. Attention Core (RoPE + Softmax + KV Cache)
    rope_style = "split"
    attn_obj = dna_semantic_compression.GenomicAttention(
        gaje_llm.n_head,
        gaje_llm.n_head_kv,
        gaje_llm.head_dim,
        blk0_hf.input_layernorm.weight.detach().numpy().tolist(),
        blk0_hf.input_layernorm.variance_epsilon,
        gaje_llm.rope_base,
        rope_style,
    )
    # Forward token by token through attention object up to pos = 4
    for p_idx in range(len(input_ids)):
        p_emb = emb_all[0, p_idx, :].numpy()
        p_norm = (p_emb / np.sqrt(np.mean(p_emb**2) + eps)) * attn_norm_w
        p_q = np.array(
            mock_blk0.attn_layer.q_gen.linear.forward(p_norm.tolist(), False)
        )
        p_k = np.array(
            mock_blk0.attn_layer.k_gen.linear.forward(p_norm.tolist(), False)
        )
        p_v = np.array(
            mock_blk0.attn_layer.v_gen.linear.forward(p_norm.tolist(), False)
        )
        attn_out_rust = np.array(
            attn_obj.forward_attention(p_q.tolist(), p_k.tolist(), p_v.tolist(), p_idx)
        )

    # Print PyTorch attention weights vs Rust attention weights for token 4
    with torch.no_grad():
        q_t_4d = (
            blk0_hf.self_attn.q_proj(blk0_hf.input_layernorm(emb_all))
            .view(1, len(input_ids), 9, 64)
            .transpose(1, 2)
        )
        k_t_4d = (
            blk0_hf.self_attn.k_proj(blk0_hf.input_layernorm(emb_all))
            .view(1, len(input_ids), 3, 64)
            .transpose(1, 2)
        )
        pos_ids = torch.arange(len(input_ids)).unsqueeze(0)
        cos_hf, sin_hf = hf_model.model.rotary_emb(emb_all, pos_ids)
        q_rot_t, k_rot_t = apply_rotary_pos_emb(q_t_4d, k_t_4d, cos_hf, sin_hf)

        # q_rot_t and k_rot_t are already (1, 9, 5, 64) and (1, 3, 5, 64)

        print(
            f"[*] Rust block 0 rope_style in model: {getattr(mock_blk0.rust_block, 'rope_style', 'unknown')}"
        )
        scale = 1.0 / np.sqrt(64)
        scores_hf_h0 = (
            (q_rot_t[0, 0, -1, :] @ k_rot_t[0, 0, :, :].T * scale)
            .softmax(dim=-1)
            .numpy()
        )
        print(f"HF Head 0 Softmax Scores at pos 4: {scores_hf_h0}")

        # Rust manual calculation of Head 0 scores:
        q0_rust = p_q[:64]
        # RoPE on q0_rust at pos 4
        q0_rot_rust = q0_rust.copy()
        for i in range(32):
            freq = 1.0 / (gaje_llm.rope_base ** ((2.0 * i) / 64.0))
            theta = 4 * freq
            s_v, c_v = np.sin(theta), np.cos(theta)
            v0 = q0_rot_rust[i]
            v1 = q0_rot_rust[i + 32]
            q0_rot_rust[i] = v0 * c_v - v1 * s_v
            q0_rot_rust[i + 32] = v0 * s_v + v1 * c_v

        # K cache for head 0 across tokens 0..4
        rust_k_list = []
        for p_i in range(len(input_ids)):
            p_e = emb_all[0, p_i, :].numpy()
            p_n = (p_e / np.sqrt(np.mean(p_e**2) + eps)) * attn_norm_w
            p_k_vec = np.array(
                mock_blk0.attn_layer.k_gen.linear.forward(p_n.tolist(), False)
            )[:64]
            # RoPE on k0_rust at pos p_i
            k0_rot = p_k_vec.copy()
            for i in range(32):
                freq = 1.0 / (gaje_llm.rope_base ** ((2.0 * i) / 64.0))
                theta = p_i * freq
                s_v, c_v = np.sin(theta), np.cos(theta)
                v0 = k0_rot[i]
                v1 = k0_rot[i + 32]
                k0_rot[i] = v0 * c_v - v1 * s_v
                k0_rot[i + 32] = v0 * s_v + v1 * c_v
            rust_k_list.append(k0_rot)

        rust_k_mat = np.array(rust_k_list)
        raw_scores_rust = (q0_rot_rust @ rust_k_mat.T) * scale
        scores_rust_h0 = np.exp(raw_scores_rust - np.max(raw_scores_rust))
        scores_rust_h0 /= np.sum(scores_rust_h0)
        q0_hf_prerope = q_t_4d[0, 0, -1, :].numpy()
        q0_rust_prerope = p_q[:64]
        cos_pre = np.dot(q0_hf_prerope, q0_rust_prerope) / (
            np.linalg.norm(q0_hf_prerope) * np.linalg.norm(q0_rust_prerope) + 1e-9
        )
        print(f"Q Head 0 PRE-RoPE CosSim: {cos_pre:.6f}")

        q0_hf_rot = q_rot_t[0, 0, -1, :].numpy()
        k0_hf_rot = k_rot_t[0, 0, -1, :].numpy()

        cos_q_rope = np.dot(q0_hf_rot, q0_rot_rust) / (
            np.linalg.norm(q0_hf_rot) * np.linalg.norm(q0_rot_rust) + 1e-9
        )
        cos_k_rope = np.dot(k0_hf_rot, rust_k_list[-1]) / (
            np.linalg.norm(k0_hf_rot) * np.linalg.norm(rust_k_list[-1]) + 1e-9
        )
        print(f"Q Head 0 Post-RoPE CosSim: {cos_q_rope:.6f}")
        print(f"K Head 0 Post-RoPE CosSim: {cos_k_rope:.6f}")

    # Check V cache alignment
    v_hf_list = []
    v_rust_list = []
    for p_i in range(len(input_ids)):
        p_e = emb_all[0, p_i, :].numpy()
        p_n = (p_e / np.sqrt(np.mean(p_e**2) + eps)) * attn_norm_w
        v_h = (
            blk0_hf.self_attn.v_proj(blk0_hf.input_layernorm(emb_all))[0, p_i, :64]
            .detach()
            .numpy()
        )
        v_r = np.array(mock_blk0.attn_layer.v_gen.linear.forward(p_n.tolist(), False))[
            :64
        ]
        v_hf_list.append(v_h)
        v_rust_list.append(v_r)
        c_v_tok = np.dot(v_h, v_r) / (np.linalg.norm(v_h) * np.linalg.norm(v_r) + 1e-9)
        print(f"V Token {p_i} Head 0 CosSim: {c_v_tok:.6f}")

    attn_h0_hf = attn_out_hf[0, -1, :64].numpy()
    attn_h0_rust = attn_out_rust[:64]
    cos_h0 = np.dot(attn_h0_hf, attn_h0_rust) / (
        np.linalg.norm(attn_h0_hf) * np.linalg.norm(attn_h0_rust) + 1e-9
    )
    print(
        f"Attention Core Head 0 CosSim: {cos_h0:.6f} | HF Head 0 Norm: {np.linalg.norm(attn_h0_hf):.4f} | Rust Head 0 Norm: {np.linalg.norm(attn_h0_rust):.4f}"
    )

    # 5. Test Attention Output Projection (w_o)
    attn_proj_rust = np.array(
        mock_blk0.attn_layer.w_o.linear.forward(attn_out_rust.tolist(), False)
    )
    cos_wo = np.dot(attn_proj_hf, attn_proj_rust) / (
        np.linalg.norm(attn_proj_hf) * np.linalg.norm(attn_proj_rust) + 1e-9
    )
    print(
        f"5. w_o Projection:   CosSim: {cos_wo:.6f} | HF Norm: {np.linalg.norm(attn_proj_hf):.4f} | Rust Norm: {np.linalg.norm(attn_proj_rust):.4f}"
    )

    # 6. Residual 1 (x + proj_attn)
    x_post_attn_rust = x_rust + attn_proj_rust
    cos_res1 = np.dot(x_post_attn_hf, x_post_attn_rust) / (
        np.linalg.norm(x_post_attn_hf) * np.linalg.norm(x_post_attn_rust) + 1e-9
    )
    print(
        f"6. Residual 1:       CosSim: {cos_res1:.6f} | HF Norm: {np.linalg.norm(x_post_attn_hf):.4f} | Rust Norm: {np.linalg.norm(x_post_attn_rust):.4f}"
    )

    # 8. FFN Norm
    ffn_norm_w = blk0_hf.post_attention_layernorm.weight.detach().numpy()
    rms_val2 = np.sqrt(np.mean(x_post_attn_rust**2) + eps)
    x_ffn_norm_rust = (x_post_attn_rust / rms_val2) * ffn_norm_w
    cos_ffn_norm = np.dot(x_ffn_norm_hf, x_ffn_norm_rust) / (
        np.linalg.norm(x_ffn_norm_hf) * np.linalg.norm(x_ffn_norm_rust) + 1e-9
    )
    print(
        f"8. FFN RMSNorm:     CosSim: {cos_ffn_norm:.6f} | HF Norm: {np.linalg.norm(x_ffn_norm_hf):.4f} | Rust Norm: {np.linalg.norm(x_ffn_norm_rust):.4f}"
    )

    # 9. Gate & Up
    gate_rust = np.array(
        mock_blk0.gate_gen.linear.forward(x_ffn_norm_rust.tolist(), False)
    )
    up_rust = np.array(mock_blk0.up_gen.linear.forward(x_ffn_norm_rust.tolist(), False))
    cos_gate = np.dot(gate_hf, gate_rust) / (
        np.linalg.norm(gate_hf) * np.linalg.norm(gate_rust) + 1e-9
    )
    cos_up = np.dot(up_hf, up_rust) / (
        np.linalg.norm(up_hf) * np.linalg.norm(up_rust) + 1e-9
    )
    print(
        f"9. Gate Projection:  CosSim: {cos_gate:.6f} | HF Norm: {np.linalg.norm(gate_hf):.4f} | Rust Norm: {np.linalg.norm(gate_rust):.4f}"
    )
    print(
        f"10. Up Projection:   CosSim: {cos_up:.6f} | HF Norm: {np.linalg.norm(up_hf):.4f} | Rust Norm: {np.linalg.norm(up_rust):.4f}"
    )
    print(f"5. Projected Attn:   HF Norm: {np.linalg.norm(attn_proj_hf):.4f}")


if __name__ == "__main__":
    trace_block0_subops()
