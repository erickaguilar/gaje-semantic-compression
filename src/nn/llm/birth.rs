// =============================================================================
// birth — Protocolo de Nacimiento de Organismos Genómicos en 2-Bits (Q2_0)
// =============================================================================
use std::sync::Arc;
use crate::io::header::Q2_0Block;
use crate::nn::attention::GenomicAttention;
use crate::nn::block::RustGenomicBlock;
use crate::nn::linear::database::WeightDatabase;
use crate::nn::linear::GenomicLinear;
use crate::nn::llm::GenomicLLM;

/// Configuración de nacimiento del organismo genómico
#[derive(Debug, Clone)]
pub struct BornConfig {
    pub name: String,
    pub vocab_size: usize,
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub intermediate_dim: usize,
    pub eps: f32,
    pub k_wta_ratio: f32,
}

impl Default for BornConfig {
    fn default() -> Self {
        Self {
            name: "max".to_string(),
            vocab_size: 4000,
            dim: 256,
            n_layers: 8,
            n_heads: 4,
            intermediate_dim: 768,
            eps: 1e-6,
            k_wta_ratio: 0.15,
        }
    }
}

/// Genera una capa lineal nacida en Q2_0 (2-bit DNA Conformal)
pub fn create_born_q2_0_linear(out_features: usize, in_features: usize) -> GenomicLinear {
    let block_size = 32;
    let n_blocks_per_row = (in_features + block_size - 1) / block_size;
    let total_blocks = out_features * n_blocks_per_row;

    let mut blocks: Vec<Q2_0Block> = Vec::with_capacity(total_blocks);

    // Inicialización de Hadamard/QPSK en 2 bits: 4 fases (A=00, C=01, G=10, T=11)
    let scale_val = 1.0f32 / (in_features as f32).sqrt();
    let min_val = -1.5f32 * scale_val;

    for row in 0..out_features {
        for b in 0..n_blocks_per_row {
            let mut qs = [0u8; 8];
            for k in 0..8 {
                // Patrón ortogonal determinista derivado de índices espaciales
                let idx0 = ((row * 31 + b * 7 + k * 4 + 0) % 4) as u8;
                let idx1 = ((row * 31 + b * 7 + k * 4 + 1) % 4) as u8;
                let idx2 = ((row * 31 + b * 7 + k * 4 + 2) % 4) as u8;
                let idx3 = ((row * 31 + b * 7 + k * 4 + 3) % 4) as u8;
                qs[k] = idx0 | (idx1 << 2) | (idx2 << 4) | (idx3 << 6);
            }
            blocks.push(Q2_0Block {
                scale: half::f16::from_f32(scale_val),
                min: half::f16::from_f32(min_val),
                qs,
            });
        }
    }

    GenomicLinear {
        weight_db: WeightDatabase::GenomicQ2_0(Arc::new(blocks)),
        epi_strands: Arc::new(Vec::new()),
        tri_strands: Arc::new(Vec::new()),
        epi_cols: Arc::new(Vec::new()),
        tri_cols: Arc::new(Vec::new()),
        anchor_indices: Arc::new(Vec::new()),
        anchor_values: Arc::new(Vec::new()),
        anchor_row_ptrs: Arc::new(Vec::new()),
        centroids: Vec::new(),
        epigenetic_centroids: Vec::new(),
        triplet_centroids: Vec::new(),
        out_features,
        in_features,
        block_size,
        rmsnorm_weight: vec![1.0; in_features],
        eps: 1e-6,
        bias: Vec::new(),
        stride: 0,
    }
}

/// Genera una capa lineal nacida en Q4_0 (4-bit) — control para matriz Q4 vs Q2 dim256×8
pub fn create_born_q4_0_linear(out_features: usize, in_features: usize) -> GenomicLinear {
    use crate::io::header::Q4_0Block;
    let block_size = 32;
    let n_blocks_per_row = (in_features + block_size - 1) / block_size;
    let total_blocks = out_features * n_blocks_per_row;
    let mut blocks: Vec<Q4_0Block> = Vec::with_capacity(total_blocks);
    let scale_val = 1.0f32 / (in_features as f32).sqrt();
    let min_val = -1.5f32 * scale_val;
    for row in 0..out_features {
        for b in 0..n_blocks_per_row {
            let mut qs = [0u8; 16];
            for k in 0..16 {
                let lo = ((row * 31 + b * 7 + k * 2 + 0) % 16) as u8;
                let hi = ((row * 31 + b * 7 + k * 2 + 1) % 16) as u8;
                qs[k] = lo | (hi << 4);
            }
            blocks.push(Q4_0Block { scale: half::f16::from_f32(scale_val), min: half::f16::from_f32(min_val), qs });
        }
    }
    GenomicLinear {
        weight_db: WeightDatabase::GenomicQ4_0(Arc::new(blocks)),
        epi_strands: Arc::new(Vec::new()),
        tri_strands: Arc::new(Vec::new()),
        epi_cols: Arc::new(Vec::new()),
        tri_cols: Arc::new(Vec::new()),
        anchor_indices: Arc::new(Vec::new()),
        anchor_values: Arc::new(Vec::new()),
        anchor_row_ptrs: Arc::new(Vec::new()),
        centroids: Vec::new(),
        epigenetic_centroids: Vec::new(),
        triplet_centroids: Vec::new(),
        out_features,
        in_features,
        block_size,
        rmsnorm_weight: vec![1.0; in_features],
        eps: 1e-6,
        bias: Vec::new(),
        stride: 0,
    }
}

/// Genera una capa lineal FP32 para embeddings o lm_head
pub fn create_born_fp32_linear(out_features: usize, in_features: usize) -> GenomicLinear {
    let total_weights = out_features * in_features;
    let mut weights = Vec::with_capacity(total_weights);
    let std_dev = 1.0f32 / (in_features as f32).sqrt();

    for i in 0..out_features {
        for j in 0..in_features {
            let phase = ((i * 37 + j * 19) % 100) as f32 / 100.0 * 2.0 - 1.0;
            weights.push(phase * std_dev);
        }
    }

    GenomicLinear {
        weight_db: WeightDatabase::GenomicF32(Arc::new(weights)),
        epi_strands: Arc::new(Vec::new()),
        tri_strands: Arc::new(Vec::new()),
        epi_cols: Arc::new(Vec::new()),
        tri_cols: Arc::new(Vec::new()),
        anchor_indices: Arc::new(Vec::new()),
        anchor_values: Arc::new(Vec::new()),
        anchor_row_ptrs: Arc::new(Vec::new()),
        centroids: Vec::new(),
        epigenetic_centroids: Vec::new(),
        triplet_centroids: Vec::new(),
        out_features,
        in_features,
        block_size: 32,
        rmsnorm_weight: vec![1.0; in_features],
        eps: 1e-6,
        bias: Vec::new(),
        stride: 0,
    }
}

/// Da a luz a un organismo genómico completo nacido en 2 bits (Q2_0)
pub fn create_born_organism(config: BornConfig) -> GenomicLLM {
    let embeddings = create_born_fp32_linear(config.vocab_size, config.dim);
    let mut blocks = Vec::with_capacity(config.n_layers);

    for idx in 0..config.n_layers {
        let attn = GenomicAttention::new(
            config.n_heads,
            config.n_heads,
            config.dim / config.n_heads,
            vec![1.0; config.dim],
            config.eps,
            10000.0,
            "rope".to_string(),
        );

        let q_gen = create_born_q2_0_linear(config.dim, config.dim);
        let k_gen = create_born_q2_0_linear(config.dim, config.dim);
        let v_gen = create_born_q2_0_linear(config.dim, config.dim);
        let w_o = create_born_q2_0_linear(config.dim, config.dim);

        let gate_gen = create_born_q2_0_linear(config.intermediate_dim, config.dim);
        let up_gen = create_born_q2_0_linear(config.intermediate_dim, config.dim);
        let w_down = create_born_q2_0_linear(config.dim, config.intermediate_dim);

        let ffn_norm = vec![1.0; config.dim];

        let blk = RustGenomicBlock::new(
            idx,
            attn,
            q_gen,
            k_gen,
            v_gen,
            w_o,
            gate_gen,
            up_gen,
            w_down,
            ffn_norm,
            config.eps,
            "silu".to_string(),
            false,
            1.0,
            0.0,
        );
        blocks.push(blk);
    }

    let output_norm = vec![1.0; config.dim];
    let lm_head = create_born_fp32_linear(config.vocab_size, config.dim);

    GenomicLLM {
        embeddings,
        blocks,
        output_norm,
        lm_head,
        eps: config.eps,
        k_wta_ratio: config.k_wta_ratio,
        topology: None,
        quantum_embeddings: None,
        gpu_layers: 0,
        use_gpu: false,
    }
}

/// Da a luz a un organismo Q4_0 control (misma arch dim256×8, 16 centroides)
/// Para matriz Q4_0 vs Q2_0 sin entrenamiento, aísla cuantización vs capacidad.
pub fn create_born_q4_0_organism(config: BornConfig) -> GenomicLLM {
    let embeddings = create_born_fp32_linear(config.vocab_size, config.dim);
    let mut blocks = Vec::with_capacity(config.n_layers);
    for idx in 0..config.n_layers {
        let attn = GenomicAttention::new(config.n_heads, config.n_heads, config.dim / config.n_heads, vec![1.0; config.dim], config.eps, 10000.0, "rope".to_string());
        let q_gen = create_born_q4_0_linear(config.dim, config.dim);
        let k_gen = create_born_q4_0_linear(config.dim, config.dim);
        let v_gen = create_born_q4_0_linear(config.dim, config.dim);
        let w_o = create_born_q4_0_linear(config.dim, config.dim);
        let gate_gen = create_born_q4_0_linear(config.intermediate_dim, config.dim);
        let up_gen = create_born_q4_0_linear(config.intermediate_dim, config.dim);
        let w_down = create_born_q4_0_linear(config.dim, config.intermediate_dim);
        let ffn_norm = vec![1.0; config.dim];
        let blk = RustGenomicBlock::new(idx, attn, q_gen, k_gen, v_gen, w_o, gate_gen, up_gen, w_down, ffn_norm, config.eps, "silu".to_string(), false, 1.0, 0.0);
        blocks.push(blk);
    }
    let output_norm = vec![1.0; config.dim];
    let lm_head = create_born_fp32_linear(config.vocab_size, config.dim);
    GenomicLLM { embeddings, blocks, output_norm, lm_head, eps: config.eps, k_wta_ratio: config.k_wta_ratio, topology: None, quantum_embeddings: None, gpu_layers: 0, use_gpu: false, }
}
