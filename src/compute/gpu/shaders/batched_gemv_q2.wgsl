// batched_gemv_q2.wgsl — Batch GEMV para Q2_0 (2 bits) - versión mínima válida
struct BatchedGemvQ2Uniforms {
    rows: u32,
    cols: u32,
    n_blocks_per_row: u32,
    batch_size: u32,
    scale: f32,
    min_val: f32,
    lr: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};
struct Q2BlockGpu {
    scale_min: u32,
    qs_low: u32,
    qs_high: u32,
};
@group(0) @binding(0) var<uniform> params: BatchedGemvQ2Uniforms;
@group(0) @binding(1) var<storage, read> input_activations: array<f32>;
@group(0) @binding(2) var<storage, read> q2_blocks: array<Q2BlockGpu>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

fn unpack_f16_low(packed: u32) -> f32 {
    let raw = packed & 0xFFFFu;
    return unpack2x16float(raw).x;
}
fn unpack_f16_high(packed: u32) -> f32 {
    let raw = (packed >> 16u) & 0xFFFFu;
    return unpack2x16float(raw).x;
}
fn decode_q2_weight(q: u32, scale: f32, min_val: f32) -> f32 {
    if (q == 0u) { return min_val; }
    if (q == 1u) { return min_val + scale * 0.3333; }
    if (q == 2u) { return min_val + scale * 0.6666; }
    return min_val + scale;
}
@compute @workgroup_size(32, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let bx = global_id.x;
    let by = global_id.y;
    let total_blocks_per_row = params.n_blocks_per_row;
    let block_row = bx / total_blocks_per_row;
    let block_col = bx % total_blocks_per_row;
    if (block_row >= params.rows || block_col >= params.n_blocks_per_row || by >= params.batch_size) { return; }
    let token_offset = by * params.cols;
    let k_base = block_col * 32u;
    let block = q2_blocks[bx];
    let block_scale = unpack_f16_low(block.scale_min);
    let block_min = unpack_f16_high(block.scale_min);
    var accum: f32 = 0.0;
    for (var k: u32 = 0u; k < 32u; k = k + 1u) {
        let w_idx = token_offset + k_base + k;
        if (w_idx >= params.cols) { continue; }
        var q_val: u32 = 0u;
        if (k < 16u) {
            q_val = (block.qs_low >> (k * 2u)) & 0x3u;
        } else {
            q_val = (block.qs_high >> ((k - 16u) * 2u)) & 0x3u;
        }
        let w = decode_q2_weight(q_val, block_scale, block_min);
        let x = input_activations[w_idx];
        accum = fma(x, w, accum);
    }
    let output_offset = (by * params.rows + block_row) * params.cols + k_base;
    if (output_offset < arrayLength(&output)) {
        output[output_offset] = accum;
    }
}
