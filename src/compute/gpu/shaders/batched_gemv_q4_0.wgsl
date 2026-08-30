// batched_gemv_q4_0.wgsl — Batch GEMV Q4_0 16 centroides, workgroup(32,8,1)
struct BatchedGemvQ4Uniforms {
    rows: u32, cols: u32, n_blocks_per_row: u32, batch_size: u32,
    scale: f32, min_val: f32, lr: f32, _pad0: u32, _pad1: u32, _pad2: u32,
};
struct Q4BlockGpu { scale_min: u32, qs: array<u32,4>, } // 16 bytes = 4 u32
@group(0) @binding(0) var<uniform> params: BatchedGemvQ4Uniforms;
@group(0) @binding(1) var<storage, read> input_activations: array<f32>;
@group(0) @binding(2) var<storage, read> q4_blocks: array<Q4BlockGpu>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
fn unpack_f16_low(p: u32) -> f32 { return unpack2x16float(p & 0xFFFFu).x; }
fn unpack_f16_high(p: u32) -> f32 { return unpack2x16float((p>>16u) & 0xFFFFu).x; }
@compute @workgroup_size(32,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bx = gid.x; let by = gid.y;
    let tpr = params.n_blocks_per_row;
    let br = bx / tpr; let bc = bx % tpr;
    if (br >= params.rows || bc >= tpr || by >= params.batch_size) { return; }
    let tok_off = by * params.cols; let k_base = bc * 32u;
    let blk = q4_blocks[bx];
    let sc = unpack_f16_low(blk.scale_min); let mn = unpack_f16_high(blk.scale_min);
    var acc: f32 = 0.0;
    for (var k: u32 = 0u; k < 32u; k = k + 1u) {
        let w_idx = tok_off + k_base + k; if (w_idx >= params.cols) { continue; }
        let word = blk.qs[k/8u]; let shift = (k%8u)*4u; let q = (word >> shift) & 0xFu;
        let w = f32(q) * sc + mn;
        acc = fma(input_activations[w_idx], w, acc);
    }
    let out_off = (by * params.rows + br) * params.cols + k_base;
    if (out_off < arrayLength(&output)) { output[out_off] = acc; }
}
