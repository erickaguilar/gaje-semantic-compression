// =============================================================================
// ste_q2_backward.wgsl — Straight-Through Estimator (STE) Cuaternario en GPU
// =============================================================================
// Actualización masiva y paralela de tensores Q2_0 (2 bits por peso + scale/min)
// Cada bloque Q2_0 contiene 32 pesos empaquetados en 12 bytes (3 words u32).
// Word 0: scale (f16) | min (f16) empaquetados en 32 bits
// Word 1: qs[0..3] (4 bytes, 16 dibits)
// Word 2: qs[4..7] (4 bytes, 16 dibits)
// =============================================================================

struct SteUniforms {
    rows: u32,
    cols: u32,
    lr: f32,
    n_blocks_per_row: u32,
};

struct Q2BlockGpu {
    scale_min: u32,
    qs_low: u32,
    qs_high: u32,
};

@group(0) @binding(0) var<uniform> params: SteUniforms;
@group(0) @binding(1) var<storage, read> grad_output: array<f32>;
@group(0) @binding(2) var<storage, read> input_activations: array<f32>;
@group(0) @binding(3) var<storage, read_write> q2_blocks: array<Q2BlockGpu>;

fn unpack_f16_low(packed: u32) -> f32 {
    let raw = packed & 0xFFFFu;
    return unpack2x16float(raw).x;
}
fn unpack_f16_high(packed: u32) -> f32 {
    let raw = (packed >> 16u) & 0xFFFFu;
    return unpack2x16float(raw).x;
}
fn pack_f16_pair(s: f32, m: f32) -> u32 {
    return pack2x16float(vec2<f32>(s, m));
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let total_blocks = params.rows * params.n_blocks_per_row;
    let block_idx = global_id.x;
    if (block_idx >= total_blocks) { return; }
    let row = block_idx / params.n_blocks_per_row;
    let b = block_idx % params.n_blocks_per_row;
    let g = grad_output[row];
    if (abs(g) < 1e-6) { return; }
    var blk = q2_blocks[block_idx];
    let scale = unpack_f16_low(blk.scale_min);
    let min_val = unpack_f16_high(blk.scale_min);
    let k_base = b * 32u;
    var g_scale: f32 = 0.0;
    var g_min: f32 = 0.0;
    var q_vals: array<u32, 32>;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        let shift = k * 2u;
        let q = (blk.qs_low >> shift) & 0x3u;
        q_vals[k] = q;
        let x_idx = k_base + k;
        if (x_idx < params.cols) {
            let x = input_activations[x_idx];
            let gw = g * x;
            g_scale = g_scale + gw * f32(q);
            g_min = g_min + gw;
        }
    }
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        let shift = k * 2u;
        let q = (blk.qs_high >> shift) & 0x3u;
        q_vals[k + 16u] = q;
        let x_idx = k_base + k + 16u;
        if (x_idx < params.cols) {
            let x = input_activations[x_idx];
            let gw = g * x;
            g_scale = g_scale + gw * f32(q);
            g_min = g_min + gw;
        }
    }
    let new_scale = clamp(scale - params.lr * g_scale, 1e-4, 10.0);
    let new_min = clamp(min_val - params.lr * g_min, -20.0, 20.0);
    var new_qs_low: u32 = 0u;
    var new_qs_high: u32 = 0u;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        let x_idx = k_base + k;
        var q_new = q_vals[k];
        if (x_idx < params.cols) {
            let x = input_activations[x_idx];
            let continuous_w = f32(q_vals[k]) * scale + min_val - params.lr * g * x;
            let unrounded = (continuous_w - new_min) / max(new_scale, 1e-4);
            q_new = u32(clamp(round(unrounded), 0.0, 3.0));
        }
        new_qs_low = new_qs_low | (q_new << (k * 2u));
    }
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        let x_idx = k_base + k + 16u;
        var q_new = q_vals[k + 16u];
        if (x_idx < params.cols) {
            let x = input_activations[x_idx];
            let continuous_w = f32(q_vals[k + 16u]) * scale + min_val - params.lr * g * x;
            let unrounded = (continuous_w - new_min) / max(new_scale, 1e-4);
            q_new = u32(clamp(round(unrounded), 0.0, 3.0));
        }
        new_qs_high = new_qs_high | (q_new << (k * 2u));
    }
    blk.scale_min = pack_f16_pair(new_scale, new_min);
    blk.qs_low = new_qs_low;
    blk.qs_high = new_qs_high;
    q2_blocks[block_idx] = blk;
}
