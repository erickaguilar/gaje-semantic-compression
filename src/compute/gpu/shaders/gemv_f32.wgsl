// =============================================================================
// gemv_f32.wgsl — Multiplicación Matriz-Vector FP32 Paralela en GPU
// =============================================================================

struct Uniforms {
    rows: u32, // M: out_features
    cols: u32, // K: in_features
};

@group(0) @binding(0) var<uniform> params: Uniforms;
@group(0) @binding(1) var<storage, read> weights: array<f32>; // M x K (Row-Major)
@group(0) @binding(2) var<storage, read> input_vec: array<f32>; // K
@group(0) @binding(3) var<storage, read_write> output_vec: array<f32>; // M

const WORKGROUP_SIZE: u32 = 64u;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= params.rows) {
        return;
    }

    var sum: f32 = 0.0;
    let row_offset = row * params.cols;

    // Vectorized 4-element loop where possible
    let cols_vec4 = params.cols / 4u;
    for (var c: u32 = 0u; c < cols_vec4; c = c + 1u) {
        let idx = row_offset + c * 4u;
        let v_idx = c * 4u;
        sum = sum + weights[idx + 0u] * input_vec[v_idx + 0u];
        sum = sum + weights[idx + 1u] * input_vec[v_idx + 1u];
        sum = sum + weights[idx + 2u] * input_vec[v_idx + 2u];
        sum = sum + weights[idx + 3u] * input_vec[v_idx + 3u];
    }

    // Remainder
    for (var c: u32 = cols_vec4 * 4u; c < params.cols; c = c + 1u) {
        sum = sum + weights[row_offset + c] * input_vec[c];
    }

    output_vec[row] = sum;
}
