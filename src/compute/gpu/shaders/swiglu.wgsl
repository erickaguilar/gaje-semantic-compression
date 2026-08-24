// =============================================================================
// swiglu.wgsl — Activación y Fusión SiLU (SwiGLU) en GPU
// =============================================================================

struct SwigluParams {
    len: u32,
    h_scale: f32,
};

@group(0) @binding(0) var<uniform> params: SwigluParams;
@group(0) @binding(1) var<storage, read> gate: array<f32>;
@group(0) @binding(2) var<storage, read> up: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_vec: array<f32>;

fn silu(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.len) {
        return;
    }

    let g = gate[idx];
    let u = up[idx];
    let activated = silu(g) * u * params.h_scale;
    output_vec[idx] = activated;
}
