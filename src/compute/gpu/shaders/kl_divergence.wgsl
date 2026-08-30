// kl_divergence.wgsl — Divergencia KL + Cross Entropy en GPU con vocabulario dinámico
struct KlDivergenceUniforms {
    alpha: f32,
    temperature: f32,
    batch_size: u32,
    vocab_size: u32,
};
@group(0) @binding(0) var<uniform> params: KlDivergenceUniforms;
@group(0) @binding(1) var<storage, read> teacher_probs: array<f32>;
@group(0) @binding(2) var<storage, read> student_probs: array<f32>;
@group(0) @binding(3) var<storage, read_write> loss_output: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.batch_size) { return; }

    let base = idx * params.vocab_size;
    var kl: f32 = 0.0;
    var ce: f32 = 0.0;
    let v_len = params.vocab_size;

    for (var j: u32 = 0u; j < v_len; j = j + 1u) {
        let p_t = teacher_probs[base + j];
        let p_s = student_probs[base + j];
        if (p_t > 1e-6 && p_s > 1e-6) {
            kl = kl + p_t * (log(p_t) - log(p_s));
            ce = ce - p_t * log(p_s);
        }
    }
    let total = params.alpha * kl + (1.0 - params.alpha) * ce;
    loss_output[idx] = total / f32(params.batch_size);
}
