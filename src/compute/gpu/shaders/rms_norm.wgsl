// =============================================================================
// rms_norm.wgsl — Normalización RMS Paralela en GPU
// =============================================================================

struct RmsParams {
    len: u32,
    eps: f32,
};

@group(0) @binding(0) var<uniform> params: RmsParams;
@group(0) @binding(1) var<storage, read> input_vec: array<f32>;
@group(0) @binding(2) var<storage, read> weight_vec: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_vec: array<f32>;

var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let tid = local_id.x;
    let gid = global_id.x;

    // Step 1: Thread-local sum of squares
    var local_sq: f32 = 0.0;
    var i: u32 = gid;
    while (i < params.len) {
        let val = input_vec[i];
        local_sq = local_sq + val * val;
        i = i + (256u * 64u); // Stride if large
    }
    shared_sum[tid] = local_sq;
    workgroupBarrier();

    // Step 2: Workgroup reduction
    for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + s];
        }
        workgroupBarrier();
    }

    // Step 3: Compute scale factor
    let mean_sq = shared_sum[0] / f32(params.len);
    let scale = 1.0 / sqrt(mean_sq + params.eps);

    // Step 4: Scale and multiply by weight
    var j: u32 = gid;
    while (j < params.len) {
        output_vec[j] = input_vec[j] * scale * weight_vec[j];
        j = j + (256u * 64u);
    }
}
