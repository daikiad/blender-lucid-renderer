// double_test.wgsl - Phase 1a end-to-end smoke shader
//
// Reads each element of `src`, doubles it, writes to `dst`.
// Used to verify the WGSL -> pipeline -> dispatch -> readback path is wired up.

@group(0) @binding(0) var<storage, read>       src : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < arrayLength(&src)) {
        dst[i] = src[i] * 2.0;
    }
}
