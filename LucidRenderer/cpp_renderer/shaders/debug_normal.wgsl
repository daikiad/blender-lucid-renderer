// debug_normal.wgsl - Phase 1b GPU debug renderer (normal visualization)
//
// One thread per pixel; brute-force loops over every triangle in the scene.
// For each pixel it builds a camera ray, finds the closest hit via
// Möller-Trumbore, and writes the surface normal mapped to [0, 1] as RGB.
// Misses get a flat 0.5 gray to match the CPU `traceNormal` path's
// `n * 0.5 + 0.5` -> `(0,0,0) * 0.5 + 0.5 == (0.5, 0.5, 0.5)` convention.
//
// The Y-flip (so the image is Blender-convention orientation) is done in
// the store, mirroring how `PyRenderer::render_debug` writes pixels.

struct Camera {
    pos:        vec3<f32>, _pad0: f32,
    fwd:        vec3<f32>, _pad1: f32,
    right:      vec3<f32>, _pad2: f32,
    up:         vec3<f32>, fov_scale: f32,
    aspect:     f32,
    tile_x:     u32,
    tile_y:     u32,
    tile_w:     u32,
    tile_h:     u32,
    full_w:     u32,
    full_h:     u32,
    _pad3:      u32,
};

struct Triangle {
    v0: vec3<f32>, _pad0: f32,
    v1: vec3<f32>, _pad1: f32,
    v2: vec3<f32>, _pad2: f32,
    n0: vec3<f32>, _pad3: f32,
    n1: vec3<f32>, _pad4: f32,
    n2: vec3<f32>, smooth_flag: f32,
};

@group(0) @binding(0) var<uniform>             cam  : Camera;
@group(0) @binding(1) var<storage, read>       tris : array<Triangle>;
@group(0) @binding(2) var<storage, read_write> out_pixels : array<vec4<f32>>;

// Möller-Trumbore. Returns true when the ray hits the triangle in front of
// the origin (t > 0) with valid barycentrics. Writes t/u/v on hit.
fn intersect_tri(orig: vec3<f32>, dir: vec3<f32>,
                 v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>,
                 out_t: ptr<function, f32>,
                 out_u: ptr<function, f32>,
                 out_v: ptr<function, f32>) -> bool {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let h  = cross(dir, e2);
    let a  = dot(e1, h);
    if (abs(a) < 1e-8) { return false; }
    let f = 1.0 / a;
    let s = orig - v0;
    let u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) { return false; }
    let q = cross(s, e1);
    let v = f * dot(dir, q);
    if (v < 0.0 || u + v > 1.0) { return false; }
    let t = f * dot(e2, q);
    if (t <= 1e-4) { return false; }
    *out_t = t;
    *out_u = u;
    *out_v = v;
    return true;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= cam.tile_w || gid.y >= cam.tile_h) { return; }

    // Pixel center in global image space
    let gx = f32(cam.tile_x + gid.x) + 0.5;
    let gy = f32(cam.tile_y + gid.y) + 0.5;
    let ndc_x = (2.0 * gx / f32(cam.full_w) - 1.0) * cam.aspect;
    let ndc_y =  1.0 - 2.0 * gy / f32(cam.full_h);

    let dir = normalize(cam.fwd
                        + cam.right * (ndc_x * cam.fov_scale)
                        + cam.up    * (ndc_y * cam.fov_scale));

    var best_t = 1e30;
    var best_n = vec3<f32>(0.0, 0.0, 0.0);
    var hit    = false;

    let n_tris = arrayLength(&tris);
    for (var i: u32 = 0u; i < n_tris; i = i + 1u) {
        let tri = tris[i];
        var t: f32; var u: f32; var v: f32;
        if (intersect_tri(cam.pos, dir, tri.v0, tri.v1, tri.v2,
                          &t, &u, &v) && t < best_t) {
            best_t = t;
            hit    = true;
            let w  = 1.0 - u - v;
            if (tri.smooth_flag > 0.5) {
                best_n = normalize(w * tri.n0 + u * tri.n1 + v * tri.n2);
            } else {
                best_n = normalize(cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
            }
        }
    }

    var color: vec3<f32>;
    if (hit) {
        color = best_n * 0.5 + vec3<f32>(0.5, 0.5, 0.5);
    } else {
        color = vec3<f32>(0.5, 0.5, 0.5);
    }

    // Y-flip on store: row 0 of the GPU buffer corresponds to the
    // bottom-most row of the input tile (Blender / pybind convention).
    let out_y = cam.tile_h - 1u - gid.y;
    let idx   = out_y * cam.tile_w + gid.x;
    out_pixels[idx] = vec4<f32>(color, 1.0);
}
