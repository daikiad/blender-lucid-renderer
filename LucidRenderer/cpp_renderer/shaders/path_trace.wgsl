// path_trace.wgsl - Phase 2a GPU path tracer
//
// Mirrors the CPU `traceSimple()` integrator: BSDF sampling only, no NEE/MIS.
// Lambertian diffuse + emissive triangles. Russian roulette from depth 3.
// Per-pixel inner loop sums radiance over `params.samples` paths
// (un-normalized — Python's _accumulate_samples handles time-averaging).

struct PathParams {
    pos:           vec3<f32>, _p0: f32,
    fwd:           vec3<f32>, _p1: f32,
    right:         vec3<f32>, _p2: f32,
    up:            vec3<f32>, fov_scale: f32,
    aspect:        f32,
    tile_x:        u32,
    tile_y:        u32,
    tile_w:        u32,
    tile_h:        u32,
    full_w:        u32,
    full_h:        u32,
    _p3:           u32,
    samples:       u32,
    sample_offset: u32,
    max_bounces:   u32,
    frame_seed:    u32,
    env_color:     vec3<f32>,
    env_strength:  f32,
    point_light_count: u32,
    _p4: u32, _p5: u32, _p6: u32,
};

struct PointLight {
    pos:       vec3<f32>, _pad0: f32,
    color:     vec3<f32>, intensity: f32,
};

struct Triangle {
    v0:           vec3<f32>, _p0: f32,
    v1:           vec3<f32>, _p1: f32,
    v2:           vec3<f32>, _p2: f32,
    n0:           vec3<f32>, _p3: f32,
    n1:           vec3<f32>, _p4: f32,
    n2:           vec3<f32>, smooth_flag: f32,
    albedo:       vec3<f32>, _p5: f32,
    emission:     vec3<f32>, _p6: f32,
};

@group(0) @binding(0) var<uniform>             params       : PathParams;
@group(0) @binding(1) var<storage, read>       tris         : array<Triangle>;
@group(0) @binding(2) var<storage, read_write> out_pixels   : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       point_lights : array<PointLight>;

// ----------------------------------------------------------------------------
// PCG random helpers
// ----------------------------------------------------------------------------

fn pcg_seed(x: u32, y: u32, s: u32, f: u32) -> u32 {
    var h: u32 =
        x * 0x9e3779b1u +
        y * 0x85ebca6bu +
        s * 0xc2b2ae35u +
        f * 0x27d4eb2fu;
    h = h ^ (h >> 16u); h = h * 0x85ebca6bu;
    h = h ^ (h >> 13u); h = h * 0xc2b2ae35u;
    h = h ^ (h >> 16u);
    return h | 1u;
}

fn pcg_next(state: ptr<function, u32>) -> u32 {
    let old = *state;
    *state = old * 747796405u + 2891336453u;
    let word = ((old >> ((old >> 28u) + 4u)) ^ old) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_f32(state: ptr<function, u32>) -> f32 {
    // Strip the top bit so 0x7FFFFFFF -> ~0.9999... ; never quite 1.0.
    let u = pcg_next(state) >> 9u;             // [0, 2^23 - 1]
    return f32(u) * (1.0 / 8388608.0);          // [0, 1)
}

// ----------------------------------------------------------------------------
// Orthonormal basis (Duff et al. 2017)
// ----------------------------------------------------------------------------

fn build_ortho_basis_t(n: vec3<f32>) -> vec3<f32> {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    return vec3<f32>(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
}

fn build_ortho_basis_b(n: vec3<f32>) -> vec3<f32> {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    return vec3<f32>(b, s + n.y * n.y * a, -n.y);
}

fn cosine_hemisphere(n: vec3<f32>, u1: f32, u2: f32) -> vec3<f32> {
    let phi = 6.2831853 * u1;                       // 2π · u1
    let r   = sqrt(u2);
    let local = vec3<f32>(
        r * cos(phi),
        r * sin(phi),
        sqrt(max(0.0, 1.0 - u2))
    );
    let t  = build_ortho_basis_t(n);
    let bt = build_ortho_basis_b(n);
    let dir = local.x * t + local.y * bt + local.z * n;
    return normalize(dir);
}

// ----------------------------------------------------------------------------
// Möller-Trumbore (same epsilons as debug_normal.wgsl)
// ----------------------------------------------------------------------------

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

// ----------------------------------------------------------------------------
// Main path tracer
// ----------------------------------------------------------------------------

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.tile_w || gid.y >= params.tile_h) { return; }
    if (params.tile_w == 0u || params.tile_h == 0u)      { return; }

    // Camera primary ray — same NDC math as debug shader.
    let gx = f32(params.tile_x + gid.x) + 0.5;
    let gy = f32(params.tile_y + gid.y) + 0.5;
    let ndc_x = (2.0 * gx / f32(params.full_w) - 1.0) * params.aspect;
    let ndc_y =  1.0 - 2.0 * gy / f32(params.full_h);
    let primary_dir = normalize(
        params.fwd
        + params.right * (ndc_x * params.fov_scale)
        + params.up    * (ndc_y * params.fov_scale)
    );

    var radiance_sum = vec3<f32>(0.0, 0.0, 0.0);
    let n_tris = arrayLength(&tris);

    for (var s: u32 = 0u; s < params.samples; s = s + 1u) {
        var rng = pcg_seed(
            params.tile_x + gid.x,
            params.tile_y + gid.y,
            params.sample_offset + s,
            params.frame_seed
        );
        var orig = params.pos;
        var dir  = primary_dir;
        var throughput = vec3<f32>(1.0, 1.0, 1.0);
        var radiance   = vec3<f32>(0.0, 0.0, 0.0);

        for (var b: u32 = 0u; b < params.max_bounces; b = b + 1u) {
            // ---- Brute-force closest-hit search ----
            var best_t = 1e30;
            var hit = false;
            var best_n        = vec3<f32>(0.0, 0.0, 1.0);
            var best_albedo   = vec3<f32>(0.0, 0.0, 0.0);
            var best_emission = vec3<f32>(0.0, 0.0, 0.0);

            for (var i: u32 = 0u; i < n_tris; i = i + 1u) {
                let tri = tris[i];
                var t: f32; var u: f32; var v: f32;
                if (intersect_tri(orig, dir, tri.v0, tri.v1, tri.v2,
                                   &t, &u, &v) && t < best_t) {
                    best_t = t;
                    hit = true;
                    let w = 1.0 - u - v;
                    if (tri.smooth_flag > 0.5) {
                        best_n = normalize(w * tri.n0 + u * tri.n1 + v * tri.n2);
                    } else {
                        best_n = normalize(cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
                    }
                    best_albedo   = tri.albedo;
                    best_emission = tri.emission;
                }
            }

            // ---- Termination cases ----
            if (!hit) {
                radiance = radiance + throughput * (params.env_color * params.env_strength);
                break;
            }
            if (any(best_emission > vec3<f32>(1e-6, 1e-6, 1e-6))) {
                radiance = radiance + throughput * best_emission;
                break;
            }

            // ---- NEE: direct contribution from every point light ----
            // Point lights have area=0, so they can never be hit by BSDF sampling.
            // No double-counting with the indirect bounce below.
            let hit_point = orig + dir * best_t;
            let inv_pi = 0.31830988618;  // 1/π for the Lambertian f = albedo/π
            for (var li: u32 = 0u; li < params.point_light_count; li = li + 1u) {
                let light = point_lights[li];
                let to_light = light.pos - hit_point;
                let d2 = dot(to_light, to_light);
                if (d2 < 1e-8) { continue; }
                let d = sqrt(d2);
                let light_dir = to_light / d;
                let cos_theta = dot(best_n, light_dir);
                if (cos_theta <= 0.0) { continue; }

                // Shadow ray — brute-force occlusion test against every triangle.
                let shadow_orig = hit_point + best_n * 1e-3;
                let max_t = d - 2e-3;
                var occluded = false;
                for (var k: u32 = 0u; k < n_tris; k = k + 1u) {
                    let st  = tris[k];
                    var t: f32; var u: f32; var v: f32;
                    if (intersect_tri(shadow_orig, light_dir, st.v0, st.v1, st.v2,
                                       &t, &u, &v) && t < max_t) {
                        occluded = true;
                        break;
                    }
                }
                if (occluded) { continue; }

                // Irradiance from an isotropic point light: L = color · energy / d²
                // Lambertian BRDF f = albedo / π. Delta-direction light so pdf=1.
                let L = light.color * light.intensity / d2;
                radiance = radiance + throughput * best_albedo * inv_pi * L * cos_theta;
            }

            // ---- Lambertian: throughput accumulates albedo (for indirect bounce) ----
            throughput = throughput * best_albedo;

            // ---- Russian roulette from depth 3 ----
            if (b >= 3u) {
                let p = min(max(throughput.r, max(throughput.g, throughput.b)), 0.95);
                if (rand_f32(&rng) > p) { break; }
                throughput = throughput / p;
            }

            // ---- Next ray: cosine-weighted hemisphere ----
            let u1 = rand_f32(&rng);
            let u2 = rand_f32(&rng);
            dir  = cosine_hemisphere(best_n, u1, u2);
            // Offset along the hit normal to dodge self-intersection.
            // 1e-3 (vs CPU's 1e-4) — FP32 grazing-angle slack.
            orig = (orig + dir * best_t) + best_n * 1e-3;
        }

        radiance_sum = radiance_sum + radiance;
    }

    // Y-flip on store — same convention as debug_normal.wgsl.
    let out_y = params.tile_h - 1u - gid.y;
    let idx   = out_y * params.tile_w + gid.x;
    out_pixels[idx] = vec4<f32>(radiance_sum, 1.0);
}
