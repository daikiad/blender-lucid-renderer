// path_trace.wgsl - Phase 2a + Phase 1c GPU path tracer
//
// Phase 2a: BSDF sampling only (no NEE/MIS for indirect), Lambertian diffuse
// + emissive triangles + NEE for point lights, Russian roulette from depth 3.
// Phase 1c: BVH-accelerated closest-hit + any-hit (shadow) traversal.
//
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
    bvh_node_count: u32,
    _p5: u32, _p6: u32,
};

// PointLight (3 vec4s / 48 B). `emission` is already premultiplied on the
// C++ side: for `radius > 0` it's `color * energy / (π * area)` (Lambertian
// sphere radiance); for `radius == 0` it's `color * energy / (4π)` (point
// intensity I [W/sr]). The shader branches on `radius` to apply the correct
// measure-conversion to the contribution integral.
struct PointLight {
    pos:       vec3<f32>, radius: f32,
    emission:  vec3<f32>, area:   f32,
    _pad:      vec4<f32>,
};

// Triangle (10 vec4s / 160 B). Material fields cover the full Principled BSDF
// socket set (albedo / metallic / roughness / transmission / ior / emission)
// constant-folded at uv=(0,0). The trailing reserved vec4 leaves room for
// future per-fragment UV or normal-map tangent without re-aligning.
struct Triangle {
    v0:           vec3<f32>, _p0: f32,
    v1:           vec3<f32>, _p1: f32,
    v2:           vec3<f32>, _p2: f32,
    n0:           vec3<f32>, _p3: f32,
    n1:           vec3<f32>, _p4: f32,
    n2:           vec3<f32>, smooth_flag: f32,
    albedo:       vec3<f32>, metallic:    f32,
    emission:     vec3<f32>, roughness:   f32,
    transmission: f32, ior: f32, _p5: f32, _p6: f32,
    _reserved:    vec4<f32>,
};

// BVH node layout (32 bytes / 2 vec4s, std430). Encoding matches GpuBvhNode:
//   leaf:     left = triStart (>= 0), right_or_count = triCount (>= 1)
//   internal: left = left_child,      right_or_count = -(right_child + 1)
// Detection: `right_or_count > 0` → leaf; `right_or_count < 0` → internal.
struct BvhNode {
    bmin: vec3<f32>, left: i32,
    bmax: vec3<f32>, right_or_count: i32,
};

@group(0) @binding(0) var<uniform>             params       : PathParams;
@group(0) @binding(1) var<storage, read>       tris         : array<Triangle>;
@group(0) @binding(2) var<storage, read_write> out_pixels   : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       point_lights : array<PointLight>;
@group(0) @binding(4) var<storage, read>       bvh_nodes    : array<BvhNode>;

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
// Principled BSDF — constants
// ----------------------------------------------------------------------------
// Mirror the CPU side ([bsdf/ggx.hpp:21-24], [bsdf/fresnel.hpp:24]) exactly.

const MIN_ROUGHNESS              : f32 = 0.01;
const GGX_EPSILON                : f32 = 1e-6;
const FRESNEL_EPSILON            : f32 = 1e-6;
const INV_PI                     : f32 = 0.31830988618;
const DELTA_ROUGHNESS_THRESHOLD  : f32 = 0.05;

// ----------------------------------------------------------------------------
// Local <-> world space helpers (TBN where t,b are tangents, n is normal)
// ----------------------------------------------------------------------------

fn world_to_local(w: vec3<f32>, n: vec3<f32>, t: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(w, t), dot(w, b), dot(w, n));
}

fn local_to_world(l: vec3<f32>, n: vec3<f32>, t: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return l.x * t + l.y * b + l.z * n;
}

// ----------------------------------------------------------------------------
// Fresnel ([bsdf/fresnel.hpp])
// ----------------------------------------------------------------------------

fn fresnel_schlick(cos_theta: f32, f0: f32) -> f32 {
    let m = 1.0 - cos_theta;
    let m2 = m * m;
    return f0 + (1.0 - f0) * m2 * m2 * m;
}

fn fresnel_schlick_color(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    let m = 1.0 - cos_theta;
    let m2 = m * m;
    let m5 = m2 * m2 * m;
    return f0 + (vec3<f32>(1.0, 1.0, 1.0) - f0) * m5;
}

// Exact dielectric Fresnel. `eta = n_incident / n_transmitted`.
// Returns 1.0 on total internal reflection.
fn fresnel_dielectric(cos_theta_i_raw: f32, eta: f32) -> f32 {
    let cos_theta_i = clamp(abs(cos_theta_i_raw), 0.0, 1.0);
    let sin2_i = 1.0 - cos_theta_i * cos_theta_i;
    let sin2_t = eta * eta * sin2_i;
    if (sin2_t >= 1.0) { return 1.0; }            // TIR
    let cos_theta_t = sqrt(max(0.0, 1.0 - sin2_t));
    let eta_cos_i = eta * cos_theta_i;
    let eta_cos_t = eta * cos_theta_t;
    let rs = (eta_cos_i - cos_theta_t) / (eta_cos_i + cos_theta_t + FRESNEL_EPSILON);
    let rp = (cos_theta_i - eta_cos_t) / (cos_theta_i + eta_cos_t + FRESNEL_EPSILON);
    return 0.5 * (rs * rs + rp * rp);
}

// ----------------------------------------------------------------------------
// GGX microfacet model ([bsdf/ggx.hpp])
// alpha = roughness²  (Blender / Cycles convention)
// ----------------------------------------------------------------------------

fn ggx_d(n_dot_h: f32, roughness: f32) -> f32 {
    let alpha  = roughness * roughness;
    let a2     = alpha * alpha;
    let n_h2   = n_dot_h * n_dot_h;
    let denom  = n_h2 * (a2 - 1.0) + 1.0;
    return a2 * INV_PI / (denom * denom + GGX_EPSILON);
}

fn ggx_g1(n_dot_v: f32, roughness: f32) -> f32 {
    let alpha = roughness * roughness;
    let a2    = alpha * alpha;
    let s     = sqrt(max(0.0, a2 + (1.0 - a2) * n_dot_v * n_dot_v));
    return 2.0 * n_dot_v / (n_dot_v + s);
}

// Height-correlated Smith G2.
fn ggx_g2(n_dot_l: f32, n_dot_v: f32, roughness: f32) -> f32 {
    if (n_dot_l <= 0.0 || n_dot_v <= 0.0) { return 0.0; }
    let alpha = roughness * roughness;
    let a2    = alpha * alpha;
    let lam_l = sqrt(max(0.0, a2 + (1.0 - a2) * n_dot_l * n_dot_l));
    let lam_v = sqrt(max(0.0, a2 + (1.0 - a2) * n_dot_v * n_dot_v));
    let denom = n_dot_v * lam_l + n_dot_l * lam_v;
    if (denom < GGX_EPSILON) { return 0.0; }
    return 2.0 * n_dot_l * n_dot_v / denom;
}

// GGX VNDF sampling (Heitz 2018). Returns the half-vector in world space.
fn sample_ggx_vndf(wo: vec3<f32>, roughness: f32, u1: f32, u2: f32,
                   n: vec3<f32>, t: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    // 1. wo into local space.
    let wo_local = world_to_local(wo, n, t, b);
    let alpha = roughness * roughness;
    // 2. Stretch onto the unit hemisphere of an isotropic roughness=1 distribution.
    let wo_stretched = normalize(vec3<f32>(wo_local.x * alpha,
                                            wo_local.y * alpha,
                                            wo_local.z));
    // 3. Basis around the stretched vector. Fallback (1,0,0) when wo nearly +Z.
    var t1: vec3<f32>;
    if (wo_stretched.z < 0.9999) {
        let c = cross(vec3<f32>(0.0, 0.0, 1.0), wo_stretched);
        let len = length(c);
        if (len > 1e-6) {
            t1 = c / len;
        } else {
            t1 = vec3<f32>(1.0, 0.0, 0.0);
        }
    } else {
        t1 = vec3<f32>(1.0, 0.0, 0.0);
    }
    let t2 = cross(wo_stretched, t1);
    // 4. Disk sample, projected with reverse Heitz stretching trick.
    let r   = sqrt(u1);
    let phi = 6.2831853 * u2;
    let p1  = r * cos(phi);
    var p2  = r * sin(phi);
    let s   = 0.5 * (1.0 + wo_stretched.z);
    p2 = (1.0 - s) * sqrt(max(0.0, 1.0 - p1 * p1)) + s * p2;
    // 5. Reconstruct half-vector in stretched local space and unstretch.
    let h_stretched = p1 * t1 + p2 * t2
        + sqrt(max(0.0, 1.0 - p1 * p1 - p2 * p2)) * wo_stretched;
    let h_local = normalize(vec3<f32>(h_stretched.x * alpha,
                                       h_stretched.y * alpha,
                                       max(0.0, h_stretched.z)));
    return local_to_world(h_local, n, t, b);
}

fn pdf_ggx_vndf(wo: vec3<f32>, h: vec3<f32>, roughness: f32, n: vec3<f32>) -> f32 {
    let n_dot_h = dot(n, h);
    let v_dot_h = dot(wo, h);
    let n_dot_v = dot(n, wo);
    if (n_dot_h <= 0.0 || v_dot_h <= 0.0 || n_dot_v <= 0.0) { return 0.0; }
    return ggx_d(n_dot_h, roughness) * ggx_g1(n_dot_v, roughness) / (4.0 * n_dot_v);
}

fn pdf_cosine_hemisphere(n_dot_l: f32) -> f32 {
    return max(0.0, n_dot_l) * INV_PI;
}

// ----------------------------------------------------------------------------
// MaterialParams + BSDFSample (mirrors `bsdf::MaterialParams` / `bsdf::BSDFSample`)
// ----------------------------------------------------------------------------
// `kind` discriminant in BSDFSample:
//   0 = EvaluatedBSDF      — throughput weight = f * |cosθ| / pdf
//   1 = PrecomputedWeight  — caller uses `weight` directly (delta-like)
//   2 = invalid            — terminate the path

struct MaterialParams {
    albedo:       vec3<f32>,
    metallic:     f32,
    roughness:    f32,
    transmission: f32,
    ior:          f32,
    emission:     vec3<f32>,
};

struct BSDFSample {
    wi:     vec3<f32>,
    kind:   u32,
    f:      vec3<f32>,
    pdf:    f32,
    weight: vec3<f32>,
};

fn material_from_tri(tri: Triangle) -> MaterialParams {
    var m: MaterialParams;
    m.albedo       = tri.albedo;
    m.metallic     = tri.metallic;
    m.roughness    = max(tri.roughness, MIN_ROUGHNESS);
    m.transmission = tri.transmission;
    m.ior          = tri.ior;
    m.emission     = tri.emission;
    return m;
}

// ----------------------------------------------------------------------------
// BSDF evaluation ([bsdf/bsdf.hpp:126-190])
// ----------------------------------------------------------------------------
// Diffuse Lambert: f_d = albedo / π. Specular: GGX VNDF with Schlick Fresnel
// (F0 = lerp(0.04, albedo, metallic)). Energy-conserving blend uses the
// dielectric Fresnel (F0 = 0.04) to attenuate diffuse: kd = (1 - F) * (1 - metallic).

fn eval_diffuse(mat: MaterialParams, n_dot_l: f32, n_dot_v: f32) -> vec3<f32> {
    if (n_dot_l <= 0.0 || n_dot_v <= 0.0) { return vec3<f32>(0.0, 0.0, 0.0); }
    return mat.albedo * INV_PI;
}

fn eval_specular(mat: MaterialParams,
                 wo: vec3<f32>, wi: vec3<f32>, n: vec3<f32>,
                 n_dot_l: f32, n_dot_v: f32) -> vec3<f32> {
    if (n_dot_l <= 0.0 || n_dot_v <= 0.0) { return vec3<f32>(0.0, 0.0, 0.0); }
    let h_raw = wo + wi;
    let h_len = length(h_raw);
    if (h_len < GGX_EPSILON) { return vec3<f32>(0.0, 0.0, 0.0); }
    let h = h_raw / h_len;
    let n_dot_h = max(dot(n, h), 0.0);
    let v_dot_h = max(dot(wo, h), 0.0);
    let r = mat.roughness;
    let D = ggx_d(n_dot_h, r);
    // Matches CPU's combined-denominator form:
    //   spec = D * 0.5 / (NdotV·λ_L + NdotL·λ_V + ε)
    let alpha = r * r;
    let a2    = alpha * alpha;
    let lam_l = sqrt(max(0.0, a2 + (1.0 - a2) * n_dot_l * n_dot_l));
    let lam_v = sqrt(max(0.0, a2 + (1.0 - a2) * n_dot_v * n_dot_v));
    let G_over_denom = 0.5 / (n_dot_v * lam_l + n_dot_l * lam_v + GGX_EPSILON);
    // F0 = lerp(0.04, albedo, metallic) — metallic surfaces inherit the
    // basecolor as their F0; non-metals use a constant ~4% dielectric F0.
    let f0_dielectric = vec3<f32>(0.04, 0.04, 0.04);
    let f0 = mat.albedo * mat.metallic + f0_dielectric * (1.0 - mat.metallic);
    let F  = fresnel_schlick_color(v_dot_h, f0);
    return F * (D * G_over_denom);
}

fn eval_bsdf(mat: MaterialParams,
             wo: vec3<f32>, wi: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let n_dot_l = dot(n, wi);
    let n_dot_v = dot(n, wo);
    if (n_dot_l <= 0.0 || n_dot_v <= 0.0) { return vec3<f32>(0.0, 0.0, 0.0); }

    let specular = eval_specular(mat, wo, wi, n, n_dot_l, n_dot_v);

    let h_raw = wo + wi;
    let h_len = length(h_raw);
    let h     = select(vec3<f32>(0.0, 0.0, 1.0), h_raw / h_len, h_len > GGX_EPSILON);
    let v_dot_h = max(dot(wo, h), 0.0);

    // Diffuse attenuation uses the *dielectric* F0 = 0.04 (not metallic F0).
    let f0_d = vec3<f32>(0.04, 0.04, 0.04);
    let F    = fresnel_schlick_color(v_dot_h, f0_d);
    let kd   = (vec3<f32>(1.0, 1.0, 1.0) - F) * (1.0 - mat.metallic);
    let diffuse = eval_diffuse(mat, n_dot_l, n_dot_v) * kd;

    return diffuse + specular;
}

fn pdf_bsdf(mat: MaterialParams,
            wo: vec3<f32>, wi: vec3<f32>, n: vec3<f32>) -> f32 {
    let n_dot_l = dot(n, wi);
    if (n_dot_l <= 0.0) { return 0.0; }
    // Specular probability — matches CPU formula exactly.
    var spec_prob: f32;
    if (mat.metallic > 0.99) {
        spec_prob = 1.0;
    } else {
        spec_prob = 0.5 * (1.0 + mat.metallic) * (1.0 - mat.roughness * 0.5);
        spec_prob = clamp(spec_prob, 0.1, 0.9);
    }
    let h_raw = wo + wi;
    let h_len = length(h_raw);
    let h     = select(vec3<f32>(0.0, 0.0, 1.0), h_raw / h_len, h_len > GGX_EPSILON);
    let pdf_spec = pdf_ggx_vndf(wo, h, mat.roughness, n);
    let pdf_diff = pdf_cosine_hemisphere(n_dot_l);
    return (spec_prob * pdf_spec + (1.0 - spec_prob) * pdf_diff) * (1.0 - mat.transmission);
}

// Reflection: r = i - 2(i·n)n. `i` is the *incident* direction (points into the
// surface); pass `-wo` from a path tracer.
fn reflect_dir(i: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    return i - 2.0 * dot(i, n) * n;
}

// Refraction matching CPU `render::refract` ([units/render_units.hpp:845]):
//   cos_i = -dot(i, n);  sin²t = η² (1 - cos²i)
//   r = η·i + (η·cos_i − cos_t)·n
// Returns false on total internal reflection (`out_dir` left untouched).
fn refract_dir(i: vec3<f32>, n: vec3<f32>, eta: f32,
               out_dir: ptr<function, vec3<f32>>) -> bool {
    let cos_i = -dot(i, n);
    let sin2_t = eta * eta * (1.0 - cos_i * cos_i);
    if (sin2_t > 1.0) { return false; }
    let cos_t = sqrt(max(0.0, 1.0 - sin2_t));
    *out_dir = normalize(eta * i + (eta * cos_i - cos_t) * n);
    return true;
}

// ----------------------------------------------------------------------------
// BSDF sampling — incremental staging:
//   Step 3: cosine-hemisphere only for opaque materials. (LANDED)
//   Step 5: glass / transmission branch (this step).
//   Step 6: GGX-VNDF specular sampling + PrecomputedWeight for near-delta mirrors.
// ----------------------------------------------------------------------------

fn sample_bsdf(mat: MaterialParams, wo: vec3<f32>, n: vec3<f32>,
               u1: f32, u2: f32, u3: f32, u4: f32) -> BSDFSample {
    var s: BSDFSample;
    s.weight = vec3<f32>(0.0, 0.0, 0.0);

    // ---- Glass / transmission branch ([bsdf/bsdf.hpp:204-276]) ----
    if (mat.transmission > 0.0 && u3 < mat.transmission) {
        let wo_dot_n = dot(wo, n);
        let front_face = wo_dot_n > 0.0;
        let face_n = select(-n, n, front_face);
        let eta    = select(mat.ior, 1.0 / mat.ior, front_face);

        // Sample microfacet half-vector with GGX VNDF in face-normal space.
        let t  = build_ortho_basis_t(face_n);
        let bt = build_ortho_basis_b(face_n);
        let h  = sample_ggx_vndf(wo, mat.roughness, u1, u2, face_n, t, bt);

        let cos_theta_i = abs(dot(wo, h));
        let F = fresnel_dielectric(cos_theta_i, eta);

        let incident = -wo;
        // u4 picks Fresnel reflection vs refraction at the microfacet.
        if (u4 < F) {
            // Fresnel reflection (glass front face acts like a rough mirror).
            s.wi = reflect_dir(incident, h);
            let n_dot_l = dot(face_n, s.wi);
            let n_dot_v = dot(face_n, wo);
            if (n_dot_l <= 0.0 || n_dot_v <= 0.0) {
                s.kind = 2u; return s;
            }
            let G2 = ggx_g2(n_dot_l, n_dot_v, mat.roughness);
            let G1 = ggx_g1(n_dot_v, mat.roughness);
            let w  = G2 / (G1 + GGX_EPSILON);
            s.kind   = 1u;
            s.weight = vec3<f32>(w, w, w);
            return s;
        } else {
            // Refraction through microfacet (Snell on the *microfacet* normal h).
            var refracted: vec3<f32>;
            if (!refract_dir(incident, h, eta, &refracted)) {
                // TIR: behave as a reflection (same weight formula).
                s.wi = reflect_dir(incident, h);
                let n_dot_l = dot(face_n, s.wi);
                let n_dot_v = dot(face_n, wo);
                if (n_dot_l <= 0.0 || n_dot_v <= 0.0) {
                    s.kind = 2u; return s;
                }
                let G2 = ggx_g2(n_dot_l, n_dot_v, mat.roughness);
                let G1 = ggx_g1(n_dot_v, mat.roughness);
                let w  = G2 / (G1 + GGX_EPSILON);
                s.kind   = 1u;
                s.weight = vec3<f32>(w, w, w);
                return s;
            }
            s.wi = refracted;
            // Use |·| because the refracted ray lies on the OTHER side of face_n.
            let n_dot_l = abs(dot(s.wi, face_n));
            let n_dot_v = abs(dot(wo, face_n));
            let G2 = ggx_g2(n_dot_l, n_dot_v, mat.roughness);
            let G1 = ggx_g1(n_dot_v, mat.roughness);
            let w  = G2 / (G1 + GGX_EPSILON);
            s.kind   = 1u;
            s.weight = vec3<f32>(w, w, w);
            return s;
        }
    }

    // ---- Opaque: diffuse + specular blend ([bsdf/bsdf.hpp:278-344]) ----
    var spec_prob: f32;
    if (mat.metallic > 0.99) {
        spec_prob = 1.0;
    } else {
        spec_prob = 0.5 * (1.0 + mat.metallic) * (1.0 - mat.roughness * 0.5);
        spec_prob = clamp(spec_prob, 0.1, 0.9);
    }

    let t  = build_ortho_basis_t(n);
    let bt = build_ortho_basis_b(n);
    let n_dot_v = max(dot(n, wo), GGX_EPSILON);

    if (u1 < spec_prob) {
        // GGX-VNDF specular sample.
        let u1_adj = u1 / spec_prob;
        let h = sample_ggx_vndf(wo, mat.roughness, u1_adj, u2, n, t, bt);
        s.wi = reflect_dir(-wo, h);
        let n_dot_l = dot(n, s.wi);
        if (n_dot_l <= 0.0) {
            s.kind = 2u; return s;
        }
        let f = eval_bsdf(mat, wo, s.wi, n);
        let pdf_spec = pdf_ggx_vndf(wo, h, mat.roughness, n);
        let pdf_diff = pdf_cosine_hemisphere(n_dot_l);
        let pdf_raw  = (spec_prob * pdf_spec + (1.0 - spec_prob) * pdf_diff)
                       * (1.0 - mat.transmission);
        // Near-delta specular (very smooth metallic): convert to PrecomputedWeight
        // so the integrator absorbs f·cosθ/pdf in a single multiply — avoids the
        // numerical blowup of pdf→0 at vanishing roughness.
        if (mat.roughness < DELTA_ROUGHNESS_THRESHOLD && mat.metallic > 0.5) {
            let c = max(n_dot_l, 0.0);
            let w = f * (c / max(pdf_raw, GGX_EPSILON));
            s.kind   = 1u;
            s.weight = w;
            return s;
        }
        s.kind = 0u;
        s.f    = f;
        s.pdf  = pdf_raw;
        return s;
    } else {
        // Cosine-hemisphere diffuse sample.
        let u1_adj = (u1 - spec_prob) / max(1.0 - spec_prob, GGX_EPSILON);
        s.wi = cosine_hemisphere(n, u1_adj, u2);
        let n_dot_l = dot(n, s.wi);
        if (n_dot_l <= 0.0) {
            s.kind = 2u; return s;
        }
        let f = eval_bsdf(mat, wo, s.wi, n);

        // Half-vector for the specular PDF contribution to the blend.
        let h_raw = wo + s.wi;
        let h_len = length(h_raw);
        let h     = select(vec3<f32>(0.0, 0.0, 1.0), h_raw / h_len, h_len > GGX_EPSILON);
        let pdf_spec = pdf_ggx_vndf(wo, h, mat.roughness, n);
        let pdf_diff = pdf_cosine_hemisphere(n_dot_l);
        let pdf_raw  = (spec_prob * pdf_spec + (1.0 - spec_prob) * pdf_diff)
                       * (1.0 - mat.transmission);
        s.kind = 0u;
        s.f    = f;
        s.pdf  = pdf_raw;
        return s;
    }
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
// Robust AABB slab test using precomputed inverse direction. Treats infinities
// from a 0-component direction correctly (using sign of inv_dir).
// Returns true if the ray hits the slab in [0, t_max].
// ----------------------------------------------------------------------------

fn intersect_aabb(orig: vec3<f32>, inv_dir: vec3<f32>,
                  bmin: vec3<f32>, bmax: vec3<f32>, t_max: f32) -> bool {
    let t0 = (bmin - orig) * inv_dir;
    let t1 = (bmax - orig) * inv_dir;
    let tlo = min(t0, t1);
    let thi = max(t0, t1);
    let t_enter = max(max(tlo.x, tlo.y), tlo.z);
    let t_exit  = min(min(thi.x, thi.y), thi.z);
    return t_exit >= max(t_enter, 0.0) && t_enter <= t_max;
}

// ----------------------------------------------------------------------------
// BVH closest-hit traversal — stack-based, prunes by best_t.
// Writes hit info via pointer args.
// ----------------------------------------------------------------------------

struct ClosestHit {
    hit: bool,
    best_t: f32,
    best_n: vec3<f32>,
    mat:    MaterialParams,
};

fn trace_closest(orig: vec3<f32>, dir: vec3<f32>) -> ClosestHit {
    var result: ClosestHit;
    result.hit = false;
    result.best_t = 1e30;
    result.best_n = vec3<f32>(0.0, 0.0, 1.0);
    // Initial MaterialParams (zeroed — overwritten on hit).
    result.mat.albedo       = vec3<f32>(0.0, 0.0, 0.0);
    result.mat.metallic     = 0.0;
    result.mat.roughness    = 1.0;
    result.mat.transmission = 0.0;
    result.mat.ior          = 1.45;
    result.mat.emission     = vec3<f32>(0.0, 0.0, 0.0);

    if (params.bvh_node_count == 0u) { return result; }

    let inv_dir = vec3<f32>(1.0, 1.0, 1.0) / dir;

    var stack: array<i32, 64>;
    stack[0] = 0;
    var sp: i32 = 1;

    while (sp > 0) {
        sp = sp - 1;
        let node = bvh_nodes[stack[sp]];

        if (!intersect_aabb(orig, inv_dir, node.bmin, node.bmax, result.best_t)) {
            continue;
        }

        if (node.right_or_count > 0) {
            // Leaf — test triangles [left, left + right_or_count).
            let start = node.left;
            let count = node.right_or_count;
            for (var i: i32 = 0; i < count; i = i + 1) {
                let tri = tris[u32(start + i)];
                var t: f32; var u: f32; var v: f32;
                if (intersect_tri(orig, dir, tri.v0, tri.v1, tri.v2,
                                   &t, &u, &v) && t < result.best_t) {
                    result.best_t = t;
                    result.hit = true;
                    let w = 1.0 - u - v;
                    if (tri.smooth_flag > 0.5) {
                        result.best_n = normalize(w * tri.n0 + u * tri.n1 + v * tri.n2);
                    } else {
                        result.best_n = normalize(cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
                    }
                    result.mat = material_from_tri(tri);
                }
            }
        } else {
            // Internal — push children. Decode right child from -(right+1).
            let right = -(node.right_or_count + 1);
            if (sp < 62) {
                stack[sp] = right;     sp = sp + 1;
                stack[sp] = node.left; sp = sp + 1;
            }
        }
    }
    return result;
}

// ----------------------------------------------------------------------------
// BVH any-hit traversal (shadow ray) — returns true if any triangle is hit
// with t < max_t. Early-out as soon as one is found.
// ----------------------------------------------------------------------------

fn trace_any(orig: vec3<f32>, dir: vec3<f32>, max_t: f32) -> bool {
    if (params.bvh_node_count == 0u) { return false; }

    let inv_dir = vec3<f32>(1.0, 1.0, 1.0) / dir;

    var stack: array<i32, 64>;
    stack[0] = 0;
    var sp: i32 = 1;

    while (sp > 0) {
        sp = sp - 1;
        let node = bvh_nodes[stack[sp]];

        if (!intersect_aabb(orig, inv_dir, node.bmin, node.bmax, max_t)) {
            continue;
        }

        if (node.right_or_count > 0) {
            let start = node.left;
            let count = node.right_or_count;
            for (var i: i32 = 0; i < count; i = i + 1) {
                let tri = tris[u32(start + i)];
                var t: f32; var u: f32; var v: f32;
                if (intersect_tri(orig, dir, tri.v0, tri.v1, tri.v2,
                                   &t, &u, &v) && t < max_t) {
                    return true;
                }
            }
        } else {
            let right = -(node.right_or_count + 1);
            if (sp < 62) {
                stack[sp] = right;     sp = sp + 1;
                stack[sp] = node.left; sp = sp + 1;
            }
        }
    }
    return false;
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
            let h = trace_closest(orig, dir);

            // ---- Termination cases ----
            if (!h.hit) {
                radiance = radiance + throughput * (params.env_color * params.env_strength);
                break;
            }
            if (any(h.mat.emission > vec3<f32>(1e-6, 1e-6, 1e-6))) {
                radiance = radiance + throughput * h.mat.emission;
                break;
            }

            let mat = h.mat;
            let n_geom    = h.best_n;
            let wo        = -dir;
            let front_face = dot(wo, n_geom) > 0.0;
            let is_transmissive = mat.transmission > 0.5;

            // CPU rule ([integrator/path_tracer.hpp:109-117]):
            //   shading_n = (transmission<0.5 && !front_face) ? -n_geom : n_geom
            //   sample_n  = transmission>0.5 ? n_geom : shading_n
            var shading_n = n_geom;
            if (!is_transmissive && !front_face) {
                shading_n = -n_geom;
            }
            let sample_n = select(shading_n, n_geom, is_transmissive);

            let hit_point = orig + dir * h.best_t;

            // ---- NEE: direct contribution from every POINT light ----
            // Two CPU-parity branches ([light.hpp:196-233]):
            //   radius == 0 (delta): irradiance = emission / d² ; pdf = 1/sr
            //   radius > 0  (sphere): uniform sample on sphere surface;
            //                         emission stays (raw radiance), divide by
            //                         pdf_w = d² / (area · cos_light).
            // The C++ side has already absorbed the energy/(4π) for delta and
            // energy/(π·area) for sphere into `light.emission`, so the shader
            // does not divide by 4π again.
            //
            // Skip NEE entirely when transmission > 0.5 (glass relies on the
            // BSDF-sampled transmitted/reflected ray for direct lighting).
            if (!is_transmissive) {
                for (var li: u32 = 0u; li < params.point_light_count; li = li + 1u) {
                    let light = point_lights[li];

                    var contribution = vec3<f32>(0.0, 0.0, 0.0);
                    var light_dir    = vec3<f32>(0.0, 0.0, 1.0);
                    var light_dist   = 0.0;
                    var skip         = true;

                    if (light.radius > 0.0) {
                        // Sphere — uniform sample on the surface using 2 RNG
                        // draws independent from the BSDF sample (rng state is
                        // advanced; same paths still see consistent splits).
                        let lu1 = rand_f32(&rng);
                        let lu2 = rand_f32(&rng);
                        let z   = 1.0 - 2.0 * lu1;
                        let rr  = sqrt(max(0.0, 1.0 - z * z));
                        let phi = 6.2831853 * lu2;
                        let off = vec3<f32>(rr * cos(phi), rr * sin(phi), z);
                        let light_p = light.pos + off * light.radius;
                        let light_n = off;                         // outward sphere normal
                        let to_light = light_p - hit_point;
                        let d2 = dot(to_light, to_light);
                        if (d2 >= 1e-8) {
                            let d = sqrt(d2);
                            light_dir  = to_light / d;
                            light_dist = d;
                            let cos_light = dot(light_n, -light_dir);
                            let cos_theta = dot(shading_n, light_dir);
                            if (cos_light >= 1e-6 && cos_theta > 0.0) {
                                let f = eval_bsdf(mat, wo, light_dir, shading_n);
                                // contribution = f · L · cosθ_surface / pdf_w
                                //              = f · L · cosθ_surface · area · cosθ_light / d²
                                contribution = f * light.emission * cos_theta
                                               * (light.area * cos_light / d2);
                                skip = false;
                            }
                        }
                    } else {
                        // Delta point — single direction, pdf is a δ.
                        let to_light = light.pos - hit_point;
                        let d2 = dot(to_light, to_light);
                        if (d2 >= 1e-8) {
                            let d = sqrt(d2);
                            light_dir  = to_light / d;
                            light_dist = d;
                            let cos_theta = dot(shading_n, light_dir);
                            if (cos_theta > 0.0) {
                                let f = eval_bsdf(mat, wo, light_dir, shading_n);
                                contribution = f * (light.emission / d2) * cos_theta;
                                skip = false;
                            }
                        }
                    }

                    if (skip) { continue; }

                    let shadow_orig = hit_point + shading_n * 1e-3;
                    let max_t       = light_dist - 2e-3;
                    if (trace_any(shadow_orig, light_dir, max_t)) { continue; }

                    radiance = radiance + throughput * contribution;
                }
            }

            // ---- Russian roulette from depth 3 ----
            if (b >= 3u) {
                let est = throughput * mat.albedo;
                let p = min(max(est.r, max(est.g, est.b)), 0.95);
                if (rand_f32(&rng) > p) { break; }
                throughput = throughput / p;
            }

            // ---- Next ray: sample_bsdf (Step 3 = diffuse path only) ----
            let u1 = rand_f32(&rng);
            let u2 = rand_f32(&rng);
            let u3 = rand_f32(&rng);
            let u4 = rand_f32(&rng);
            let bs = sample_bsdf(mat, wo, sample_n, u1, u2, u3, u4);
            if (bs.kind == 2u) { break; }

            var weight: vec3<f32>;
            if (bs.kind == 1u) {
                weight = bs.weight;
            } else {
                let c = abs(dot(sample_n, bs.wi));
                if (c <= 1e-6 || bs.pdf < 1e-6) { break; }
                weight = bs.f * (c / bs.pdf);
            }
            throughput = throughput * weight;

            // Offset along the geometric normal in the direction the new ray
            // is going. For reflection wi·n_geom > 0 (offset outward); for
            // transmission wi·n_geom < 0 (offset inward) so the ray clears the
            // opposite side of the surface.
            let offset_dir = select(-n_geom, n_geom, dot(bs.wi, n_geom) > 0.0);
            let new_orig = hit_point + offset_dir * 1e-3;
            dir  = bs.wi;
            orig = new_orig;
        }

        radiance_sum = radiance_sum + radiance;
    }

    // Y-flip on store — same convention as debug_normal.wgsl. Additive so the
    // same shader serves both the sync single-dispatch model (caller clears
    // out_pixels before submitting) and the async accumulator model (caller
    // clears once, then submits many 1-sample dispatches in a worker thread).
    // The .w channel carries the running sample count contributed by this
    // pixel, used to normalise on readback in the async path. The sync path
    // ignores .w and rewrites it to 1.0 on the CPU side ([pybind_renderer.hpp]).
    let out_y = params.tile_h - 1u - gid.y;
    let idx   = out_y * params.tile_w + gid.x;
    let prev  = out_pixels[idx];
    out_pixels[idx] = vec4<f32>(prev.xyz + radiance_sum, prev.w + f32(params.samples));
}
