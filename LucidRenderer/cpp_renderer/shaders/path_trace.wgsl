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
    light_count: u32,
    bvh_node_count: u32,
    algorithm: u32,    // 0=simple, 1=nee, 2=mis
    _p6: u32,
};

// Unified light (8 vec4s / 128 B). Mirrors C++ `GpuLight`. The shader branches
// on `type` to interpret type-specific fields; unused slots are zeroed on the
// CPU pack side. `emission` is already premultiplied (delta point: color *
// energy/(4π); sphere point: color * energy/(π·area); SUN/SPOT: irradiance
// scaled per JSON parse; AREA / EMISSIVE_MESH: raw radiance).
struct Light {
    kind:           u32, area:          f32, selection_pdf: f32, _p0:        u32,
    position:       vec3<f32>, _p1:     f32,
    emission:       vec3<f32>, _p2:     f32,
    normal:         vec3<f32>, radius:  f32,
    right:          vec3<f32>, sizeX:   f32,
    up:             vec3<f32>, sizeY:   f32,
    v1:             vec3<f32>, spotAngle: f32,
    v2:             vec3<f32>, spotBlend: f32,
};

// Light type discriminants (mirror enum class GpuLightType in path_tracer.hpp)
const LT_POINT:         u32 = 0u;
const LT_SUN:           u32 = 1u;
const LT_SPOT:          u32 = 2u;
const LT_AREA_RECT:     u32 = 3u;
const LT_AREA_DISK:     u32 = 4u;
const LT_AREA_ELLIPSE:  u32 = 5u;
const LT_EMISSIVE_MESH: u32 = 6u;

// Integrator algorithm discriminants
const ALGO_SIMPLE: u32 = 0u;
const ALGO_NEE:    u32 = 1u;
const ALGO_MIS:    u32 = 2u;

// Triangle (10 vec4s / 160 B). Material fields cover the full Principled BSDF
// socket set (albedo / metallic / roughness / transmission / ior / emission)
// constant-folded at uv=(0,0). The reserved vec4's first slot now carries
// `light_idx` (bit-cast from i32) — index into `lights` buffer when this
// triangle is an EMISSIVE_MESH light, -1 otherwise. Used to compute
// pdf_light in MIS when a BSDF sample hits the emissive surface.
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
    light_idx_f:  f32,   // bitcast<i32> for the light index, -1 if none
    _reserved2:   f32,
    _reserved3:   f32,
    _reserved4:   f32,
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
@group(0) @binding(3) var<storage, read>       lights       : array<Light>;
@group(0) @binding(4) var<storage, read>       bvh_nodes    : array<BvhNode>;
@group(0) @binding(5) var<storage, read>       light_cdf    : array<f32>;

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
// Light selection + sampling (mirrors CPU sceneLights + sampleLightTyped)
// ----------------------------------------------------------------------------
//
// `LightSample`:
//   position [m], direction (normalised, surface→light), distance [m],
//   emission [W/(sr·m²)] (radiance, premultiplied on CPU side), normal,
//   pdf [1/sr] in solid-angle measure (0 means "delta" — no MIS partner),
//   delta=true for POINT(radius=0)/SUN/SPOT.

struct LightSample {
    position:  vec3<f32>,
    direction: vec3<f32>,
    distance:  f32,
    emission:  vec3<f32>,
    normal:    vec3<f32>,
    pdf:       f32,
    delta:     u32,   // 1 = delta light (no MIS partner); 0 = area / sphere
};

// select_light: area-weighted CDF binary search. Returns the chosen light
// index; the selection_pdf is read from the Light struct.
fn select_light(u: f32) -> u32 {
    if (params.light_count == 0u) { return 0u; }
    var lo: u32 = 0u;
    var hi: u32 = params.light_count;
    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (light_cdf[mid] < u) { lo = mid + 1u; } else { hi = mid; }
    }
    return min(lo, params.light_count - 1u);
}

fn sample_triangle_point(v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>,
                         u1: f32, u2: f32) -> vec3<f32> {
    let su = sqrt(u1);
    let b0 = 1.0 - su;
    let b1 = u2 * su;
    let b2 = 1.0 - b0 - b1;
    return v0 * b0 + v1 * b1 + v2 * b2;
}

fn sample_light(light: Light, shading_point: vec3<f32>,
                u1: f32, u2: f32) -> LightSample {
    var s: LightSample;
    s.position  = light.position;
    s.normal    = light.normal;
    s.emission  = light.emission;
    s.direction = vec3<f32>(0.0, 0.0, 1.0);
    s.distance  = 0.0;
    s.pdf       = 0.0;
    s.delta     = 0u;

    switch (light.kind) {
        case 0u: {   // LT_POINT
            if (light.radius > 1e-6) {
                // Sphere — uniform surface sample, area-pdf for MIS.
                let z   = 1.0 - 2.0 * u1;
                let rr  = sqrt(max(0.0, 1.0 - z * z));
                let phi = 6.2831853 * u2;
                let off = vec3<f32>(rr * cos(phi), rr * sin(phi), z);
                s.position = light.position + off * light.radius;
                s.normal   = off;
                let to_light = s.position - shading_point;
                let d2 = max(dot(to_light, to_light), 1e-12);
                s.distance  = sqrt(d2);
                s.direction = to_light / s.distance;
                let cos_l = dot(s.normal, -s.direction);
                if (cos_l < 1e-6) {
                    s.pdf = 0.0;
                } else {
                    s.pdf = d2 / (light.area * cos_l);
                }
            } else {
                // Delta point: emission already absorbs 1/(4π); contribution
                // applies 1/d² at integration time. pdf = 1 (delta).
                let to_light = light.position - shading_point;
                let d2 = max(dot(to_light, to_light), 1e-12);
                s.distance  = sqrt(d2);
                s.direction = to_light / s.distance;
                s.emission  = light.emission / d2;
                s.pdf       = 1.0;
                s.delta     = 1u;
            }
        }
        case 1u: {   // LT_SUN
            s.direction = -light.normal;
            s.distance  = 1e6;
            s.position  = shading_point + s.direction * s.distance;
            s.pdf       = 1.0;
            s.delta     = 1u;
        }
        case 2u: {   // LT_SPOT
            let to_light = light.position - shading_point;
            let d2 = max(dot(to_light, to_light), 1e-12);
            s.distance  = sqrt(d2);
            s.direction = to_light / s.distance;
            // Cone test against light.normal (axis pointing outward from light).
            let cos_angle = dot(light.normal, -s.direction);
            let cos_cone  = cos(light.spotAngle * 0.5);
            if (cos_angle < cos_cone) {
                s.emission = vec3<f32>(0.0, 0.0, 0.0);
            } else if (light.spotBlend > 0.0) {
                let t = (cos_angle - cos_cone) / max(1e-6, 1.0 - cos_cone);
                let falloff = min(1.0, t / light.spotBlend);
                s.emission = light.emission * falloff / d2;
            } else {
                s.emission = light.emission / d2;
            }
            s.pdf   = 1.0;
            s.delta = 1u;
        }
        case 3u: {   // LT_AREA_RECT
            let lx = light.sizeX * (u1 - 0.5);
            let ly = light.sizeY * (u2 - 0.5);
            s.position = light.position + light.right * lx + light.up * ly;
            let to_light = s.position - shading_point;
            let d2 = max(dot(to_light, to_light), 1e-12);
            s.distance  = sqrt(d2);
            s.direction = to_light / s.distance;
            let cos_l = dot(s.normal, -s.direction);
            if (cos_l < 1e-6) {
                s.pdf = 0.0;
            } else {
                s.pdf = d2 / (light.area * cos_l);
            }
        }
        case 4u, 5u: {  // LT_AREA_DISK / ELLIPSE
            let r_sample = sqrt(u1);
            let theta    = 6.2831853 * u2;
            let lx = light.sizeX * 0.5 * r_sample * cos(theta);
            let ly = light.sizeY * 0.5 * r_sample * sin(theta);
            s.position = light.position + light.right * lx + light.up * ly;
            let to_light = s.position - shading_point;
            let d2 = max(dot(to_light, to_light), 1e-12);
            s.distance  = sqrt(d2);
            s.direction = to_light / s.distance;
            let cos_l = dot(s.normal, -s.direction);
            if (cos_l < 1e-6) {
                s.pdf = 0.0;
            } else {
                s.pdf = d2 / (light.area * cos_l);
            }
        }
        case 6u, default: {   // LT_EMISSIVE_MESH (triangle barycentric sample)
            s.position = sample_triangle_point(light.position, light.v1, light.v2, u1, u2);
            let to_light = s.position - shading_point;
            let d2 = max(dot(to_light, to_light), 1e-12);
            s.distance  = sqrt(d2);
            s.direction = to_light / s.distance;
            // Emissive triangles are 2-sided in CPU: use |cos|.
            let cos_l = abs(dot(s.normal, -s.direction));
            if (cos_l < 1e-6) {
                s.pdf = 0.0;
            } else {
                s.pdf = d2 / (light.area * cos_l);
            }
        }
    }
    return s;
}

// pdf_light_sample: solid-angle pdf for a hit on `light` at `light_p` from
// `shading_p`, given the light's surface normal at that point. Returns 0
// for delta lights or invalid geometries (back-face, parallel ray, etc.).
fn pdf_light_sample(light: Light, shading_p: vec3<f32>,
                    light_p: vec3<f32>, light_n: vec3<f32>) -> f32 {
    // Delta lights are not sampled by BSDF / native intersection — no MIS partner.
    if (light.kind == LT_SUN || light.kind == LT_SPOT) { return 0.0; }
    if (light.kind == LT_POINT && light.radius < 1e-6) { return 0.0; }
    if (light.area < 1e-12) { return 0.0; }
    let to_light = light_p - shading_p;
    let d2 = dot(to_light, to_light);
    if (d2 < 1e-12) { return 0.0; }
    let d = sqrt(d2);
    let dir = to_light / d;
    // EMISSIVE_MESH triangles are 2-sided; other area lights use signed cos.
    let two_sided = light.kind == LT_EMISSIVE_MESH;
    let cos_l_raw = dot(light_n, -dir);
    let cos_l = select(cos_l_raw, abs(cos_l_raw), two_sided);
    if (cos_l < 1e-6) { return 0.0; }
    return d2 / (light.area * cos_l);
}

// Power-heuristic MIS weight with β=2 (standard balance/power heuristic).
fn mis_power_heuristic(pdf_a: f32, pdf_b: f32) -> f32 {
    let a2 = pdf_a * pdf_a;
    let b2 = pdf_b * pdf_b;
    let denom = a2 + b2;
    if (denom < 1e-12) { return 0.0; }
    return a2 / denom;
}

// ----------------------------------------------------------------------------
// Native-light primary intersection (POINT sphere, AREA rect/disk/ellipse)
// EMISSIVE_MESH triangles are part of the BVH so they're caught by trace_closest.
// CPU intersectNativeLights ([scene_lights.hpp:176]) only fires at depth > 0;
// here the caller passes the depth (b) to gate visibility.
// ----------------------------------------------------------------------------

struct NativeLightHit {
    hit:       u32,
    t:         f32,
    light_idx: i32,
    point:     vec3<f32>,
    normal:    vec3<f32>,
    emission:  vec3<f32>,
};

fn intersect_sphere(orig: vec3<f32>, dir: vec3<f32>,
                    center: vec3<f32>, radius: f32,
                    out_t: ptr<function, f32>,
                    out_n: ptr<function, vec3<f32>>) -> bool {
    let oc = orig - center;
    let b  = dot(oc, dir);
    let c  = dot(oc, oc) - radius * radius;
    let disc = b * b - c;
    if (disc < 0.0) { return false; }
    let sq = sqrt(disc);
    var t = -b - sq;
    if (t < 1e-4) { t = -b + sq; }
    if (t < 1e-4) { return false; }
    let p = orig + dir * t;
    *out_t = t;
    *out_n = (p - center) / radius;
    return true;
}

// Disk / ellipse intersection: plane test + 2D ellipse coordinate test.
fn intersect_plane_ellipse(orig: vec3<f32>, dir: vec3<f32>,
                           pos: vec3<f32>, n: vec3<f32>,
                           right: vec3<f32>, up: vec3<f32>,
                           rx: f32, ry: f32,
                           rectangle: bool,
                           out_t: ptr<function, f32>) -> bool {
    let denom = dot(n, dir);
    if (abs(denom) < 1e-6) { return false; }
    let t = dot(pos - orig, n) / denom;
    if (t < 1e-4) { return false; }
    let p = orig + dir * t - pos;
    let lx = dot(p, right);
    let ly = dot(p, up);
    if (rectangle) {
        if (abs(lx) > rx || abs(ly) > ry) { return false; }
    } else {
        let nx = lx / rx;
        let ny = ly / ry;
        if (nx * nx + ny * ny > 1.0) { return false; }
    }
    *out_t = t;
    return true;
}

fn intersect_native_lights(orig: vec3<f32>, dir: vec3<f32>,
                           max_t: f32) -> NativeLightHit {
    var result: NativeLightHit;
    result.hit       = 0u;
    result.t         = max_t;
    result.light_idx = -1;
    result.point     = vec3<f32>(0.0, 0.0, 0.0);
    result.normal    = vec3<f32>(0.0, 0.0, 1.0);
    result.emission  = vec3<f32>(0.0, 0.0, 0.0);

    for (var i: u32 = 0u; i < params.light_count; i = i + 1u) {
        let light = lights[i];
        if (light.kind == LT_POINT) {
            if (light.radius < 1e-6) { continue; }
            var t: f32;
            var n: vec3<f32>;
            if (intersect_sphere(orig, dir, light.position, light.radius, &t, &n)
                && t < result.t) {
                result.hit       = 1u;
                result.t         = t;
                result.light_idx = i32(i);
                result.point     = orig + dir * t;
                result.normal    = n;
                result.emission  = light.emission;
            }
        } else if (light.kind == LT_AREA_RECT
                || light.kind == LT_AREA_DISK
                || light.kind == LT_AREA_ELLIPSE) {
            // Front-face only (CPU: facing = dot(n, ray.dir) < 0 means front).
            if (dot(light.normal, dir) >= 0.0) { continue; }
            var t: f32;
            let rectangle = light.kind == LT_AREA_RECT;
            let rx = select(light.sizeX * 0.5, light.sizeX * 0.5, rectangle);
            let ry = select(light.sizeY * 0.5, light.sizeY * 0.5, rectangle);
            // For rectangle the half-size used by the test is sizeX/sizeY/2 too.
            let hx = light.sizeX * 0.5;
            let hy = light.sizeY * 0.5;
            if (intersect_plane_ellipse(orig, dir, light.position, light.normal,
                                         light.right, light.up,
                                         hx, hy, rectangle, &t)
                && t < result.t) {
                result.hit       = 1u;
                result.t         = t;
                result.light_idx = i32(i);
                result.point     = orig + dir * t;
                result.normal    = light.normal;
                result.emission  = light.emission;
            }
        }
        // SUN/SPOT/EMISSIVE_MESH skipped (delta or in BVH).
    }
    return result;
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
    light_idx: i32,   // bitcast<i32>(tri.light_idx_f) on hit; -1 otherwise
};

fn trace_closest(orig: vec3<f32>, dir: vec3<f32>) -> ClosestHit {
    var result: ClosestHit;
    result.hit = false;
    result.best_t = 1e30;
    result.best_n = vec3<f32>(0.0, 0.0, 1.0);
    result.light_idx = -1;
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
                    result.light_idx = bitcast<i32>(tri.light_idx_f);
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

        // last_bsdf_pdf: PDF of the BSDF sample that *produced* the current
        // direction `dir`. Used to weight BSDF-sampled lighting via MIS
        // (mirrors CPU's `lastBsdfPdf` tracking in traceMIS). 0 means
        // "delta / first hit" → no MIS partner, weight = 1.
        var last_bsdf_pdf = 0.0;

        for (var b: u32 = 0u; b < params.max_bounces; b = b + 1u) {
            // Closest BVH triangle hit AND closest native-light primary
            // intersection (POINT spheres + AREA primitives). The smaller t
            // wins. Native lights are invisible to the camera ray (b == 0).
            let h = trace_closest(orig, dir);
            let tri_t = select(1e30, h.best_t, h.hit);
            var lh: NativeLightHit;
            lh.hit = 0u; lh.t = 1e30; lh.light_idx = -1;
            if (b > 0u && params.light_count > 0u
                && params.algorithm != ALGO_SIMPLE) {
                lh = intersect_native_lights(orig, dir, tri_t);
            }

            // Case 1: native light hit before any triangle.
            if (lh.hit == 1u && lh.t < tri_t) {
                var mis_w = 1.0;
                if (params.algorithm == ALGO_MIS && last_bsdf_pdf > 1e-6) {
                    let light = lights[lh.light_idx];
                    let pdf_l = pdf_light_sample(light, orig, lh.point, lh.normal)
                                * light.selection_pdf;
                    mis_w = mis_power_heuristic(last_bsdf_pdf, pdf_l);
                }
                radiance = radiance + throughput * lh.emission * mis_w;
                break;
            }

            // Case 2: nothing hit → env light, terminate.
            if (!h.hit) {
                radiance = radiance + throughput * (params.env_color * params.env_strength);
                break;
            }

            // Case 3: emissive triangle hit. With MIS, weight by light pdf
            // when the triangle is in the light buffer.
            if (any(h.mat.emission > vec3<f32>(1e-6, 1e-6, 1e-6))) {
                var mis_w = 1.0;
                if (params.algorithm == ALGO_MIS
                    && last_bsdf_pdf > 1e-6
                    && h.light_idx >= 0) {
                    let light = lights[h.light_idx];
                    let hp = orig + dir * h.best_t;
                    let pdf_l = pdf_light_sample(light, orig, hp, h.best_n)
                                * light.selection_pdf;
                    mis_w = mis_power_heuristic(last_bsdf_pdf, pdf_l);
                }
                radiance = radiance + throughput * h.mat.emission * mis_w;
                break;
            }

            let mat = h.mat;
            let n_geom    = h.best_n;
            let wo        = -dir;
            let front_face = dot(wo, n_geom) > 0.0;
            let is_transmissive = mat.transmission > 0.5;

            var shading_n = n_geom;
            if (!is_transmissive && !front_face) {
                shading_n = -n_geom;
            }
            let sample_n = select(shading_n, n_geom, is_transmissive);
            let hit_point = orig + dir * h.best_t;

            // ---------------- NEE step (algorithm-dependent) ----------------
            // ALGO_SIMPLE: Phase 2a behaviour — only POINT (delta) lights
            // contribute, no other types. Kept for backward compat / debugging.
            // ALGO_NEE / ALGO_MIS: select_light + sample_light + shadow ray.
            if (!is_transmissive && params.light_count > 0u) {
                if (params.algorithm == ALGO_SIMPLE) {
                    // Legacy POINT-only path: walk all lights, evaluate only
                    // those with type==LT_POINT (delta sphere or delta point).
                    for (var li: u32 = 0u; li < params.light_count; li = li + 1u) {
                        let light = lights[li];
                        if (light.kind != LT_POINT) { continue; }
                        let lu1 = rand_f32(&rng);
                        let lu2 = rand_f32(&rng);
                        let ls = sample_light(light, hit_point, lu1, lu2);
                        if (ls.pdf < 1e-6) { continue; }
                        let cos_theta = dot(shading_n, ls.direction);
                        if (cos_theta <= 0.0) { continue; }
                        let shadow_orig = hit_point + shading_n * 1e-3;
                        if (trace_any(shadow_orig, ls.direction, ls.distance - 2e-3)) {
                            continue;
                        }
                        let f = eval_bsdf(mat, wo, ls.direction, shading_n);
                        if (ls.delta == 1u) {
                            // emission already includes inverse-square (delta) or
                            // raw radiance (sphere/area); pdf=1 means delta direction.
                            radiance = radiance + throughput * f * ls.emission * cos_theta;
                        } else {
                            // Sphere: f * L * cosθ / pdf_w (no select_prob in simple).
                            radiance = radiance + throughput * f * ls.emission
                                                   * cos_theta / ls.pdf;
                        }
                    }
                } else {
                    // NEE / MIS: importance-sampled light selection.
                    let su = rand_f32(&rng);
                    let lu1 = rand_f32(&rng);
                    let lu2 = rand_f32(&rng);
                    let light_idx = select_light(su);
                    let light = lights[light_idx];
                    let ls = sample_light(light, hit_point, lu1, lu2);
                    let select_prob = light.selection_pdf;
                    if (ls.pdf > 1e-6 && select_prob > 1e-6) {
                        let cos_theta = dot(shading_n, ls.direction);
                        if (cos_theta > 0.0) {
                            let shadow_orig = hit_point + shading_n * 1e-3;
                            if (!trace_any(shadow_orig, ls.direction,
                                           ls.distance - 2e-3)) {
                                let f = eval_bsdf(mat, wo, ls.direction, shading_n);
                                let pdf_l_combined = ls.pdf * select_prob;
                                var contribution = f * ls.emission * cos_theta
                                                   / pdf_l_combined;
                                if (params.algorithm == ALGO_MIS && ls.delta == 0u) {
                                    let pdf_b = pdf_bsdf(mat, wo, ls.direction, shading_n);
                                    let mis_w = mis_power_heuristic(pdf_l_combined, pdf_b);
                                    contribution = contribution * mis_w;
                                }
                                radiance = radiance + throughput * contribution;
                            }
                        }
                    }
                }
            }

            // ---- Russian roulette from depth 3 ----
            if (b >= 3u) {
                let est = throughput * mat.albedo;
                let p = min(max(est.r, max(est.g, est.b)), 0.95);
                if (rand_f32(&rng) > p) { break; }
                throughput = throughput / p;
            }

            // ---- Next ray: BSDF sample ----
            let u1 = rand_f32(&rng);
            let u2 = rand_f32(&rng);
            let u3 = rand_f32(&rng);
            let u4 = rand_f32(&rng);
            let bs = sample_bsdf(mat, wo, sample_n, u1, u2, u3, u4);
            if (bs.kind == 2u) { break; }

            var weight: vec3<f32>;
            var bsdf_pdf_for_next = 0.0;
            if (bs.kind == 1u) {
                // PrecomputedWeight = delta-like (glass / near-mirror); breaks
                // the MIS chain — next hit's emission MIS weight = 1.
                weight = bs.weight;
                bsdf_pdf_for_next = 0.0;
            } else {
                let c = abs(dot(sample_n, bs.wi));
                if (c <= 1e-6 || bs.pdf < 1e-6) { break; }
                weight = bs.f * (c / bs.pdf);
                bsdf_pdf_for_next = bs.pdf;
            }
            throughput = throughput * weight;
            last_bsdf_pdf = bsdf_pdf_for_next;

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
