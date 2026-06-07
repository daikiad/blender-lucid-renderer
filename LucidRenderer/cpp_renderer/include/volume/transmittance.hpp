/**
 * transmittance.hpp - Volume Sampling Helpers
 * ============================================
 *
 * Free functions for participating-media integration shared between
 * `renderer.hpp` (debug viz, Stage A integrator) and `path_tracer.hpp`
 * (`traceVolumeMIS`).
 *
 * Provides:
 *   - ray_aabb_segment                      (slab clip)
 *   - sample_grid_trilinear                 (voxel sample)
 *   - transmittance_along_segment           (Stage A, absorption-only)
 *   - extinction_transmittance_along_segment (Stage MIS, full σ_t)
 *   - sample_volume_distance + VolumeScatterEvent (delta-tracking)
 */

#pragma once

#include "../units/render_units.hpp"
#include "../math/random.hpp"
#include "../core/ray.hpp"
#include "../core/material.hpp"
#include "../core/scene.hpp"
#include <algorithm>
#include <cmath>

// Forward-declared so the volume-transmittance helpers can call it before its
// own definition appears below.
inline float sample_grid_trilinear(const VolumeProperties& vol,
                                    float wx, float wy, float wz);

// Clip the ray segment [0, t_max] against a world-axis-aligned bbox.
// Standard slab test; returns true if there's a non-empty interval inside
// the box, with t_enter / t_exit set in metres.
inline bool ray_aabb_segment(const Ray& ray, float t_max,
                              const float bbox_min[3], const float bbox_max[3],
                              float& t_enter, float& t_exit) {
    const auto ro = render::displacement_from_origin(ray.origin)
                        .numerical_value_in(render::si::metre);
    const auto rd = ray.direction.vec();
    float t0 = 0.0f;
    float t1 = t_max;
    for (int axis = 0; axis < 3; ++axis) {
        const float r_o = axis == 0 ? ro.x : axis == 1 ? ro.y : ro.z;
        const float r_d = axis == 0 ? rd.x : axis == 1 ? rd.y : rd.z;
        if (std::abs(r_d) < 1e-12f) {
            if (r_o < bbox_min[axis] || r_o > bbox_max[axis]) return false;
            continue;
        }
        const float inv = 1.0f / r_d;
        float ta = (bbox_min[axis] - r_o) * inv;
        float tb = (bbox_max[axis] - r_o) * inv;
        if (ta > tb) std::swap(ta, tb);
        if (ta > t0) t0 = ta;
        if (tb < t1) t1 = tb;
        if (t0 > t1) return false;
    }
    t_enter = std::max(0.0f, t0);
    t_exit  = t1;
    return t_exit > t_enter;
}

// =============================================================================
// Volume transmittance — Stage A (absorption only).
// =============================================================================
//
// Convention (Cycles Principled-Volume style):
//   σ_a,c = vol.density * grid_sample(x) * (1 - vol.color_c)
//   T_c   = exp(-∫ σ_a,c ds)
//   Apply by:  throughput *= transmittance_along_segment(...)
//
// Stage A skips scattering entirely. Use for `traceVolumeSimple` and any
// "smoke just darkens what's behind" debug visualization.
//
// For the full scattering integrator use `extinction_transmittance_along_segment`
// below — same shape but uses σ_t (the color factor goes to throughput instead).
inline render::AttenuationRGB transmittance_along_segment(
    const Scene& scene, const Ray& ray, render::Length t_max) {

    float tau_r = 0.0f, tau_g = 0.0f, tau_b = 0.0f;
    const float t_max_f = t_max.numerical_value_in(render::si::metre);

    const auto ro = render::displacement_from_origin(ray.origin)
                        .numerical_value_in(render::si::metre);
    const auto rd = ray.direction.vec();

    for (const auto& mesh : scene.meshes) {
        const auto& vol = mesh.material.volume;
        if (!vol.present() || !vol.has_grid()) continue;

        float t_in = 0.0f, t_out = 0.0f;
        if (!ray_aabb_segment(ray, t_max_f,
                              vol.grid_world_min, vol.grid_world_max,
                              t_in, t_out)) {
            continue;
        }
        const float seg_len = t_out - t_in;
        if (seg_len <= 0.0f) continue;

        const float extent = std::max({
            vol.grid_world_max[0] - vol.grid_world_min[0],
            vol.grid_world_max[1] - vol.grid_world_min[1],
            vol.grid_world_max[2] - vol.grid_world_min[2],
        });
        const int max_dim = std::max({
            vol.grid_dims[0], vol.grid_dims[1], vol.grid_dims[2]});
        const float voxel_size = extent / std::max(1, max_dim);
        const float step = std::max(1e-4f, voxel_size * 0.5f);
        const int n_steps = std::min(2048, static_cast<int>(seg_len / step) + 1);

        for (int i = 0; i < n_steps; ++i) {
            const float t = t_in + (static_cast<float>(i) + 0.5f) * step;
            if (t > t_out) break;
            const float wx = ro.x + rd.x * t;
            const float wy = ro.y + rd.y * t;
            const float wz = ro.z + rd.z * t;
            const float d  = sample_grid_trilinear(vol, wx, wy, wz);
            if (d <= 0.0f) continue;
            const float sigma_a = vol.density * d * step;
            tau_r += sigma_a * (1.0f - vol.color.r);
            tau_g += sigma_a * (1.0f - vol.color.g);
            tau_b += sigma_a * (1.0f - vol.color.b);
        }
    }

    return render::make_attenuation_rgb(
        std::exp(-tau_r), std::exp(-tau_g), std::exp(-tau_b));
}

// =============================================================================
// Volume extinction transmittance — for `traceVolumeMIS` (scattering integrator).
// =============================================================================
//
// Same shape as `transmittance_along_segment` but integrates σ_t (extinction)
// instead of σ_a (absorption). In Cycles' Principled Volume convention σ_t is
// monochromatic (= density · grid_sample), so the per-channel result here will
// be the same r/g/b — we still return AttenuationRGB to keep the call sites
// symmetric with the absorption-only helper and to leave room for per-channel
// σ_t (spectral grids) later.
//
// Why this exists alongside `transmittance_along_segment`: that helper is the
// Stage-A absorption-only approximation. With scattering modelled, the right
// extinction is the full σ_t — the color factor moves to the throughput at
// each scatter event (`throughput *= color`, since σ_s/σ_t = color).
inline render::AttenuationRGB extinction_transmittance_along_segment(
    const Scene& scene, const Ray& ray, render::Length t_max) {

    float tau_r = 0.0f, tau_g = 0.0f, tau_b = 0.0f;
    const float t_max_f = t_max.numerical_value_in(render::si::metre);

    const auto ro = render::displacement_from_origin(ray.origin)
                        .numerical_value_in(render::si::metre);
    const auto rd = ray.direction.vec();

    for (const auto& mesh : scene.meshes) {
        const auto& vol = mesh.material.volume;
        if (!vol.present() || !vol.has_grid()) continue;

        float t_in = 0.0f, t_out = 0.0f;
        if (!ray_aabb_segment(ray, t_max_f,
                              vol.grid_world_min, vol.grid_world_max,
                              t_in, t_out)) {
            continue;
        }
        const float seg_len = t_out - t_in;
        if (seg_len <= 0.0f) continue;

        const float extent = std::max({
            vol.grid_world_max[0] - vol.grid_world_min[0],
            vol.grid_world_max[1] - vol.grid_world_min[1],
            vol.grid_world_max[2] - vol.grid_world_min[2],
        });
        const int max_dim = std::max({
            vol.grid_dims[0], vol.grid_dims[1], vol.grid_dims[2]});
        const float voxel_size = extent / std::max(1, max_dim);
        const float step = std::max(1e-4f, voxel_size * 0.5f);
        const int n_steps = std::min(2048, static_cast<int>(seg_len / step) + 1);

        // Cycles convention per channel (sigma_total = density * sample):
        //   sigma_s,c = sigma_total * color_c          (scatter)
        //   sigma_a,c = sigma_total * (1 - color_c)    (intrinsic absorption from Color)
        //              + sigma_total * absorption_color_c   (extra absorption tint)
        //   sigma_t,c = sigma_s,c + sigma_a,c = sigma_total * (1 + absorption_color_c)
        //
        // So extinction is monochromatic when absorption_color = 0, regardless
        // of Color. Color only steers the scattering albedo (handled at the
        // scatter event in the integrator) — it does NOT change how fast light
        // attenuates through the medium.
        const float ext_r = 1.0f + vol.absorption_color.r;
        const float ext_g = 1.0f + vol.absorption_color.g;
        const float ext_b = 1.0f + vol.absorption_color.b;
        for (int i = 0; i < n_steps; ++i) {
            const float t = t_in + (static_cast<float>(i) + 0.5f) * step;
            if (t > t_out) break;
            const float wx = ro.x + rd.x * t;
            const float wy = ro.y + rd.y * t;
            const float wz = ro.z + rd.z * t;
            const float d  = sample_grid_trilinear(vol, wx, wy, wz);
            if (d <= 0.0f) continue;
            const float base = vol.density * d * step;
            tau_r += base * ext_r;
            tau_g += base * ext_g;
            tau_b += base * ext_b;
        }
    }

    return render::make_attenuation_rgb(
        std::exp(-tau_r), std::exp(-tau_g), std::exp(-tau_b));
}

// =============================================================================
// Delta-tracking distance sampler for `traceVolumeMIS`.
// =============================================================================
struct VolumeScatterEvent {
    bool                       happened = false;
    render::Length             t        = render::metres(0.0f);
    const VolumeProperties*    vol      = nullptr;
};

inline VolumeScatterEvent sample_volume_distance(
    const Scene& scene, const Ray& ray, render::Length t_max) {

    VolumeScatterEvent out;
    const float t_max_f = t_max.numerical_value_in(render::si::metre);

    const auto ro = render::displacement_from_origin(ray.origin)
                        .numerical_value_in(render::si::metre);
    const auto rd = ray.direction.vec();

    float best_t = t_max_f;

    for (const auto& mesh : scene.meshes) {
        const auto& vol = mesh.material.volume;
        if (!vol.present() || !vol.has_grid()) continue;
        if (vol.density <= 0.0f || vol.grid_max <= 0.0f) continue;

        float t_in = 0.0f, t_out = 0.0f;
        if (!ray_aabb_segment(ray, t_max_f,
                              vol.grid_world_min, vol.grid_world_max,
                              t_in, t_out)) {
            continue;
        }
        if (t_in >= best_t) continue;
        const float seg_end = std::min(t_out, best_t);
        if (seg_end <= t_in) continue;

        // Majorant must bound per-channel sigma_t = density*sample*(1+abs_c).
        // Color does NOT enter the extinction in Cycles' convention (it only
        // steers the scattering albedo), so the majorant is independent of
        // color. For abs=0 this collapses to monochromatic majorant.
        const float ext_max = std::max({
            1.0f + vol.absorption_color.r,
            1.0f + vol.absorption_color.g,
            1.0f + vol.absorption_color.b});
        const float sigma_max = vol.density * vol.grid_max * ext_max;
        if (sigma_max <= 0.0f) continue;

        float t = t_in;
        bool   scattered = false;
        for (int iter = 0; iter < 4096; ++iter) {
            const float u_step = std::max(1e-12f, 1.0f - randf());
            t += -std::log(u_step) / sigma_max;
            if (t >= seg_end) break;

            const float wx = ro.x + rd.x * t;
            const float wy = ro.y + rd.y * t;
            const float wz = ro.z + rd.z * t;
            const float d  = sample_grid_trilinear(vol, wx, wy, wz);
            // Use a representative scalar sigma_t for the accept/null decision.
            // Take the max channel so we never miss a real event in any channel;
            // the per-channel chromatic weighting is applied at the scatter site.
            const float sigma_t = vol.density * d * ext_max;

            if (randf() * sigma_max < sigma_t) {
                scattered = true;
                break;
            }
        }

        if (scattered && t < best_t) {
            best_t   = t;
            out.happened = true;
            out.t        = render::metres(t);
            out.vol      = &vol;
        }
    }

    return out;
}

// Trilinear sample of a dense float grid stored x-fastest.
// world_pos must be inside [grid_world_min, grid_world_max].
inline float sample_grid_trilinear(const VolumeProperties& vol,
                                    float wx, float wy, float wz) {
    const float u = (wx - vol.grid_world_min[0])
                  / std::max(1e-12f, vol.grid_world_max[0] - vol.grid_world_min[0]);
    const float v = (wy - vol.grid_world_min[1])
                  / std::max(1e-12f, vol.grid_world_max[1] - vol.grid_world_min[1]);
    const float w = (wz - vol.grid_world_min[2])
                  / std::max(1e-12f, vol.grid_world_max[2] - vol.grid_world_min[2]);
    if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f || w < 0.0f || w > 1.0f) {
        return 0.0f;
    }
    const int nx = vol.grid_dims[0];
    const int ny = vol.grid_dims[1];
    const int nz = vol.grid_dims[2];
    const float fx = u * static_cast<float>(nx - 1);
    const float fy = v * static_cast<float>(ny - 1);
    const float fz = w * static_cast<float>(nz - 1);
    const int ix = std::clamp(static_cast<int>(fx), 0, nx - 2);
    const int iy = std::clamp(static_cast<int>(fy), 0, ny - 2);
    const int iz = std::clamp(static_cast<int>(fz), 0, nz - 2);
    const float tx = fx - static_cast<float>(ix);
    const float ty = fy - static_cast<float>(iy);
    const float tz = fz - static_cast<float>(iz);

    auto idx = [nx, ny](int i, int j, int k) {
        return static_cast<size_t>(i)
             + static_cast<size_t>(nx) * static_cast<size_t>(j)
             + static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(k);
    };
    const float c000 = vol.grid_density[idx(ix,   iy,   iz  )];
    const float c100 = vol.grid_density[idx(ix+1, iy,   iz  )];
    const float c010 = vol.grid_density[idx(ix,   iy+1, iz  )];
    const float c110 = vol.grid_density[idx(ix+1, iy+1, iz  )];
    const float c001 = vol.grid_density[idx(ix,   iy,   iz+1)];
    const float c101 = vol.grid_density[idx(ix+1, iy,   iz+1)];
    const float c011 = vol.grid_density[idx(ix,   iy+1, iz+1)];
    const float c111 = vol.grid_density[idx(ix+1, iy+1, iz+1)];

    const float c00 = c000 * (1.0f - tx) + c100 * tx;
    const float c10 = c010 * (1.0f - tx) + c110 * tx;
    const float c01 = c001 * (1.0f - tx) + c101 * tx;
    const float c11 = c011 * (1.0f - tx) + c111 * tx;
    const float c0  = c00  * (1.0f - ty) + c10  * ty;
    const float c1  = c01  * (1.0f - ty) + c11  * ty;
    return c0 * (1.0f - tz) + c1 * tz;
}
