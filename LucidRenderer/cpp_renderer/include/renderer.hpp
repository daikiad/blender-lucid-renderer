/**
 * renderer.hpp - Core Path Tracing Renderer (Compatibility Header)
 * =================================================================
 * 
 * This file provides backward compatibility by including all the
 * modular components of the renderer.
 * 
 * New code should include specific headers:
 * - math/random.hpp
 * - core/ray.hpp, core/material.hpp, core/scene.hpp
 * - geometry/intersection.hpp, geometry/bvh.hpp
 * - bsdf/fresnel.hpp, bsdf/ggx.hpp, bsdf/bsdf.hpp
 * - light/light.hpp, light/scene_lights.hpp
 * - integrator/path_tracer.hpp
 * - units/render_units.hpp (for type-safe physical quantities)
 * 
 * Physical Units Reference:
 * - Position, distance: [m] (meters)
 * - Area: [m²] (square meters)
 * - Angle: [rad] (radians)
 * - Solid angle: [sr] (steradians)
 * - Power: [W] (watts)
 * - Radiance: [W/(sr·m²)] (watts per steradian per square meter)
 * - Irradiance: [W/m²] (watts per square meter)
 * - BSDF: [1/sr] (inverse steradians)
 * - PDF: [1/sr] (inverse steradians)
 */

#pragma once

// Physical units (include first for type definitions)
#include "units/render_units.hpp"

// Math utilities
#include "math/random.hpp"

// Core structures
#include "core/ray.hpp"
#include "core/material.hpp"
#include "core/scene.hpp"

// Geometry and intersection
#include "geometry/intersection.hpp"
#include "geometry/bvh.hpp"

// Node evaluator forward declarations (implemented in node_evaluator.cpp)
// Note: evaluateNode returns AttenuationRGB for color values from material nodes
render::AttenuationRGB evaluateNode(const NodeTree &tree, const std::string &nodeName, const std::string &socketName, const render::Vec2f &uv);
render::AttenuationRGB getAlbedoFromNodeTree(const NodeTree &tree, const render::Vec2f &uv);
render::RGB3f getEmissionFromNodeTree(const NodeTree &tree, const render::Vec2f &uv);
float getTransmissionFromNodeTree(const NodeTree &tree, const render::Vec2f &uv);
float getIORFromNodeTree(const NodeTree &tree, const render::Vec2f &uv);
float getMetallicFromNodeTree(const NodeTree &tree, const render::Vec2f &uv);
float getRoughnessFromNodeTree(const NodeTree &tree, const render::Vec2f &uv);

// ========== Debug Rendering Functions ==========

// Debug mode: return normal as color (visualized as RGB)
inline render::AttenuationRGB traceNormal(const Scene &scene, const Ray &ray, bool useAABB = true) {
    Hit hit = intersectScene(scene, ray, render::metres(0.0f), geometry::RAY_T_MAX_TYPED, useAABB);
    if (hit.hit) {
        return render::make_attenuation_rgb(hit.normal.vec().x, hit.normal.vec().y, hit.normal.vec().z);
    }
    return render::make_attenuation_rgb(0, 0, 0);
}

// Debug mode: return albedo from material
inline render::AttenuationRGB traceAlbedo(const Scene &scene, const Ray &ray, bool useAABB = true) {
    Hit hit = intersectScene(scene, ray, render::metres(0.0f), geometry::RAY_T_MAX_TYPED, useAABB);
    if (hit.hit) {
        if (hit.material->useNodes && hit.material->nodeTree.valid) {
            return getAlbedoFromNodeTree(hit.material->nodeTree, hit.uv);
        }
        return hit.material->albedo;
    }
    return render::make_attenuation_rgb(0, 0, 0);
}

// Debug mode: return emission from material (converted from radiance for display)
// Returns AttenuationRGB for use in the debug visualization buffer (untyped pixel grid).
inline render::AttenuationRGB traceEmission(const Scene &scene, const Ray &ray, bool useAABB = true) {
    Hit hit = intersectScene(scene, ray, render::metres(0.0f), geometry::RAY_T_MAX_TYPED, useAABB);
    if (hit.hit) {
        if (hit.material->useNodes && hit.material->nodeTree.valid) {
            // Node-evaluated emission color comes back as untagged RGB3f
            return render::as_attenuation(getEmissionFromNodeTree(hit.material->nodeTree, hit.uv));
        }
        // Camera-sensitivity result is PixelRGB; re-tag for debug-buffer use
        return render::as_attenuation(render::to_rgb3f(
            render::apply_camera_sensitivity(hit.material->emission, render::kDefaultCameraSensitivity)));
    }
    return render::make_attenuation_rgb(0, 0, 0);
}

// Volume sampling helpers (slab clip, voxel trilinear sample, Beer-Lambert
// transmittance variants, delta-tracking distance sampler) live in a
// dedicated header so both this compatibility shim and the path-tracer
// integrators can use them without including all of `renderer.hpp`.
#include "volume/transmittance.hpp"

// Debug mode: thickness map of volume-shaded meshes.
//
// Homogeneous (no grid): hit front face, trace from inside to find back face,
// output (1 - exp(-density * thickness)) * color.
//
// Heterogeneous (smoke grid attached): hit front face, then ray-march through
// the volume's bbox at fixed step size, accumulating density * step at each
// sample. Output (1 - exp(-integrated_density)) * color.
inline render::AttenuationRGB traceVolume(const Scene &scene, const Ray &ray, bool useAABB = true) {
    Hit hit = intersectScene(scene, ray, render::metres(0.0f), geometry::RAY_T_MAX_TYPED, useAABB);
    if (!hit.hit || !hit.material->volume.present()) {
        return render::make_attenuation_rgb(0.0f, 0.0f, 0.0f);
    }
    const auto& vol = hit.material->volume;

    // ---- Heterogeneous ray-march ----
    if (vol.has_grid()) {
        constexpr float kEntryOffset = 1e-4f;
        const auto entry_pt = ray.at(hit.t);
        Ray inside_ray(entry_pt + ray.direction * render::metres(kEntryOffset),
                        ray.direction);
        Hit exit_hit = intersectScene(scene, inside_ray,
                                       render::metres(0.0f),
                                       geometry::RAY_T_MAX_TYPED, useAABB);
        const float t_inside = exit_hit.hit
            ? exit_hit.t.numerical_value_in(render::si::metre)
            : 0.0f;
        if (t_inside <= 0.0f) {
            return render::make_attenuation_rgb(0.0f, 0.0f, 0.0f);
        }

        // Step size: aim for ~2 samples per voxel along the longest axis.
        const float extent = std::max({
            vol.grid_world_max[0] - vol.grid_world_min[0],
            vol.grid_world_max[1] - vol.grid_world_min[1],
            vol.grid_world_max[2] - vol.grid_world_min[2],
        });
        const int max_dim = std::max({
            vol.grid_dims[0], vol.grid_dims[1], vol.grid_dims[2]});
        const float voxel_size = extent / std::max(1, max_dim);
        const float step = std::max(1e-4f, voxel_size * 0.5f);
        const int n_steps = std::min(2048, static_cast<int>(t_inside / step) + 1);

        const auto rd = inside_ray.direction.vec();
        const auto ro = render::displacement_from_origin(
                            inside_ray.origin).numerical_value_in(render::si::metre);

        float integrated = 0.0f;
        float max_sampled = 0.0f;
        int   nonzero_steps = 0;
        for (int i = 0; i < n_steps; ++i) {
            const float t = (static_cast<float>(i) + 0.5f) * step;
            if (t > t_inside) break;
            const float wx = ro.x + rd.x * t;
            const float wy = ro.y + rd.y * t;
            const float wz = ro.z + rd.z * t;
            const float d = sample_grid_trilinear(vol, wx, wy, wz);
            integrated += d * step;
            if (d > max_sampled) max_sampled = d;
            if (d > 0.0f) ++nonzero_steps;
        }

        (void)nonzero_steps;
        // Debug viz: show the max density sampled along the ray, modulated by
        // a Beer-Lambert-style opacity from the integrated extinction. Using
        // the max is a "did the ray hit any smoke?" indicator that doesn't
        // require enough thickness for the integral to register; the opacity
        // term smoothly turns up the brightness where the ray genuinely
        // travels through dense regions. Real Beer-Lambert without the max
        // sentinel goes pitch black on thin / partially-occluded volumes
        // because the next-hit (other geometry inside the domain) shrinks
        // t_inside to near zero. Phase 2 (real PT integration) replaces
        // this debug formula with proper participating-medium transport.
        const float opacity = 1.0f - std::exp(-vol.density * integrated * 10.0f);
        const float vis = std::max(max_sampled, opacity);
        return render::make_attenuation_rgb(vol.color.r * vis,
                                             vol.color.g * vis,
                                             vol.color.b * vis);
    }

    // ---- Homogeneous (Phase 1 behaviour) ----
    constexpr float kEntryOffset = 1e-4f;
    auto entry_pt = ray.at(hit.t);
    Ray inside_ray(entry_pt + ray.direction * render::metres(kEntryOffset),
                    ray.direction);
    Hit exit_hit = intersectScene(scene, inside_ray,
                                   render::metres(0.0f),
                                   geometry::RAY_T_MAX_TYPED, useAABB);
    if (!exit_hit.hit) {
        const float t = 1.0f - std::exp(-vol.density);
        return render::make_attenuation_rgb(vol.color.r * t,
                                             vol.color.g * t,
                                             vol.color.b * t);
    }
    const float thickness = exit_hit.t.numerical_value_in(render::si::metre);
    const float t = 1.0f - std::exp(-vol.density * thickness);
    return render::make_attenuation_rgb(vol.color.r * t,
                                         vol.color.g * t,
                                         vol.color.b * t);
}
