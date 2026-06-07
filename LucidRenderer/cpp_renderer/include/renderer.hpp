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

// Debug mode: thickness map of volume-shaded meshes (Beer-Lambert lite).
// Hit the front face of a volume; continue the ray a hair past the entry point
// and find the next intersection (assumed to be the back face of the same
// convex volume — non-convex / overlapping volumes are out of scope for the
// debug pass). Output `(1 - exp(-density * thickness)) * volume.color`, which
// is what the surface visually absorbs / scatters relative to a black
// background. Non-volume meshes return black.
inline render::AttenuationRGB traceVolume(const Scene &scene, const Ray &ray, bool useAABB = true) {
    Hit hit = intersectScene(scene, ray, render::metres(0.0f), geometry::RAY_T_MAX_TYPED, useAABB);
    if (!hit.hit || !hit.material->volume.present()) {
        return render::make_attenuation_rgb(0.0f, 0.0f, 0.0f);
    }

    // Step a small amount past the entry point and trace again to find the
    // exit. The new ray's `t` is the thickness through the medium.
    constexpr float kEntryOffset = 1e-4f;
    auto entry_pt = ray.at(hit.t);
    Ray inside_ray(entry_pt + ray.direction * render::metres(kEntryOffset),
                    ray.direction);
    Hit exit_hit = intersectScene(scene, inside_ray,
                                   render::metres(0.0f),
                                   geometry::RAY_T_MAX_TYPED, useAABB);
    if (!exit_hit.hit) {
        // Ray escaped to infinity without exiting (e.g. open-mesh volume).
        // Treat thickness as one "unit length" so the debug still shows
        // something rather than going pitch black on edge cases.
        const auto& vol = hit.material->volume;
        const float t = 1.0f - std::exp(-vol.density);
        return render::make_attenuation_rgb(vol.color.r * t,
                                             vol.color.g * t,
                                             vol.color.b * t);
    }

    const float thickness = exit_hit.t.numerical_value_in(render::si::metre);
    const auto& vol = hit.material->volume;
    const float t = 1.0f - std::exp(-vol.density * thickness);
    return render::make_attenuation_rgb(vol.color.r * t,
                                         vol.color.g * t,
                                         vol.color.b * t);
}
