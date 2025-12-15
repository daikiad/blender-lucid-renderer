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
render::AttenuationRGB getEmissionFromNodeTree(const NodeTree &tree, const render::Vec2f &uv);
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
inline render::AttenuationRGB traceEmission(const Scene &scene, const Ray &ray, bool useAABB = true) {
    Hit hit = intersectScene(scene, ray, render::metres(0.0f), geometry::RAY_T_MAX_TYPED, useAABB);
    if (hit.hit) {
        if (hit.material->useNodes && hit.material->nodeTree.valid) {
            return getEmissionFromNodeTree(hit.material->nodeTree, hit.uv);
        }
        return render::apply_camera_sensitivity(hit.material->emission, render::kDefaultCameraSensitivity);
    }
    return render::make_attenuation_rgb(0, 0, 0);
}
