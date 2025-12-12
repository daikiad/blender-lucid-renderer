/**
 * renderer.hpp - Core Path Tracing Renderer (Compatibility Header)
 * =================================================================
 * 
 * This file provides backward compatibility by including all the
 * modular components of the renderer.
 * 
 * New code should include specific headers:
 * - math/vec3_unit.hpp, math/random.hpp
 * - core/ray.hpp, core/material.hpp, core/scene.hpp
 * - geometry/intersection.hpp, geometry/bvh.hpp
 * - bsdf/fresnel.hpp, bsdf/ggx.hpp, bsdf/bsdf.hpp
 * - light/light.hpp, light/scene_lights.hpp
 * - integrator/path_tracer.hpp
 * - units/units.hpp (for type-safe physical quantities)
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
#include "units/units.hpp"

// Math utilities
#include "math/vec3_unit.hpp"
#include "math/vec2.hpp"
#include "math/random.hpp"

// Core structures
#include "core/ray.hpp"
#include "core/material.hpp"
#include "core/scene.hpp"

// Geometry and intersection
#include "geometry/intersection.hpp"
#include "geometry/bvh.hpp"

// Node evaluator forward declarations (implemented in node_evaluator.cpp)
diy::Vec3U<mp_units::one> evaluateNode(const NodeTree &tree, const std::string &nodeName, const std::string &socketName, const Vec2 &uv);
diy::Vec3U<mp_units::one> getAlbedoFromNodeTree(const NodeTree &tree, const Vec2 &uv);
diy::Vec3U<mp_units::one> getEmissionFromNodeTree(const NodeTree &tree, const Vec2 &uv);
float getTransmissionFromNodeTree(const NodeTree &tree, const Vec2 &uv);
float getIORFromNodeTree(const NodeTree &tree, const Vec2 &uv);
float getMetallicFromNodeTree(const NodeTree &tree, const Vec2 &uv);
float getRoughnessFromNodeTree(const NodeTree &tree, const Vec2 &uv);

// ========== Debug Rendering Functions ==========

// Debug mode: return normal as color
inline diy::Vec3U<mp_units::one> traceNormal(const Scene &scene, const Ray &ray, bool useAABB = true) {
    Hit hit = intersectScene(scene, ray, 0.0f, geometry::RAY_T_MAX, useAABB);
    if (hit.hit) {
        return diy::Vec3U<mp_units::one>(hit.normal.x_raw(), hit.normal.y_raw(), hit.normal.z_raw());
    }
    return diy::Vec3U<mp_units::one>(0, 0, 0);
}

// Debug mode: return albedo from material
inline diy::Vec3U<mp_units::one> traceAlbedo(const Scene &scene, const Ray &ray, bool useAABB = true) {
    Hit hit = intersectScene(scene, ray, 0.0f, geometry::RAY_T_MAX, useAABB);
    if (hit.hit) {
        if (hit.material.useNodes && hit.material.nodeTree.valid) {
            return getAlbedoFromNodeTree(hit.material.nodeTree, hit.uv);
        }
        return hit.material.albedo;
    }
    return diy::Vec3U<mp_units::one>(0, 0, 0);
}

// Debug mode: return emission from material
inline diy::Vec3U<mp_units::one> traceEmission(const Scene &scene, const Ray &ray, bool useAABB = true) {
    Hit hit = intersectScene(scene, ray, 0.0f, geometry::RAY_T_MAX, useAABB);
    if (hit.hit) {
        if (hit.material.useNodes && hit.material.nodeTree.valid) {
            return getEmissionFromNodeTree(hit.material.nodeTree, hit.uv);
        }
        return diy::Vec3U<mp_units::one>(
            diy::units::to_radiance(hit.material.emission.x),
            diy::units::to_radiance(hit.material.emission.y),
            diy::units::to_radiance(hit.material.emission.z)
        );
    }
    return diy::Vec3U<mp_units::one>(0, 0, 0);
}
