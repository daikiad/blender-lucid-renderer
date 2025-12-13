/**
 * bvh.hpp - BVH Traversal and Scene Intersection
 * ===============================================
 * 
 * High-level intersection functions using BVH acceleration:
 * - BVH traversal for single mesh
 * - Full scene intersection
 * - Shadow ray testing
 * - Environment color lookup
 */

#pragma once
#include "../units/render_units.hpp"
#include "../core/ray.hpp"
#include "../core/material.hpp"
#include "../core/scene.hpp"
#include "intersection.hpp"

namespace geometry {

// ========== BVH Traversal ==========

/**
 * Traverse BVH and find closest intersection with mesh triangles (unit-typed)
 * Uses stack-based traversal for performance (non-recursive)
 * @param mesh      Mesh to test against
 * @param ray       Ray to trace
 * @param invDir    Precomputed inverse direction (use computeInverseDirection)
 * @param result    Hit result (updated if closer hit found)
 * @param meshIdx   Index of this mesh in scene (for result tracking)
 * @param tMin      Minimum valid t value [m]
 * @param tMax      Maximum valid t value [m] (use result.t to find closest)
 */
inline void traverseBVH(const Mesh& mesh, const Ray& ray, const render::Vec3f& invDir, 
                        Hit& result, int meshIdx = -1,
                        render::Length tMin = RAY_T_MIN_TYPED, 
                        render::Length tMax = RAY_T_MAX_TYPED) {
    // Use the smaller of tMax and current result.t
    render::Length currentTMax = std::min(tMax, result.t);
    
    if (mesh.bvh.nodes.empty()) {
        // Fallback: linear scan if no BVH
        for (size_t ti = 0; ti < mesh.triangles.size(); ++ti) {
            const Triangle& tri = mesh.triangles[ti];
            const auto& a = mesh.vertices[tri.i0];
            const auto& b = mesh.vertices[tri.i1];
            const auto& c = mesh.vertices[tri.i2];
            render::BarycentricCoord2 baryUV;
            auto hitResult = intersectTriangle(ray, a, b, c, baryUV, tMin, currentTMax);
            if (hitResult) {
                render::Length t = *hitResult;
                currentTMax = t;  // Update for subsequent tests
                render::Position point = ray.at(t);
                render::Normal normal;
                float u = render::bary_u(baryUV);
                float v = render::bary_v(baryUV);
                float w = render::bary_w(baryUV);
                if (tri.smooth) {
                    // Interpolate vertex normals (access underlying Vec3f)
                    render::Vec3f nInterp = tri.n0.vec() * w + tri.n1.vec() * u + tri.n2.vec() * v;
                    normal = render::make_normal_or_default(nInterp, tri.faceNormal);
                } else {
                    normal = tri.faceNormal;
                }
                result.setFromTyped(t, point, normal);
                result.hit = true;
                if (tri.hasUV) {
                    result.uv = tri.uv0 * w + tri.uv1 * u + tri.uv2 * v;
                    result.hasUV = true;
                } else {
                    result.hasUV = false;
                }
                result.material = &mesh.material;
                result.meshIdx = meshIdx;
                result.triIdx = (int)ti;
            }
        }
        return;
    }
    
    // Stack-based BVH traversal (non-recursive for speed)
    // Fixed-size stack is faster than vector (no heap allocation)
    // Max depth 64 supports up to 2^64 triangles theoretically
    // Typical BVH depth is ~20-30 even for millions of triangles
    int stack[64];
    int stackPtr = 0;
    stack[stackPtr++] = 0;  // Start with root node
    
    while (stackPtr > 0) {
        int nodeIdx = stack[--stackPtr];
        const BVHNode& node = mesh.bvh.nodes[nodeIdx];
        
        // Test ray against node bounds
        if (!intersectAABB(ray, invDir, node.bmin, node.bmax)) continue;
        
        if (node.isLeaf()) {
            // Test triangles in this leaf
            for (int i = 0; i < node.triCount; ++i) {
                int triIdx = mesh.bvh.triIndices[node.triStart + i];
                const Triangle& tri = mesh.triangles[triIdx];
                const auto& a = mesh.vertices[tri.i0];
                const auto& b = mesh.vertices[tri.i1];
                const auto& c = mesh.vertices[tri.i2];
                render::BarycentricCoord2 baryUV;
                auto hitResult = intersectTriangle(ray, a, b, c, baryUV, tMin, currentTMax);
                if (hitResult) {
                    render::Length t = *hitResult;
                    currentTMax = t;  // Update for subsequent tests
                    render::Position point = ray.at(t);
                    render::Normal normal;
                    float u = render::bary_u(baryUV);
                    float v = render::bary_v(baryUV);
                    float w = render::bary_w(baryUV);
                    if (tri.smooth) {
                        // Interpolate vertex normals (access underlying Vec3f)
                        render::Vec3f nInterp = tri.n0.vec() * w + tri.n1.vec() * u + tri.n2.vec() * v;
                        normal = render::make_normal_or_default(nInterp, tri.faceNormal);
                    } else {
                        normal = tri.faceNormal;
                    }
                    result.setFromTyped(t, point, normal);
                    result.hit = true;
                    if (tri.hasUV) {
                        result.uv = tri.uv0 * w + tri.uv1 * u + tri.uv2 * v;
                        result.hasUV = true;
                    } else {
                        result.hasUV = false;
                    }
                    result.material = &mesh.material;
                    result.meshIdx = meshIdx;
                    result.triIdx = triIdx;
                }
            }
        } else {
            // Push children onto stack (check bounds to prevent overflow)
            if (stackPtr < 62) {
                if (node.right >= 0) stack[stackPtr++] = node.right;
                if (node.left >= 0) stack[stackPtr++] = node.left;
            }
        }
    }
}

// ========== Scene Intersection ==========

/**
 * Find closest intersection with entire scene (unit-typed)
 * @param scene     Scene to test against
 * @param ray       Ray to trace
 * @param tMin      Minimum valid t value [m] (default: RAY_T_MIN_TYPED)
 * @param tMax      Maximum valid t value [m] (default: RAY_T_MAX_TYPED)
 * @param useAABB   Whether to use mesh-level AABB culling (default: true)
 * @return Hit result with closest intersection
 */
inline Hit intersectScene(const Scene& scene, const Ray& ray, 
                          render::Length tMin = RAY_T_MIN_TYPED, 
                          render::Length tMax = RAY_T_MAX_TYPED,
                          bool useAABB = true) {
    Hit result;
    result.setFromTyped(tMax, render::make_position(0.0f, 0.0f, 0.0f), 
                        *render::make_normal(0.0f, 1.0f, 0.0f));
    result.hit = false;
    
    // Precompute inverse direction for faster AABB tests
    render::Vec3f invDir = computeInverseDirection(ray.direction);
    
    for (size_t mi = 0; mi < scene.meshes.size(); ++mi) {
        const Mesh& m = scene.meshes[mi];
        // First check mesh-level AABB
        if (useAABB && !intersectAABB(ray, invDir, m.bmin, m.bmax)) continue;
        
        // Use BVH for triangle intersection (pass typed Length directly)
        traverseBVH(m, ray, invDir, result, (int)mi, tMin, result.t);
    }
    
    return result;
}

// ========== Environment Color ==========

/**
 * Get environment/background color for ray
 * @param ray   Ray direction (for future HDRI support)
 * @param env   Environment settings
 * @return Environment radiance (ColorRGB)
 */
inline render::ColorRGB getEnvironmentColor(const Ray& ray, const Environment& env) {
    return env.color * env.strength;
}

// ========== Shadow Testing ==========

/**
 * Test if a point is in shadow from a light (unit-typed)
 * @param scene     Scene to test against
 * @param point     Surface point
 * @param lightDir  Direction toward light
 * @param lightDist Distance to light [m]
 * @return true if occluded
 */
inline bool isInShadow(const Scene& scene, const render::Position& point, 
                       const render::Direction& lightDir, render::Length lightDist) {
    Ray shadowRay(point, lightDir);
    Hit hit = intersectScene(scene, shadowRay, RAY_T_MIN_TYPED, lightDist);
    return hit.hit;
}

} // namespace geometry

// ========== Global Function Aliases (Backward Compatibility) ==========
// These allow existing code to work without namespace qualification

using geometry::intersectScene;
using geometry::getEnvironmentColor;
using geometry::isInShadow;
