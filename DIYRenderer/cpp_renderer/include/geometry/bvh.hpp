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
#include "../math/vec3.hpp"
#include "../core/ray.hpp"
#include "../core/material.hpp"
#include "../core/scene.hpp"
#include "intersection.hpp"

namespace geometry {

// ========== BVH Traversal ==========

/**
 * Traverse BVH and find closest intersection with mesh triangles
 * Uses stack-based traversal for performance (non-recursive)
 * @param mesh      Mesh to test against
 * @param ray       Ray to trace
 * @param invDir    Precomputed inverse direction (use computeInverseDirection)
 * @param result    Hit result (updated if closer hit found)
 * @param meshIdx   Index of this mesh in scene (for result tracking)
 */
inline void traverseBVH(const Mesh& mesh, const Ray& ray, const Vec3& invDir, 
                        Hit& result, int meshIdx = -1) {
    if (mesh.bvh.nodes.empty()) {
        // Fallback: linear scan if no BVH
        for (size_t ti = 0; ti < mesh.triangles.size(); ++ti) {
            const Triangle& tri = mesh.triangles[ti];
            const Vec3& a = mesh.vertices[tri.i0];
            const Vec3& b = mesh.vertices[tri.i1];
            const Vec3& c = mesh.vertices[tri.i2];
            float u, v;
            float t = intersectTriangle(ray, a, b, c, u, v);
            if (t > 0.0001f && t < result.t) {
                result.t = t;
                result.hit = true;
                result.point = ray.o + ray.d * t;
                float w = 1.0f - u - v;
                if (tri.smooth) {
                    result.normal = tri.n0 * w + tri.n1 * u + tri.n2 * v;
                    result.normal.normalize();
                } else {
                    result.normal = tri.faceNormal;
                }
                if (tri.hasUV) {
                    result.uv = tri.uv0 * w + tri.uv1 * u + tri.uv2 * v;
                }
                result.material = mesh.material;
                result.meshIdx = meshIdx;
                result.triIdx = (int)ti;
            }
        }
        return;
    }
    
    // Stack-based BVH traversal (non-recursive for speed)
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
                const Vec3& a = mesh.vertices[tri.i0];
                const Vec3& b = mesh.vertices[tri.i1];
                const Vec3& c = mesh.vertices[tri.i2];
                float u, v;
                float t = intersectTriangle(ray, a, b, c, u, v);
                if (t > 0.0001f && t < result.t) {
                    result.t = t;
                    result.hit = true;
                    result.point = ray.o + ray.d * t;
                    float w = 1.0f - u - v;
                    if (tri.smooth) {
                        result.normal = tri.n0 * w + tri.n1 * u + tri.n2 * v;
                        result.normal.normalize();
                    } else {
                        result.normal = tri.faceNormal;
                    }
                    if (tri.hasUV) {
                        result.uv = tri.uv0 * w + tri.uv1 * u + tri.uv2 * v;
                    }
                    result.material = mesh.material;
                    result.meshIdx = meshIdx;
                    result.triIdx = triIdx;
                }
            }
        } else {
            // Push children onto stack
            if (node.right >= 0) stack[stackPtr++] = node.right;
            if (node.left >= 0) stack[stackPtr++] = node.left;
        }
    }
}

// ========== Scene Intersection ==========

/**
 * Find closest intersection with entire scene
 * @param scene     Scene to test against
 * @param ray       Ray to trace
 * @param useAABB   Whether to use mesh-level AABB culling (default: true)
 * @return Hit result with closest intersection
 */
inline Hit intersectScene(const Scene& scene, const Ray& ray, bool useAABB = true) {
    Hit result;
    result.t = 1e30f;
    result.hit = false;
    
    // Precompute inverse direction for faster AABB tests
    Vec3 invDir = computeInverseDirection(ray.d);
    
    for (size_t mi = 0; mi < scene.meshes.size(); ++mi) {
        const Mesh& m = scene.meshes[mi];
        // First check mesh-level AABB
        if (useAABB && !intersectAABB(ray, invDir, m.bmin, m.bmax)) continue;
        
        // Use BVH for triangle intersection
        traverseBVH(m, ray, invDir, result, (int)mi);
    }
    
    return result;
}

// ========== Environment Color ==========

/**
 * Get environment/background color for ray
 * @param ray   Ray direction (for future HDRI support)
 * @param env   Environment settings
 * @return Environment color
 */
inline Vec3 getEnvironmentColor(const Ray& ray, const Environment& env) {
    return env.color * env.strength;
}

/**
 * Legacy version for backward compatibility
 */
inline Vec3 getEnvironmentColor(const Ray& ray) {
    return Vec3(0.0f, 0.0f, 0.0f);
}

// ========== Shadow Testing ==========

/**
 * Test if a point is in shadow from a light
 * @param scene     Scene to test against
 * @param point     Surface point
 * @param lightDir  Direction toward light
 * @param lightDist Distance to light
 * @return true if occluded
 */
inline bool isInShadow(const Scene& scene, const Vec3& point, const Vec3& lightDir, float lightDist) {
    Ray shadowRay{point + lightDir * 0.001f, lightDir};
    Hit hit = intersectScene(scene, shadowRay, true);
    return hit.hit && hit.t < lightDist;
}

} // namespace geometry

// ========== Global Function Aliases (Backward Compatibility) ==========
// These allow existing code to work without namespace qualification

using geometry::intersectScene;
using geometry::getEnvironmentColor;
using geometry::isInShadow;
