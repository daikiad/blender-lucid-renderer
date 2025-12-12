/**
 * scene.hpp - Scene, Mesh, and BVH Structures
 * ============================================
 * 
 * Core scene structures for ray tracing:
 * - Triangle: Triangle geometry with normals and UVs
 * - BVH: Bounding Volume Hierarchy for acceleration
 * - Mesh: Collection of triangles with material
 * - Environment: World/background settings
 * - Scene: Complete scene with meshes and lights
 * 
 * Physical Units:
 * - All vertex positions: Position3 [m]
 * - Bounding box coordinates: Position3 [m]
 * - Normals: Direction3 (dimensionless, normalized)
 * - UV coordinates: Vec2 (dimensionless [0,1])
 * - Colors: Color3 (dimensionless [0,1])
 */

#pragma once
#include "../math/vec3_unit.hpp"
#include "../math/vec2.hpp"
#include "../units/units.hpp"
#include "ray.hpp"
#include "material.hpp"
#include <vector>
#include <algorithm>

// Forward declaration for Light (defined in light/light.hpp)
struct Light;

// ========== Triangle Structure ==========

struct Triangle { 
    int i0, i1, i2;              // Vertex indices
    diy::Direction3 faceNormal;  // Face normal (flat shading)
    diy::Direction3 n0, n1, n2;  // Vertex normals (smooth shading)
    Vec2 uv0, uv1, uv2;          // UV coordinates per vertex
    diy::Position3 centroid;     // For BVH construction
    bool smooth;                 // Use smooth shading?
    bool hasUV;                  // Does triangle have UV coordinates?
    
    Triangle() : i0(0), i1(0), i2(0), smooth(false), hasUV(false) {}
};

// ========== BVH (Bounding Volume Hierarchy) ==========

struct BVHNode {
    diy::Position3 bmin, bmax;    // Bounding box of this node
    int left, right;              // Child indices (-1 for leaf)
    int triStart, triCount;       // Triangle range for leaf nodes
    
    bool isLeaf() const { return triCount > 0; }
};

struct BVH {
    std::vector<BVHNode> nodes;
    std::vector<int> triIndices;  // Reordered triangle indices
    
    // Build BVH from mesh triangles
    void build(const std::vector<diy::Position3>& vertices, std::vector<Triangle>& triangles) {
        if(triangles.empty()) return;
        
        // Initialize triangle indices and compute centroids
        triIndices.resize(triangles.size());
        for(size_t i = 0; i < triangles.size(); ++i) {
            triIndices[i] = (int)i;
            Triangle& tri = triangles[i];
            const auto& a = vertices[tri.i0];
            const auto& b = vertices[tri.i1];
            const auto& c = vertices[tri.i2];
            tri.centroid = (a + b + c) / 3.0f;
        }
        
        // Reserve space for nodes (2n-1 nodes for n triangles in worst case)
        nodes.reserve(2 * triangles.size());
        
        // Build recursively
        buildRecursive(vertices, triangles, 0, (int)triangles.size());
    }
    
private:
    int buildRecursive(const std::vector<diy::Position3>& vertices, std::vector<Triangle>& triangles, 
                       int start, int end) {
        int nodeIdx = (int)nodes.size();
        nodes.push_back(BVHNode());
        BVHNode& node = nodes[nodeIdx];
        
        // Compute bounds for this node
        constexpr float inf = 1e30f;
        node.bmin = diy::Position3{inf * mp_units::si::metre, inf * mp_units::si::metre, inf * mp_units::si::metre};
        node.bmax = diy::Position3{-inf * mp_units::si::metre, -inf * mp_units::si::metre, -inf * mp_units::si::metre};
        
        for(int i = start; i < end; ++i) {
            const Triangle& tri = triangles[triIndices[i]];
            for(int v = 0; v < 3; ++v) {
                const auto& vert = vertices[v == 0 ? tri.i0 : (v == 1 ? tri.i1 : tri.i2)];
                node.bmin = diy::Position3{
                    std::min(node.bmin[0], vert[0]),
                    std::min(node.bmin[1], vert[1]),
                    std::min(node.bmin[2], vert[2])
                };
                node.bmax = diy::Position3{
                    std::max(node.bmax[0], vert[0]),
                    std::max(node.bmax[1], vert[1]),
                    std::max(node.bmax[2], vert[2])
                };
            }
        }
        
        int count = end - start;
        
        // Leaf node if few triangles
        if(count <= 4) {
            node.triStart = start;
            node.triCount = count;
            node.left = node.right = -1;
            return nodeIdx;
        }
        
        // Find best split axis (longest extent)
        auto extent = node.bmax - node.bmin;  // Direction3 (displacement)
        int axis = 0;
        if(extent[1] > extent[0]) axis = 1;
        if(extent[2] > (axis == 0 ? extent[0] : extent[1])) axis = 2;
        
        // Sort by centroid along split axis
        int mid = (start + end) / 2;
        std::nth_element(triIndices.begin() + start, triIndices.begin() + mid, 
                        triIndices.begin() + end,
                        [&](int a, int b) {
                            return triangles[a].centroid[axis] < triangles[b].centroid[axis];
                        });
        
        // Build children
        node.triStart = 0;
        node.triCount = 0;  // Not a leaf
        node.left = buildRecursive(vertices, triangles, start, mid);
        node.right = buildRecursive(vertices, triangles, mid, end);
        
        return nodeIdx;
    }
};

// ========== Mesh Structure ==========

struct Mesh {
    std::vector<diy::Position3> vertices;
    std::vector<Triangle> triangles;
    Material material;
    BVH bvh;  // BVH acceleration structure
    // axis-aligned bounding box (for whole mesh)
    diy::Position3 bmin{1e30f * mp_units::si::metre, 1e30f * mp_units::si::metre, 1e30f * mp_units::si::metre};
    diy::Position3 bmax{-1e30f * mp_units::si::metre, -1e30f * mp_units::si::metre, -1e30f * mp_units::si::metre};
};

// ========== Environment Settings ==========

/**
 * Environment - World/background settings
 * 
 * The environment provides radiance for rays that miss all geometry.
 * final_radiance = color × strength [W/(sr·m²)]
 */
struct Environment {
    diy::Color3 color = diy::Color3{0.05f, 0.05f, 0.05f};  // Background color (RGB, each [0,∞])
    float strength = 1.0f;                                  // Emission strength multiplier (dimensionless)
};

// ========== Scene Structure ==========

struct Scene {
    std::vector<Mesh> meshes;
    std::vector<Light> nativeLights;  // Blender native lights (Point, Sun, Spot, Area)
    Environment environment;          // World environment settings
};

// ========== Mesh Finalization ==========

inline void finalizeMeshBounds(Mesh &m) {
    for(const auto &v : m.vertices) {
        m.bmin = diy::Position3{
            std::min(m.bmin[0], v[0]),
            std::min(m.bmin[1], v[1]),
            std::min(m.bmin[2], v[2])
        };
        m.bmax = diy::Position3{
            std::max(m.bmax[0], v[0]),
            std::max(m.bmax[1], v[1]),
            std::max(m.bmax[2], v[2])
        };
    }
    // Add small epsilon to avoid zero-thickness boxes
    const auto eps = 0.001f * mp_units::si::metre;
    if(m.bmax[0] - m.bmin[0] < eps) { m.bmin = diy::Position3{m.bmin[0] - eps, m.bmin[1], m.bmin[2]}; m.bmax = diy::Position3{m.bmax[0] + eps, m.bmax[1], m.bmax[2]}; }
    if(m.bmax[1] - m.bmin[1] < eps) { m.bmin = diy::Position3{m.bmin[0], m.bmin[1] - eps, m.bmin[2]}; m.bmax = diy::Position3{m.bmax[0], m.bmax[1] + eps, m.bmax[2]}; }
    if(m.bmax[2] - m.bmin[2] < eps) { m.bmin = diy::Position3{m.bmin[0], m.bmin[1], m.bmin[2] - eps}; m.bmax = diy::Position3{m.bmax[0], m.bmax[1], m.bmax[2] + eps}; }
    
    for(auto &t : m.triangles) {
        const auto &a = m.vertices[t.i0];
        const auto &b = m.vertices[t.i1];
        const auto &c = m.vertices[t.i2];
        t.faceNormal = diy::normalize(diy::cross(b - a, c - a));
    }
    
    // Build BVH for this mesh
    m.bvh.build(m.vertices, m.triangles);
}
