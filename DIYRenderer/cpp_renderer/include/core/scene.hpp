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
 * Physical Units (using render:: namespace):
 * - All vertex positions: Position [m]
 * - Bounding box coordinates: Position [m]
 * - Normals: Normal (normalized)
 * - UV coordinates: render::Vec2f (dimensionless [0,1])
 * - Colors: ColorRGB (dimensionless [0,1])
 */

#pragma once
#include "../units/render_units.hpp"
#include "ray.hpp"
#include "material.hpp"
#include <vector>
#include <algorithm>

// Forward declaration for Light (defined in light/light.hpp)
struct Light;

// ========== Triangle Structure ==========

struct Triangle { 
    int i0, i1, i2;              // Vertex indices
    render::Normal faceNormal;   // Face normal (flat shading)
    render::Normal n0, n1, n2;   // Vertex normals (smooth shading)
    render::Vec2f uv0, uv1, uv2; // UV coordinates per vertex
    render::Position centroid;   // For BVH construction
    bool smooth;                 // Use smooth shading?
    bool hasUV;                  // Does triangle have UV coordinates?
    
    Triangle() : i0(0), i1(0), i2(0), 
                 centroid(render::make_position(0.0f, 0.0f, 0.0f)),
                 smooth(false), hasUV(false) {}
};

// ========== BVH (Bounding Volume Hierarchy) ==========

struct BVHNode {
    render::Position bmin, bmax;  // Bounding box of this node
    int left, right;              // Child indices (-1 for leaf)
    int triStart, triCount;       // Triangle range for leaf nodes
    
    BVHNode() : bmin(render::make_position(0.0f, 0.0f, 0.0f)),
                bmax(render::make_position(0.0f, 0.0f, 0.0f)),
                left(-1), right(-1), triStart(0), triCount(0) {}
    
    bool isLeaf() const { return triCount > 0; }
};

struct BVH {
    std::vector<BVHNode> nodes;
    std::vector<int> triIndices;  // Reordered triangle indices
    
    // Build BVH from mesh triangles
    void build(const std::vector<render::Position>& vertices, std::vector<Triangle>& triangles) {
        if(triangles.empty()) return;
        
        // Initialize triangle indices and compute centroids
        triIndices.resize(triangles.size());
        for(size_t i = 0; i < triangles.size(); ++i) {
            triIndices[i] = (int)i;
            Triangle& tri = triangles[i];
            
            // Compute centroid using typed position arithmetic
            // Centroid = v0 + (1/3)*(e01 + e02) = v0 + (v1-v0+v2-v0)/3
            const render::Position& a = vertices[tri.i0];
            const render::Position& b = vertices[tri.i1];
            const render::Position& c = vertices[tri.i2];
            
            // Edge vectors [m] (Displacement - ISQ compliant!)
            render::Displacement ab = b - a;
            render::Displacement ac = c - a;
            
            // Centroid = a + (ab + ac)/3  [Displacement / scalar = Displacement]
            tri.centroid = a + (ab + ac) / 3.0f;
        }
        
        // Reserve space for nodes (2n-1 nodes for n triangles in worst case)
        nodes.reserve(2 * triangles.size());
        
        // Build recursively
        buildRecursive(vertices, triangles, 0, (int)triangles.size());
    }
    
private:
    int buildRecursive(const std::vector<render::Position>& vertices, std::vector<Triangle>& triangles, 
                       int start, int end) {
        int nodeIdx = (int)nodes.size();
        nodes.push_back(BVHNode());
        BVHNode& node = nodes[nodeIdx];
        
        // Compute bounds for this node
        constexpr float inf = 1e30f;
        node.bmin = render::make_position(inf, inf, inf);
        node.bmax = render::make_position(-inf, -inf, -inf);
        
        for(int i = start; i < end; ++i) {
            const Triangle& tri = triangles[triIndices[i]];
            for(int v = 0; v < 3; ++v) {
                const render::Position& vert = vertices[v == 0 ? tri.i0 : (v == 1 ? tri.i1 : tri.i2)];
                node.bmin = render::pos_component_min(node.bmin, vert);
                node.bmax = render::pos_component_max(node.bmax, vert);
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
        // extent = bmax - bmin is a Displacement [m] (ISQ compliant!)
        render::Displacement extent = node.bmax - node.bmin;
        render::Length ex = render::disp_x(extent);
        render::Length ey = render::disp_y(extent);
        render::Length ez = render::disp_z(extent);
        int axis = 0;
        if (ey > ex) axis = 1;
        if (ez > (axis == 0 ? ex : ey)) axis = 2;
        
        // Sort by centroid along split axis using typed comparison
        int mid = (start + end) / 2;
        std::nth_element(triIndices.begin() + start, triIndices.begin() + mid, 
                        triIndices.begin() + end,
                        [&](int a, int b) {
                            return render::pos_component(triangles[a].centroid, axis)
                                 < render::pos_component(triangles[b].centroid, axis);
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
    std::vector<render::Position> vertices;
    std::vector<Triangle> triangles;
    Material material;
    BVH bvh;  // BVH acceleration structure
    // axis-aligned bounding box (for whole mesh)
    render::Position bmin{render::make_position(1e30f, 1e30f, 1e30f)};
    render::Position bmax{render::make_position(-1e30f, -1e30f, -1e30f)};
};

// ========== Environment Settings ==========

/**
 * Environment - World/background settings
 * 
 * The environment provides radiance for rays that miss all geometry.
 * final_radiance = color × strength [W/(sr·m²)]
 */
struct Environment {
    render::AttenuationRGB color{0.05f, 0.05f, 0.05f};  // Background color coefficient [0,1]
    float strength = 1.0f;                               // Emission strength multiplier (dimensionless)
};

// ========== Scene Structure ==========

struct Scene {
    std::vector<Mesh> meshes;
    std::vector<Light> nativeLights;  // Blender native lights (Point, Sun, Spot, Area)
    Environment environment;          // World environment settings
    Camera camera;                    // Camera with exposure settings (defined in ray.hpp)
};

// ========== Mesh Finalization ==========

inline void finalizeMeshBounds(Mesh &m) {
    for(const auto &v : m.vertices) {
        m.bmin = render::pos_component_min(m.bmin, v);
        m.bmax = render::pos_component_max(m.bmax, v);
    }
    
    // Add small epsilon to avoid zero-thickness boxes (fully typed)
    constexpr auto eps = render::metres(0.001f);
    render::Displacement extent = m.bmax - m.bmin;
    
    if (render::disp_x(extent) < eps) {
        m.bmin = m.bmin - render::kAxisX * eps;
        m.bmax = m.bmax + render::kAxisX * eps;
    }
    if (render::disp_y(extent) < eps) {
        m.bmin = m.bmin - render::kAxisY * eps;
        m.bmax = m.bmax + render::kAxisY * eps;
    }
    if (render::disp_z(extent) < eps) {
        m.bmin = m.bmin - render::kAxisZ * eps;
        m.bmax = m.bmax + render::kAxisZ * eps;
    }
    
    for(auto &t : m.triangles) {
        // Compute face normal using typed edge vectors (Displacement - ISQ compliant!)
        render::Displacement e1 = m.vertices[t.i1] - m.vertices[t.i0];  // [m]
        render::Displacement e2 = m.vertices[t.i2] - m.vertices[t.i0];  // [m]
        // Cross product using disp_to_vec helper (encapsulates extraction)
        auto normal_opt = render::make_normal(render::cross(render::disp_to_vec(e1), render::disp_to_vec(e2)));
        if (normal_opt) {
            t.faceNormal = *normal_opt;
        }
    }
    
    // Build BVH for this mesh
    m.bvh.build(m.vertices, m.triangles);
}
