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
 */

#pragma once
#include "../math/vec3.hpp"
#include "ray.hpp"
#include "material.hpp"
#include <vector>
#include <algorithm>

// Forward declaration for Light (defined in light/light.hpp)
struct Light;

// ========== Triangle Structure ==========

struct Triangle { 
    int i0, i1, i2;         // Vertex indices
    Vec3 faceNormal;        // Face normal (flat shading)
    Vec3 n0, n1, n2;        // Vertex normals (smooth shading)
    Vec2 uv0, uv1, uv2;     // UV coordinates per vertex
    Vec3 centroid;          // For BVH construction
    bool smooth;            // Use smooth shading?
    bool hasUV;             // Does triangle have UV coordinates?
    
    Triangle() : i0(0), i1(0), i2(0), smooth(false), hasUV(false) {}
};

// ========== BVH (Bounding Volume Hierarchy) ==========

struct BVHNode {
    Vec3 bmin, bmax;           // Bounding box of this node
    int left, right;           // Child indices (-1 for leaf)
    int triStart, triCount;    // Triangle range for leaf nodes
    
    bool isLeaf() const { return triCount > 0; }
};

struct BVH {
    std::vector<BVHNode> nodes;
    std::vector<int> triIndices;  // Reordered triangle indices
    
    // Build BVH from mesh triangles
    void build(const std::vector<Vec3>& vertices, std::vector<Triangle>& triangles) {
        if(triangles.empty()) return;
        
        // Initialize triangle indices and compute centroids
        triIndices.resize(triangles.size());
        for(size_t i = 0; i < triangles.size(); ++i) {
            triIndices[i] = (int)i;
            Triangle& tri = triangles[i];
            const Vec3& a = vertices[tri.i0];
            const Vec3& b = vertices[tri.i1];
            const Vec3& c = vertices[tri.i2];
            tri.centroid = Vec3((a.x+b.x+c.x)/3.0f, (a.y+b.y+c.y)/3.0f, (a.z+b.z+c.z)/3.0f);
        }
        
        // Reserve space for nodes (2n-1 nodes for n triangles in worst case)
        nodes.reserve(2 * triangles.size());
        
        // Build recursively
        buildRecursive(vertices, triangles, 0, (int)triangles.size());
    }
    
private:
    int buildRecursive(const std::vector<Vec3>& vertices, std::vector<Triangle>& triangles, 
                       int start, int end) {
        int nodeIdx = (int)nodes.size();
        nodes.push_back(BVHNode());
        BVHNode& node = nodes[nodeIdx];
        
        // Compute bounds for this node
        node.bmin = Vec3(1e30f, 1e30f, 1e30f);
        node.bmax = Vec3(-1e30f, -1e30f, -1e30f);
        for(int i = start; i < end; ++i) {
            const Triangle& tri = triangles[triIndices[i]];
            for(int v = 0; v < 3; ++v) {
                const Vec3& vert = vertices[v == 0 ? tri.i0 : (v == 1 ? tri.i1 : tri.i2)];
                node.bmin.x = std::min(node.bmin.x, vert.x);
                node.bmin.y = std::min(node.bmin.y, vert.y);
                node.bmin.z = std::min(node.bmin.z, vert.z);
                node.bmax.x = std::max(node.bmax.x, vert.x);
                node.bmax.y = std::max(node.bmax.y, vert.y);
                node.bmax.z = std::max(node.bmax.z, vert.z);
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
        Vec3 extent = node.bmax - node.bmin;
        int axis = 0;
        if(extent.y > extent.x) axis = 1;
        if(extent.z > (axis == 0 ? extent.x : extent.y)) axis = 2;
        
        // Sort by centroid along split axis
        int mid = (start + end) / 2;
        std::nth_element(triIndices.begin() + start, triIndices.begin() + mid, 
                        triIndices.begin() + end,
                        [&](int a, int b) {
                            float ca = axis == 0 ? triangles[a].centroid.x : 
                                      (axis == 1 ? triangles[a].centroid.y : triangles[a].centroid.z);
                            float cb = axis == 0 ? triangles[b].centroid.x : 
                                      (axis == 1 ? triangles[b].centroid.y : triangles[b].centroid.z);
                            return ca < cb;
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
    std::vector<Vec3> vertices;
    std::vector<Triangle> triangles;
    Material material;
    BVH bvh;  // BVH acceleration structure
    // axis-aligned bounding box (for whole mesh)
    Vec3 bmin{ 1e30f, 1e30f, 1e30f };
    Vec3 bmax{ -1e30f, -1e30f, -1e30f };
};

// ========== Environment Settings ==========

struct Environment {
    Vec3 color = Vec3(0.05f, 0.05f, 0.05f);  // Background color
    float strength = 1.0f;                    // Emission strength multiplier
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
        m.bmin.x = std::min(m.bmin.x, v.x);
        m.bmin.y = std::min(m.bmin.y, v.y);
        m.bmin.z = std::min(m.bmin.z, v.z);
        m.bmax.x = std::max(m.bmax.x, v.x);
        m.bmax.y = std::max(m.bmax.y, v.y);
        m.bmax.z = std::max(m.bmax.z, v.z);
    }
    // Add small epsilon to avoid zero-thickness boxes
    const float eps = 0.001f;
    if(m.bmax.x - m.bmin.x < eps) { m.bmin.x -= eps; m.bmax.x += eps; }
    if(m.bmax.y - m.bmin.y < eps) { m.bmin.y -= eps; m.bmax.y += eps; }
    if(m.bmax.z - m.bmin.z < eps) { m.bmin.z -= eps; m.bmax.z += eps; }
    for(auto &t : m.triangles) {
        const Vec3 &a = m.vertices[t.i0];
        const Vec3 &b = m.vertices[t.i1];
        const Vec3 &c = m.vertices[t.i2];
        t.faceNormal = Vec3::cross(b-a, c-a);
        t.faceNormal.normalize();
    }
    
    // Build BVH for this mesh
    m.bvh.build(m.vertices, m.triangles);
}
