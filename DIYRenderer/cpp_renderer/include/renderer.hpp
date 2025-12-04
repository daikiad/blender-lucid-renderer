/**
 * renderer.hpp - Core Path Tracing Renderer
 * ==========================================
 * 
 * This file contains the main rendering algorithms and data structures
 * for the DIY path tracer integrated with Blender.
 * 
 * Key Components:
 * - Vec3: 3D vector math for positions, directions, colors
 * - Ray: Ray structure for ray tracing
 * - Material: Surface material properties (albedo, emission, nodes)
 * - Scene: Collection of geometry and materials
 * - Intersection: Ray-geometry intersection routines
 * - Path Tracing: Monte Carlo integration for global illumination
 * 
 * Algorithms:
 * - Cosine-weighted hemisphere sampling (importance sampling)
 * - Russian Roulette path termination
 * - Node-based material evaluation (Principled BSDF, Emission)
 * - Debug visualization modes (normals, albedo, emission)
 */

#pragma once
#include <vector>
#include <string>
#include <limits>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <iostream>

// ========== Core Math Structures ==========

/**
 * Vec3 - 3D Vector for positions, directions, and colors
 * 
 * Used throughout the renderer for:
 * - 3D positions (world space coordinates)
 * - Direction vectors (normalized)
 * - RGB colors (linear color space)
 */
struct Vec3 {
    float x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    Vec3 operator+(const Vec3 &o) const { return {x+o.x, y+o.y, z+o.z}; }
    Vec3 operator-(const Vec3 &o) const { return {x-o.x, y-o.y, z-o.z}; }
    Vec3 operator*(float s) const { return {x*s, y*s, z*s}; }
    Vec3 operator*(const Vec3 &o) const { return {x*o.x, y*o.y, z*o.z}; }  // Component-wise multiply
    Vec3 operator/(float s) const { return {x/s, y/s, z/s}; }
    Vec3& normalize(){ float l = std::sqrt(x*x+y*y+z*z); if(l>0){ x/=l; y/=l; z/=l;} return *this; }
    float length() const { return std::sqrt(x*x+y*y+z*z); }
    static float dot(const Vec3 &a, const Vec3 &b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
    static Vec3 cross(const Vec3 &a, const Vec3 &b){ return { a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x }; }
};

// ========== Random Sampling Functions ==========
/**
 * Monte Carlo sampling utilities for path tracing
 * 
 * These functions generate random directions and points for:
 * - Hemisphere sampling (diffuse reflection)
 * - Importance sampling (cosine-weighted)
 * 
 * Uses xorshift32 for fast, high-quality random numbers.
 * Much faster than std::rand() with better statistical properties.
 */

// Thread-local PCG state for high-quality random number generation
// PCG (Permuted Congruential Generator) has excellent statistical properties
struct PCGState {
    uint64_t state;
    uint64_t inc;
};

inline PCGState& pcg_state() {
    static thread_local PCGState s = {0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL};
    return s;
}

// Seed the PCG random number generator
// Combines pixel position and sample number for unique sequences
inline void seed_random(uint32_t seed1, uint32_t seed2 = 0) {
    PCGState &s = pcg_state();
    s.state = 0;
    s.inc = ((uint64_t)seed1 << 1u) | 1u;  // Must be odd
    // Warm up
    s.state = s.state * 6364136223846793005ULL + s.inc;
    s.state += seed2;
    s.state = s.state * 6364136223846793005ULL + s.inc;
}

// Legacy single-seed version for compatibility
inline void seed_random(uint32_t seed) {
    seed_random(seed, 0);
}

// PCG32 random number generator
// Excellent statistical properties, passes all BigCrush tests
inline uint32_t pcg32() {
    PCGState &s = pcg_state();
    uint64_t oldstate = s.state;
    s.state = oldstate * 6364136223846793005ULL + s.inc;
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

// Generate random float between 0 and 1
inline float randf() { 
    return (float)pcg32() / (float)0xFFFFFFFFu;
}

// Generate random point inside unit sphere (rejection sampling)
// Used for uniform hemisphere sampling
inline Vec3 randomInUnitSphere(){
    while(true){
        Vec3 p = Vec3(randf()*2.0f-1.0f, randf()*2.0f-1.0f, randf()*2.0f-1.0f);
        if(p.length() < 1.0f) return p;
    }
}

// Generate random unit vector (uniform on sphere)
inline Vec3 randomUnitVector(){ 
    Vec3 v = randomInUnitSphere(); 
    v.normalize(); 
    return v; 
}

// Generate random direction in hemisphere around normal (cosine-weighted)
// This is importance sampling for Lambertian BRDF
// PDF = cos(theta) / PI
// Method: Add random unit vector to normal and normalize
// This naturally produces cosine-weighted distribution
inline Vec3 randomCosineDirection(const Vec3 &normal) {
    // Generate random point on unit sphere
    Vec3 random_on_sphere = randomUnitVector();
    // Add to normal - this biases toward normal direction
    // The result is naturally cosine-weighted!
    Vec3 result = normal + random_on_sphere;
    result.normalize();
    return result;
}

// ========== Glass/Refraction Functions ==========

// Reflect direction around normal
inline Vec3 reflect(const Vec3 &v, const Vec3 &n) {
    return v - n * 2.0f * Vec3::dot(v, n);
}

// Refract direction using Snell's law
// Returns zero vector if total internal reflection occurs
inline Vec3 refract(const Vec3 &uv, const Vec3 &n, float etai_over_etat) {
    float cos_theta = std::min(-Vec3::dot(uv, n), 1.0f);
    Vec3 r_out_perp = (uv + n * cos_theta) * etai_over_etat;
    float r_out_perp_len2 = Vec3::dot(r_out_perp, r_out_perp);
    if (r_out_perp_len2 > 1.0f) {
        // Total internal reflection
        return Vec3(0, 0, 0);
    }
    Vec3 r_out_parallel = n * (-std::sqrt(std::abs(1.0f - r_out_perp_len2)));
    return r_out_perp + r_out_parallel;
}

// Schlick's approximation for Fresnel reflectance
// cosine: cos(theta) where theta is angle between incident ray and normal
// ref_idx: relative index of refraction (n1/n2)
// Returns probability of reflection (0 to 1)
inline float schlickFresnelReflectance(float cosine, float ref_idx) {
    // r0 = ((n1-n2)/(n1+n2))^2 = ((1-n2/n1)/(1+n2/n1))^2 for air->material
    // But we receive ref_idx = n1/n2, so:
    // r0 = ((ref_idx - 1)/(ref_idx + 1))^2 when going from medium with higher index
    // For simplicity, use the standard formula with the IOR itself
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * std::pow((1.0f - cosine), 5.0f);
}

// ========== Node Graph Structures ==========
// These structures represent Blender's node-based material system
// Nodes connect via sockets to form a directed acyclic graph (DAG)

/**
 * SocketValue: Union type for socket values
 * 
 * CRITICAL: Must use correct field for each type to avoid bugs!
 * - VEC4 type → use v4 and v4_w fields
 * - VEC3 type → use v3 field  
 * - FLOAT type → use f field
 * 
 * Previous bug: All code was reading v3 field even for VEC4 types,
 * causing all RGBA colors (Base Color, Emission) to return (0,0,0).
 * Fixed by checking type and reading correct field.
 */
struct SocketValue {
    enum Type { FLOAT, VEC3, VEC4, STRING, BOOL, NONE };
    Type type;
    float f;          // For FLOAT - single scalar value
    Vec3 v3;          // For VEC3 - 3D vector (normals, positions)
    Vec3 v4;          // For VEC4 - RGB component (colors with alpha)
    float v4_w;       // For VEC4 - Alpha component
    std::string s;    // For STRING - texture paths, etc.
    bool b;           // For BOOL - boolean switches
    
    SocketValue() : type(NONE), f(0), v3(), v4(), v4_w(0), b(false) {}
    static SocketValue makeFloat(float val) { SocketValue sv; sv.type = FLOAT; sv.f = val; return sv; }
    static SocketValue makeVec3(float x, float y, float z) { SocketValue sv; sv.type = VEC3; sv.v3 = Vec3(x,y,z); return sv; }
    static SocketValue makeVec4(float x, float y, float z, float w) { SocketValue sv; sv.type = VEC4; sv.v4 = Vec3(x,y,z); sv.v4_w = w; return sv; }
};

/**
 * NodeSocket: Input or output socket on a node
 * Sockets can be connected to other nodes or use default values
 */
struct NodeSocket {
    std::string name;           // Socket name (e.g., "Base Color", "BSDF")
    std::string type;           // Socket type ("VALUE", "RGBA", "VECTOR", "SHADER")
    SocketValue default_value;  // Default value when not connected
    bool is_linked;             // Is this socket connected to another node?
    std::string linked_node;    // Name of connected node (if is_linked)
    std::string linked_socket;  // Name of connected socket (if is_linked)
    
    NodeSocket() : is_linked(false) {}
};

/**
 * MaterialNode: Single node in material graph
 * Examples: Principled BSDF, Mix RGB, Texture Coordinate, etc.
 */
struct MaterialNode {
    std::string name;   // Unique node name (generated by Blender)
    std::string type;   // Node type ("ShaderNodeBsdfPrincipled", "ShaderNodeMix", etc.)
    std::string label;  // User-visible label
    std::vector<NodeSocket> inputs;   // Input sockets
    std::vector<NodeSocket> outputs;  // Output sockets
    
    // Find input socket by name (returns nullptr if not found)
    const NodeSocket* findInput(const std::string &name) const {
        for(const auto &s : inputs) {
            if(s.name == name) return &s;
        }
        return nullptr;
    }
};

/**
 * NodeTree: Complete material node graph
 * Represents Blender's Shader Editor node setup
 */
struct NodeTree {
    std::vector<MaterialNode> nodes;
    bool valid;  // Is this node tree valid?
    
    NodeTree() : valid(false) {}
    
    // Find node by name (for following socket connections)
    const MaterialNode* findNode(const std::string &name) const {
        for(const auto &n : nodes) {
            if(n.name == name) return &n;
        }
        return nullptr;
    }
    
    /**
     * Find Material Output node (final node in shader graph)
     * This is the entry point for material evaluation
     */
    const MaterialNode* findOutputNode() const {
        for(const auto &n : nodes) {
            if(n.type == "ShaderNodeOutputMaterial") return &n;
        }
        return nullptr;
    }
};

// ========== Material Definition ==========
// Stores physical material properties from Blender's Principled BSDF
struct Material {
    Vec3 albedo;      // Base color (diffuse reflectance)
    float metallic;   // Metallic factor (0=dielectric, 1=metal)
    float roughness;  // Surface roughness
    Vec3 emission;    // Emission color * strength (for light sources)
    float transmission; // Glass/transparency (0=opaque, 1=fully transparent)
    float ior;        // Index of Refraction (1.0=air, 1.45=glass, 1.33=water)
    
    // Node-based material system
    NodeTree nodeTree;  // Full node graph from Blender
    bool useNodes;      // Should we use node tree or legacy properties?
    
    Material() : albedo(0.8f, 0.8f, 0.8f), metallic(0.0f), roughness(0.5f), 
                 emission(0.0f, 0.0f, 0.0f), transmission(0.0f), ior(1.45f), useNodes(false) {}
    Material(Vec3 a, float m, float r) : albedo(a), metallic(m), roughness(r), 
                 emission(0.0f, 0.0f, 0.0f), transmission(0.0f), ior(1.45f), useNodes(false) {}
    Material(Vec3 a, float m, float r, Vec3 e) : albedo(a), metallic(m), roughness(r), 
                 emission(e), transmission(0.0f), ior(1.45f), useNodes(false) {}
};

struct Ray { Vec3 o; Vec3 d; };

struct Triangle { int i0, i1, i2; Vec3 faceNormal; Vec3 centroid; };

// ========== BVH (Bounding Volume Hierarchy) ==========
// Accelerates ray-triangle intersection by organizing triangles
// in a binary tree of axis-aligned bounding boxes.

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

struct Mesh {
    std::vector<Vec3> vertices;
    std::vector<Triangle> triangles;
    Material material;
    BVH bvh;  // BVH acceleration structure
    // axis-aligned bounding box (for whole mesh)
    Vec3 bmin{ 1e30f, 1e30f, 1e30f };
    Vec3 bmax{ -1e30f, -1e30f, -1e30f };
};

// ========== Ray-Scene Intersection Result ==========
struct Hit {
    bool hit;           // Did the ray hit anything?
    float t;            // Distance along ray to hit point
    Vec3 point;         // 3D position of hit point
    Vec3 normal;        // Surface normal at hit point
    Material material;  // Material properties of hit surface
    Hit() : hit(false), t(1e30f) {}
};

struct Scene {
    std::vector<Mesh> meshes;
};

inline void finalizeMeshBounds(Mesh &m){
    for(const auto &v : m.vertices){
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
    for(auto &t : m.triangles){
        const Vec3 &a = m.vertices[t.i0];
        const Vec3 &b = m.vertices[t.i1];
        const Vec3 &c = m.vertices[t.i2];
        t.faceNormal = Vec3::cross(b-a, c-a);
        t.faceNormal.normalize();
    }
    
    // Build BVH for this mesh
    m.bvh.build(m.vertices, m.triangles);
}

// Optimized AABB test with precomputed inverse direction
inline bool rayAABBFast(const Ray &r, const Vec3 &invDir, const Vec3 &bmin, const Vec3 &bmax){
    float t1 = (bmin.x - r.o.x) * invDir.x;
    float t2 = (bmax.x - r.o.x) * invDir.x;
    float tmin = std::min(t1, t2);
    float tmax = std::max(t1, t2);
    
    t1 = (bmin.y - r.o.y) * invDir.y;
    t2 = (bmax.y - r.o.y) * invDir.y;
    tmin = std::max(tmin, std::min(t1, t2));
    tmax = std::min(tmax, std::max(t1, t2));
    
    t1 = (bmin.z - r.o.z) * invDir.z;
    t2 = (bmax.z - r.o.z) * invDir.z;
    tmin = std::max(tmin, std::min(t1, t2));
    tmax = std::min(tmax, std::max(t1, t2));
    
    return tmax >= std::max(tmin, 0.0f);
}

inline bool rayAABB(const Ray &r, const Vec3 &bmin, const Vec3 &bmax){
    // Standard slab test with proper interval tracking
    float tmin = 0.0f;
    float tmax = 1e30f;
    
    for(int i = 0; i < 3; ++i) {
        float origin = (i == 0) ? r.o.x : (i == 1) ? r.o.y : r.o.z;
        float dir = (i == 0) ? r.d.x : (i == 1) ? r.d.y : r.d.z;
        float boxmin = (i == 0) ? bmin.x : (i == 1) ? bmin.y : bmin.z;
        float boxmax = (i == 0) ? bmax.x : (i == 1) ? bmax.y : bmax.z;
        
        if(std::abs(dir) > 1e-8f) {
            float t1 = (boxmin - origin) / dir;
            float t2 = (boxmax - origin) / dir;
            if(t1 > t2) std::swap(t1, t2);
            tmin = std::max(tmin, t1);
            tmax = std::min(tmax, t2);
            if(tmin > tmax) return false;
        } else {
            // Ray parallel to slab - check if origin is inside
            if(origin < boxmin || origin > boxmax) return false;
        }
    }
    return tmax > 0.0f;  // intersection exists in front of ray
}

inline float rayTriangle(const Ray &r, const Vec3 &v0, const Vec3 &v1, const Vec3 &v2){
    // Moller-Trumbore with double-sided test (no backface culling)
    const float EPS = 1e-6f;
    Vec3 e1 = v1 - v0; Vec3 e2 = v2 - v0; Vec3 pvec = Vec3::cross(r.d, e2); float det = Vec3::dot(e1, pvec);
    if(det > -EPS && det < EPS) return -1.0f;  // parallel
    float invDet = 1.0f / det;
    Vec3 tvec = r.o - v0;
    float u = Vec3::dot(tvec, pvec) * invDet; if(u < 0.0f || u > 1.0f) return -1.0f;
    Vec3 qvec = Vec3::cross(tvec, e1);
    float v = Vec3::dot(r.d, qvec) * invDet; if(v < 0.0f || u + v > 1.0f) return -1.0f;
    float t = Vec3::dot(e2, qvec) * invDet;
    return (t > EPS) ? t : -1.0f;  // both sides valid if t > 0
}

// BVH traversal for a single mesh
inline void intersectBVH(const Mesh &mesh, const Ray &ray, const Vec3 &invDir, Hit &result) {
    if(mesh.bvh.nodes.empty()) {
        // Fallback: linear scan if no BVH
        for(const auto &tri : mesh.triangles) {
            const Vec3 &a = mesh.vertices[tri.i0];
            const Vec3 &b = mesh.vertices[tri.i1];
            const Vec3 &c = mesh.vertices[tri.i2];
            float t = rayTriangle(ray, a, b, c);
            if(t > 0.0001f && t < result.t) {
                result.t = t;
                result.hit = true;
                result.point = ray.o + ray.d * t;
                result.normal = tri.faceNormal;
                result.material = mesh.material;
            }
        }
        return;
    }
    
    // Stack-based BVH traversal (non-recursive for speed)
    int stack[64];
    int stackPtr = 0;
    stack[stackPtr++] = 0;  // Start with root node
    
    while(stackPtr > 0) {
        int nodeIdx = stack[--stackPtr];
        const BVHNode &node = mesh.bvh.nodes[nodeIdx];
        
        // Test ray against node bounds
        if(!rayAABBFast(ray, invDir, node.bmin, node.bmax)) continue;
        
        if(node.isLeaf()) {
            // Test triangles in this leaf
            for(int i = 0; i < node.triCount; ++i) {
                int triIdx = mesh.bvh.triIndices[node.triStart + i];
                const Triangle &tri = mesh.triangles[triIdx];
                const Vec3 &a = mesh.vertices[tri.i0];
                const Vec3 &b = mesh.vertices[tri.i1];
                const Vec3 &c = mesh.vertices[tri.i2];
                float t = rayTriangle(ray, a, b, c);
                if(t > 0.0001f && t < result.t) {
                    result.t = t;
                    result.hit = true;
                    result.point = ray.o + ray.d * t;
                    result.normal = tri.faceNormal;
                    result.material = mesh.material;
                }
            }
        } else {
            // Push children onto stack
            if(node.right >= 0) stack[stackPtr++] = node.right;
            if(node.left >= 0) stack[stackPtr++] = node.left;
        }
    }
}

inline Hit intersectScene(const Scene &scene, const Ray &ray, bool useAABB = true){
    Hit result;
    result.t = 1e30f;
    result.hit = false;
    
    // Precompute inverse direction for faster AABB tests
    Vec3 invDir(
        std::abs(ray.d.x) > 1e-8f ? 1.0f / ray.d.x : 1e30f,
        std::abs(ray.d.y) > 1e-8f ? 1.0f / ray.d.y : 1e30f,
        std::abs(ray.d.z) > 1e-8f ? 1.0f / ray.d.z : 1e30f
    );
    
    for(const auto &m : scene.meshes){
        // First check mesh-level AABB
        if(useAABB && !rayAABBFast(ray, invDir, m.bmin, m.bmax)) continue;
        
        // Use BVH for triangle intersection
        intersectBVH(m, ray, invDir, result);
    }
    
    // NOTE: We no longer flip normals here for backface hits.
    // Glass materials need to know the original geometric normal direction
    // to determine if we're entering or exiting the material.
    // The trace() function handles normal direction as needed.
    
    return result;
}

// ========== Node Graph Evaluation (Forward Declarations) ==========
// These functions are implemented in node_evaluator.cpp

// Evaluate a single node (recursive for connected inputs)
Vec3 evaluateNode(const NodeTree &tree, const std::string &nodeName, const std::string &socketName);

// Get albedo from node tree (follows connections from Material Output)
Vec3 getAlbedoFromNodeTree(const NodeTree &tree);

// Get emission from node tree
Vec3 getEmissionFromNodeTree(const NodeTree &tree);

// Get transmission (glass) from node tree (0.0 = opaque, 1.0 = fully transparent)
float getTransmissionFromNodeTree(const NodeTree &tree);

// Get Index of Refraction from node tree (default 1.45 for glass)
float getIORFromNodeTree(const NodeTree &tree);

// ========== Debug Rendering Functions ==========

// Debug mode: return normal as color
inline Vec3 traceNormal(const Scene &scene, const Ray &ray, bool useAABB = true){
    Hit hit = intersectScene(scene, ray, useAABB);
    if(hit.hit){
        return hit.normal;
    }
    return Vec3{0,0,0};
}

// Debug mode: return albedo (base color) from material
inline Vec3 traceAlbedo(const Scene &scene, const Ray &ray, bool useAABB = true){
    static int hitCount = 0;
    Hit hit = intersectScene(scene, ray, useAABB);
    if(hit.hit){
        if(++hitCount <= 3) {
            std::cerr << "[traceAlbedo] Hit! useNodes=" << hit.material.useNodes 
                      << " treeValid=" << hit.material.nodeTree.valid 
                      << " legacyAlbedo=(" << hit.material.albedo.x << "," << hit.material.albedo.y << "," << hit.material.albedo.z << ")\n";
        }
        // Use node tree if available, otherwise legacy properties
        if(hit.material.useNodes && hit.material.nodeTree.valid) {
            return getAlbedoFromNodeTree(hit.material.nodeTree);
        }
        return hit.material.albedo;
    }
    return Vec3{0,0,0};
}

// Debug mode: return emission from material
inline Vec3 traceEmission(const Scene &scene, const Ray &ray, bool useAABB = true){
    Hit hit = intersectScene(scene, ray, useAABB);
    if(hit.hit){
        // Use node tree if available, otherwise legacy properties
        if(hit.material.useNodes && hit.material.nodeTree.valid) {
            return getEmissionFromNodeTree(hit.material.nodeTree);
        }
        return hit.material.emission;
    }
    return Vec3{0,0,0};
}

// Sky/environment color for background
// Set to black (no environment lighting) to match Cycles with black world
inline Vec3 getEnvironmentColor(const Ray &ray){
    return Vec3(0.0f, 0.0f, 0.0f);
}

// Shadow test
inline bool isInShadow(const Scene &scene, const Vec3 &point, const Vec3 &lightDir, float lightDist){
    Ray shadowRay{point + lightDir * 0.001f, lightDir};
    Hit hit = intersectScene(scene, shadowRay, true);
    return hit.hit && hit.t < lightDist;
}

// ========== Path Tracing Core ==========
// Recursively traces a ray through the scene using Monte Carlo integration
// This implements the rendering equation with importance sampling
inline Vec3 trace(const Scene &scene, const Ray &ray, int depth, bool useAABB = true){
    // Base case: maximum recursion depth reached
    if(depth <= 0) return Vec3{0,0,0};
    
    // Russian Roulette: probabilistically terminate paths after depth 2
    // Only apply RR after a few bounces to avoid bias in direct lighting
    float rrProbability = 1.0f;
    if(depth < 3) {
        rrProbability = 0.9f;  // 90% chance to continue (more conservative)
        if(randf() > rrProbability) {
            return Vec3{0,0,0};  // Terminate path
        }
    }
    
    // Test ray against all geometry in scene
    Hit hit = intersectScene(scene, ray, useAABB);
    
    // Ray escaped to infinity - return environment/sky color
    // This provides ambient lighting from the "sky dome"
    if(!hit.hit) {
        return getEnvironmentColor(ray);
    }
    
    // Evaluate material properties (use node tree if available, fallback to legacy)
    Vec3 albedo = hit.material.albedo;
    Vec3 emission = hit.material.emission;
    float transmission = hit.material.transmission;
    float ior = hit.material.ior;
    
    if(hit.material.useNodes && hit.material.nodeTree.valid) {
        // Use node-based material evaluation
        albedo = getAlbedoFromNodeTree(hit.material.nodeTree);
        emission = getEmissionFromNodeTree(hit.material.nodeTree);
        transmission = getTransmissionFromNodeTree(hit.material.nodeTree);
        ior = getIORFromNodeTree(hit.material.nodeTree);
    }
    
    // Debug: print glass material info (only first few times)
    static int glassDebugCount = 0;
    if(transmission > 0.0f && glassDebugCount < 5) {
        glassDebugCount++;
        std::cerr << "[Glass] transmission=" << transmission << ", ior=" << ior 
                  << ", albedo=(" << albedo.x << "," << albedo.y << "," << albedo.z << ")\n";
    }
    
    // Start with emission (if this surface is a light source)
    Vec3 result = emission;
    
    Ray scattered;
    Vec3 attenuation = albedo;
    
    // Glass/Transparent material handling
    if(transmission > 0.0f) {
        // Mix between glass and diffuse based on transmission factor
        if(randf() >= transmission) {
            // Diffuse path (partial transmission - e.g. frosted glass)
            Vec3 normal = hit.normal;
            if(Vec3::dot(ray.d, normal) > 0) {
                normal = normal * -1.0f;
            }
            Vec3 scatterDir = randomCosineDirection(normal);
            scattered.o = hit.point + normal * 0.001f;
            scattered.d = scatterDir;
        } else {
            // Glass path - full refraction/reflection
            
            // Determine if we're entering or exiting the material
            // frontFace = true means ray is hitting the outside of the surface
            bool frontFace = Vec3::dot(ray.d, hit.normal) < 0;
            
            // Always use outward-facing normal for calculations
            Vec3 n = frontFace ? hit.normal : hit.normal * -1.0f;
            
            // IOR ratio: n1/n2 where n1 is current medium, n2 is the medium we're entering
            // Entering glass (frontFace=true): air(1.0) -> glass(ior), ratio = 1/ior
            // Exiting glass (frontFace=false): glass(ior) -> air(1.0), ratio = ior/1 = ior
            float refraction_ratio = frontFace ? (1.0f / ior) : ior;
            
            Vec3 unit_direction = ray.d;
            unit_direction.normalize();
            
            float cos_theta = std::min(-Vec3::dot(unit_direction, n), 1.0f);
            float sin_theta = std::sqrt(1.0f - cos_theta * cos_theta);
            
            // Check for total internal reflection
            bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
            
            // Schlick's approximation for Fresnel reflectance
            // For Schlick, we need to use the cosine from the less dense medium
            // If entering glass (frontFace), use cos_theta directly
            // If exiting glass, we should use the refracted angle's cosine
            float reflectance;
            if(cannot_refract) {
                reflectance = 1.0f;  // Total internal reflection
            } else {
                // Calculate Fresnel reflectance
                // Use the angle in the less optically dense medium
                float cos_for_fresnel = cos_theta;
                if(!frontFace) {
                    // Exiting: use the refracted angle (angle in air)
                    float sin_refracted = refraction_ratio * sin_theta;
                    cos_for_fresnel = std::sqrt(1.0f - sin_refracted * sin_refracted);
                }
                // r0 for air-glass interface: ((1 - ior) / (1 + ior))^2
                float r0 = (1.0f - ior) / (1.0f + ior);
                r0 = r0 * r0;
                reflectance = r0 + (1.0f - r0) * std::pow((1.0f - cos_for_fresnel), 5.0f);
            }
            
            // Debug output
            static int fresnelDebugCount = 0;
            if(fresnelDebugCount < 3) {
                fresnelDebugCount++;
                std::cerr << "[Fresnel] frontFace=" << frontFace << ", cos_theta=" << cos_theta 
                          << ", refraction_ratio=" << refraction_ratio << ", reflectance=" << reflectance << "\n";
            }
            
            Vec3 direction;
            if(cannot_refract || randf() < reflectance) {
                // Reflect - ray bounces off surface
                direction = reflect(unit_direction, n);
                // Offset along the reflection direction (same side as incoming ray)
                scattered.o = hit.point + n * 0.001f;
            } else {
                // Refract - ray passes through surface
                direction = refract(unit_direction, n, refraction_ratio);
                // Offset in the direction of refraction (opposite side of normal)
                scattered.o = hit.point - n * 0.001f;
                
                // Debug refraction
                static int refractDebugCount = 0;
                if(refractDebugCount < 5) {
                    refractDebugCount++;
                    float incident_angle = std::acos(cos_theta) * 180.0f / 3.14159f;
                    Vec3 dir_normalized = direction;
                    dir_normalized.normalize();
                    float refract_cos = std::abs(Vec3::dot(dir_normalized, n));
                    float refract_angle = std::acos(refract_cos) * 180.0f / 3.14159f;
                    std::cerr << "[Refract] frontFace=" << frontFace 
                              << ", incident_angle=" << incident_angle << "deg"
                              << ", refract_angle=" << refract_angle << "deg"
                              << ", ratio=" << refraction_ratio << "\n";
                    std::cerr << "  in_dir=(" << unit_direction.x << "," << unit_direction.y << "," << unit_direction.z << ")"
                              << ", n=(" << n.x << "," << n.y << "," << n.z << ")"
                              << ", out_dir=(" << direction.x << "," << direction.y << "," << direction.z << ")\n";
                }
            }
            
            scattered.d = direction;
            attenuation = Vec3(1.0f, 1.0f, 1.0f);  // Pure glass doesn't absorb light
            // For colored glass, use: attenuation = albedo;
        }
    } else {
        // Opaque diffuse material
        // For diffuse, we need the normal to face the ray (flip if backface)
        Vec3 normal = hit.normal;
        if(Vec3::dot(ray.d, normal) > 0) {
            normal = normal * -1.0f;
        }
        Vec3 scatterDir = randomCosineDirection(normal);
        scattered.o = hit.point + normal * 0.001f;
        scattered.d = scatterDir;
    }
    
    // Recursively trace the scattered ray to get incoming light
    Vec3 incomingLight = trace(scene, scattered, depth - 1, useAABB);
    
    // Rendering equation with cosine-weighted importance sampling:
    // Lo = Le + integral(BRDF * Li * cos(theta))
    // For diffuse: BRDF terms cancel with PDF
    // For glass: attenuation is the glass color/tint
    result = result + attenuation * incomingLight;
    
    // Apply Russian Roulette compensation
    if(rrProbability < 1.0f) {
        result = result * (1.0f / rrProbability);
    }
    
    return result;
}
