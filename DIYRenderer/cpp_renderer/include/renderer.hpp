#pragma once
#include <vector>
#include <string>
#include <limits>
#include <cmath>
#include <cstdlib>

struct Vec3 {
    float x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    Vec3 operator+(const Vec3 &o) const { return {x+o.x, y+o.y, z+o.z}; }
    Vec3 operator-(const Vec3 &o) const { return {x-o.x, y-o.y, z-o.z}; }
    Vec3 operator*(float s) const { return {x*s, y*s, z*s}; }
    Vec3 operator*(const Vec3 &o) const { return {x*o.x, y*o.y, z*o.z}; }
    Vec3 operator/(float s) const { return {x/s, y/s, z/s}; }
    Vec3& normalize(){ float l = std::sqrt(x*x+y*y+z*z); if(l>0){ x/=l; y/=l; z/=l;} return *this; }
    float length() const { return std::sqrt(x*x+y*y+z*z); }
    static float dot(const Vec3 &a, const Vec3 &b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
    static Vec3 cross(const Vec3 &a, const Vec3 &b){ return { a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x }; }
};

// ========== Random Sampling Functions ==========
// These are used for Monte Carlo path tracing

// Generate random float between 0 and 1
inline float randf(){ return (float)std::rand() / (float)RAND_MAX; }

// Generate random point inside unit sphere (rejection sampling)
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
inline Vec3 randomCosineDirection(const Vec3 &normal) {
    Vec3 random_dir = randomUnitVector();
    // Flip to correct hemisphere if needed
    if (Vec3::dot(random_dir, normal) < 0.0f) {
        random_dir = random_dir * -1.0f;
    }
    return random_dir;
}

// ========== Material Definition ==========
// Stores physical material properties from Blender's Principled BSDF
struct Material {
    Vec3 albedo;      // Base color (diffuse reflectance)
    float metallic;   // Metallic factor (0=dielectric, 1=metal) [currently unused]
    float roughness;  // Surface roughness [currently unused]
    Vec3 emission;    // Emission color * strength (for light sources)
    
    Material() : albedo(0.8f, 0.8f, 0.8f), metallic(0.0f), roughness(0.5f), emission(0.0f, 0.0f, 0.0f) {}
    Material(Vec3 a, float m, float r) : albedo(a), metallic(m), roughness(r), emission(0.0f, 0.0f, 0.0f) {}
    Material(Vec3 a, float m, float r, Vec3 e) : albedo(a), metallic(m), roughness(r), emission(e) {}
};

struct Ray { Vec3 o; Vec3 d; };

struct Triangle { int i0, i1, i2; Vec3 faceNormal; };

struct Mesh {
    std::vector<Vec3> vertices;
    std::vector<Triangle> triangles;
    Material material;
    // axis-aligned bounding box
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

inline Hit intersectScene(const Scene &scene, const Ray &ray, bool useAABB = true){
    Hit result;
    result.t = 1e30f;
    result.hit = false;
    
    for(const auto &m : scene.meshes){
        if(useAABB && !rayAABB(ray, m.bmin, m.bmax)) continue;
        for(const auto &tri : m.triangles){
            const Vec3 &a = m.vertices[tri.i0];
            const Vec3 &b = m.vertices[tri.i1];
            const Vec3 &c = m.vertices[tri.i2];
            float t = rayTriangle(ray, a, b, c);
            if(t > 0.0001f && t < result.t){
                result.t = t;
                result.hit = true;
                result.point = ray.o + ray.d * t;
                result.normal = tri.faceNormal;
                result.material = m.material;
                // Flip normal if hitting backface
                if(Vec3::dot(result.normal, ray.d) > 0) {
                    result.normal = result.normal * -1.0f;
                }
            }
        }
    }
    return result;
}

// Debug mode: return normal as color
inline Vec3 traceNormal(const Scene &scene, const Ray &ray, bool useAABB = true){
    Hit hit = intersectScene(scene, ray, useAABB);
    if(hit.hit){
        return hit.normal;
    }
    return Vec3{0,0,0};
}

// Sky color for background
inline Vec3 skyColor(const Ray &ray){
    float t = 0.5f * (ray.d.y + 1.0f);
    return Vec3(1.0f, 1.0f, 1.0f) * (1.0f - t) + Vec3(0.5f, 0.7f, 1.0f) * t;
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
    
    // Test ray against all geometry in scene
    Hit hit = intersectScene(scene, ray, useAABB);
    
    // Ray escaped to infinity - return black (no environment lighting)
    if(!hit.hit) return Vec3(0,0,0);
    
    // Check if we hit a light source (emission > 0)
    float emissionMagnitude = hit.material.emission.x + hit.material.emission.y + hit.material.emission.z;
    if(emissionMagnitude > 0.001f) {
        // Direct hit on light - return its emission
        return hit.material.emission;
    }
    
    // Russian roulette path termination (stochastic early exit)
    // After first few bounces, randomly terminate paths to save computation
    // Must compensate by dividing by survival probability
    float survivalProbability = 0.9f;
    if(depth < 3 && randf() > survivalProbability) {
        return Vec3(0,0,0);  // Path terminated
    }
    
    // Scatter ray in random direction using cosine-weighted sampling
    // This is importance sampling for Lambertian (diffuse) surfaces
    Vec3 scatterDir = randomCosineDirection(hit.normal);
    Ray scattered{hit.point + hit.normal * 0.001f, scatterDir};  // Offset to avoid self-intersection
    
    // Recursively trace the scattered ray to get incoming light
    Vec3 incomingLight = trace(scene, scattered, depth - 1, useAABB);
    
    // Rendering equation with cosine-weighted importance sampling:
    // Lo = integral(BRDF * Li * cos(theta))
    // BRDF (Lambertian) = albedo / PI
    // PDF (cosine-weighted) = cos(theta) / PI
    // Monte Carlo estimator: (BRDF * Li * cos(theta)) / PDF
    //                       = (albedo/PI * Li * cos(theta)) / (cos(theta)/PI)
    //                       = albedo * Li
    // The cos(theta) and PI terms cancel out!
    Vec3 result = hit.material.albedo * incomingLight;
    
    // Russian roulette compensation: divide by survival probability
    if(depth < 3) {
        result = result * (1.0f / survivalProbability);
    }
    
    return result;
}
