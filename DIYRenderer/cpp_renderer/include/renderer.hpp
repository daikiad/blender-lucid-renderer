#pragma once
#include <vector>
#include <string>
#include <limits>
#include <cmath>

struct Vec3 {
    float x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    Vec3 operator+(const Vec3 &o) const { return {x+o.x, y+o.y, z+o.z}; }
    Vec3 operator-(const Vec3 &o) const { return {x-o.x, y-o.y, z-o.z}; }
    Vec3 operator*(float s) const { return {x*s, y*s, z*s}; }
    Vec3 operator/(float s) const { return {x/s, y/s, z/s}; }
    Vec3& normalize(){ float l = std::sqrt(x*x+y*y+z*z); if(l>0){ x/=l; y/=l; z/=l;} return *this; }
    static float dot(const Vec3 &a, const Vec3 &b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
    static Vec3 cross(const Vec3 &a, const Vec3 &b){ return { a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x }; }
};

struct Ray { Vec3 o; Vec3 d; };

struct Triangle { int i0, i1, i2; Vec3 faceNormal; };

struct Mesh {
    std::vector<Vec3> vertices;
    std::vector<Triangle> triangles;
    // axis-aligned bounding box
    Vec3 bmin{ 1e30f, 1e30f, 1e30f };
    Vec3 bmax{ -1e30f, -1e30f, -1e30f };
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

inline Vec3 traceRay(const Scene &scene, const Ray &ray){
    float closest = std::numeric_limits<float>::infinity();
    Vec3 normalHit{0,0,0};
    for(const auto &m : scene.meshes){
        if(!rayAABB(ray, m.bmin, m.bmax)) continue;
        for(const auto &tri : m.triangles){
            const Vec3 &a = m.vertices[tri.i0];
            const Vec3 &b = m.vertices[tri.i1];
            const Vec3 &c = m.vertices[tri.i2];
            float t = rayTriangle(ray, a, b, c);
            if(t > 0.0f && t < closest){
                closest = t;
                normalHit = tri.faceNormal;
                // Flip normal if hitting backface
                Vec3 e1 = b - a;
                Vec3 e2 = c - a;
                Vec3 geomNormal = Vec3::cross(e1, e2);
                if(Vec3::dot(geomNormal, ray.d) > 0) {
                    normalHit = normalHit * -1.0f;
                }
            }
        }
    }
    if(closest < std::numeric_limits<float>::infinity()) return normalHit; else return Vec3{0,0,0};
}
