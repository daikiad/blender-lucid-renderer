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
    for(auto &t : m.triangles){
        const Vec3 &a = m.vertices[t.i0];
        const Vec3 &b = m.vertices[t.i1];
        const Vec3 &c = m.vertices[t.i2];
        t.faceNormal = Vec3::cross(b-a, c-a);
        t.faceNormal.normalize();
    }
}

inline bool rayAABB(const Ray &r, const Vec3 &bmin, const Vec3 &bmax){
    // Robust slab test handling near-zero direction components.
    auto inv = [](float v){ return v != 0.0f ? 1.0f / v : 1e30f; };
    float invX = inv(r.d.x); float invY = inv(r.d.y); float invZ = inv(r.d.z);
    float t1 = (bmin.x - r.o.x) * invX; float t2 = (bmax.x - r.o.x) * invX; if(t1 > t2) std::swap(t1, t2);
    float t3 = (bmin.y - r.o.y) * invY; float t4 = (bmax.y - r.o.y) * invY; if(t3 > t4) std::swap(t3, t4);
    if(t1 > t4 || t3 > t2) return false; if(t3 > t1) t1 = t3; if(t4 < t2) t2 = t4;
    float t5 = (bmin.z - r.o.z) * invZ; float t6 = (bmax.z - r.o.z) * invZ; if(t5 > t6) std::swap(t5, t6);
    if(t1 > t6 || t5 > t2) return false; // intersection exists
    return true;
}

inline float rayTriangle(const Ray &r, const Vec3 &v0, const Vec3 &v1, const Vec3 &v2){
    // Moller-Trumbore
    const float EPS = 1e-6f;
    Vec3 e1 = v1 - v0; Vec3 e2 = v2 - v0; Vec3 pvec = Vec3::cross(r.d, e2); float det = Vec3::dot(e1, pvec);
    if(det > -EPS && det < EPS) return -1.0f;
    float invDet = 1.0f / det;
    Vec3 tvec = r.o - v0;
    float u = Vec3::dot(tvec, pvec) * invDet; if(u < 0.0f || u > 1.0f) return -1.0f;
    Vec3 qvec = Vec3::cross(tvec, e1);
    float v = Vec3::dot(r.d, qvec) * invDet; if(v < 0.0f || u + v > 1.0f) return -1.0f;
    float t = Vec3::dot(e2, qvec) * invDet; if(t > EPS) return t; return -1.0f;
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
            if(t > 0.0f && t < closest){ closest = t; normalHit = tri.faceNormal; }
        }
    }
    if(closest < std::numeric_limits<float>::infinity()) return normalHit; else return Vec3{0,0,0};
}
