/**
 * vec3.hpp - 3D Vector Mathematics
 * =================================
 * 
 * Core math structures for the renderer:
 * - Vec3: 3D vector for positions, directions, and colors
 * - Vec2: 2D vector for UV coordinates
 */

#pragma once
#include <cmath>

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
    
    Vec3& normalize() { 
        float l = std::sqrt(x*x + y*y + z*z); 
        if(l > 0) { x /= l; y /= l; z /= l; } 
        return *this; 
    }
    
    float length() const { return std::sqrt(x*x + y*y + z*z); }
    float lengthSquared() const { return x*x + y*y + z*z; }
    
    static float dot(const Vec3 &a, const Vec3 &b) { 
        return a.x*b.x + a.y*b.y + a.z*b.z; 
    }
    
    static Vec3 cross(const Vec3 &a, const Vec3 &b) { 
        return { 
            a.y*b.z - a.z*b.y, 
            a.z*b.x - a.x*b.z, 
            a.x*b.y - a.y*b.x 
        }; 
    }
};

/**
 * Vec2 - 2D Vector for UV coordinates
 */
struct Vec2 {
    float x, y;
    
    Vec2() : x(0), y(0) {}
    Vec2(float x_, float y_) : x(x_), y(y_) {}
    
    Vec2 operator+(const Vec2 &o) const { return Vec2(x + o.x, y + o.y); }
    Vec2 operator-(const Vec2 &o) const { return Vec2(x - o.x, y - o.y); }
    Vec2 operator*(float s) const { return Vec2(x * s, y * s); }
};
