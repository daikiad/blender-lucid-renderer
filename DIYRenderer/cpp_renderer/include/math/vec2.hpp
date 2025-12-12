/**
 * vec2.hpp - 2D Vector for UV Coordinates
 * ========================================
 * 
 * Simple 2D vector for texture coordinates.
 * UV coordinates are dimensionless [0,1] values.
 */

#pragma once

struct Vec2 {
    float x, y;

    constexpr Vec2() : x(0), y(0) {}
    constexpr Vec2(float x_, float y_) : x(x_), y(y_) {}

    constexpr Vec2 operator+(const Vec2 &o) const { return Vec2(x + o.x, y + o.y); }
    constexpr Vec2 operator-(const Vec2 &o) const { return Vec2(x - o.x, y - o.y); }
    constexpr Vec2 operator*(float s) const { return Vec2(x * s, y * s); }
};
