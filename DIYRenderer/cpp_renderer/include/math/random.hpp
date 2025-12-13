/**
 * random.hpp - Random Number Generation
 * ======================================
 * 
 * High-quality random number generation for Monte Carlo rendering:
 * - PCG (Permuted Congruential Generator)
 * - Hash functions for decorrelation
 * - Sampling utilities
 */

#pragma once
#include "../units/render_units.hpp"
#include <cstdint>
#include <cmath>

// ========== PCG Random Number Generator ==========

// Thread-local PCG state
struct PCGState {
    uint64_t state;
    uint64_t inc;
};

inline PCGState& pcg_state() {
    static thread_local PCGState s = {0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL};
    return s;
}

// ========== Hash Functions ==========

// MurmurHash3 finalizer - very high quality bit mixing
inline uint32_t murmur3_finalize(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85ebca6bU;
    h ^= h >> 13;
    h *= 0xc2b2ae35U;
    h ^= h >> 16;
    return h;
}

// Splitmix64 - high quality 64-bit hash
inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// Jenkins hash - simple but effective for combining values
inline uint32_t jenkins_hash(uint32_t a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// ========== Seeding Functions ==========

// Seed with x, y pixel coordinates and sample index
// Uses multiple hash rounds for thorough decorrelation
inline void seed_random_xyz(uint32_t x, uint32_t y, uint32_t sampleIdx) {
    PCGState &s = pcg_state();
    
    // Combine x, y, sample using different mixing patterns
    uint32_t h1 = jenkins_hash(x + 1);
    uint32_t h2 = jenkins_hash(y + 1 + h1);
    uint32_t h3 = jenkins_hash(sampleIdx + 1 + h2);
    
    // Create two independent 64-bit seeds
    uint64_t seed1 = ((uint64_t)murmur3_finalize(h1 ^ h3) << 32) | murmur3_finalize(h2);
    uint64_t seed2 = ((uint64_t)murmur3_finalize(h2 ^ h1) << 32) | murmur3_finalize(h3);
    
    // Apply splitmix64 for final mixing
    uint64_t stream = splitmix64(seed1);
    uint64_t initseq = splitmix64(seed2);
    
    // PCG seeding - inc must be odd
    s.inc = (stream << 1u) | 1u;
    s.state = 0;
    s.state = s.state * 6364136223846793005ULL + s.inc;
    s.state += initseq;
    s.state = s.state * 6364136223846793005ULL + s.inc;
    
    // Extended warmup: discard more values for better decorrelation
    for (int i = 0; i < 8; i++) {
        s.state = s.state * 6364136223846793005ULL + s.inc;
    }
}

// Legacy seed functions for compatibility
inline void seed_random(uint32_t pixelIdx, uint32_t sampleIdx) {
    seed_random_xyz(pixelIdx % 10000, pixelIdx / 10000, sampleIdx);
}

inline void seed_random(uint32_t seed) {
    seed_random(seed, 0);
}

// ========== Random Number Generation ==========

// PCG32 random number generator
inline uint32_t pcg32() {
    PCGState &s = pcg_state();
    uint64_t oldstate = s.state;
    s.state = oldstate * 6364136223846793005ULL + s.inc;
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

// Generate random float between 0 and 1
// Uses ldexpf instead of division for ~2x faster conversion
// (pcg32() >> 9) gives 23 random bits for mantissa
// ldexpf(x, -23) = x * 2^-23
inline float randf() { 
    return std::ldexpf(static_cast<float>(pcg32() >> 9), -23);
}

// ========== Sampling Utilities ==========

// Generate random point inside unit sphere (rejection sampling)
inline render::Direction randomInUnitSphere() {
    while(true) {
        render::Vec3f p(randf()*2.0f-1.0f, randf()*2.0f-1.0f, randf()*2.0f-1.0f);
        if(p.length_squared() < 1.0f) return render::make_direction_or_default(p);
    }
}

// Generate random unit vector (uniform on sphere)
inline render::Direction randomUnitVector() { 
    // Generate using spherical coordinates for uniform distribution
    float u1 = randf();
    float u2 = randf();
    float z = 1.0f - 2.0f * u1;
    float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
    float phi = 2.0f * 3.14159265358979f * u2;
    return render::direction_from_unit_vector(render::Vec3f{r * std::cos(phi), r * std::sin(phi), z});
}

// Generate random direction in hemisphere around normal (cosine-weighted)
// This is importance sampling for Lambertian BRDF
// PDF = cos(theta) / PI
inline render::Direction randomCosineDirection(const render::Normal &normal) {
    render::Direction random_on_sphere = randomUnitVector();
    render::Vec3f result = normal.vec() + random_on_sphere.vec();
    auto dir = render::make_direction(result);
    // Fallback to normal if result is degenerate
    return dir.value_or(normal.as_direction());
}
