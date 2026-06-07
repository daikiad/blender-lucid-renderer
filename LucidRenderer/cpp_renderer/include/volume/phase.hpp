/**
 * phase.hpp - Henyey-Greenstein Phase Function
 * =============================================
 *
 * Henyey-Greenstein single-parameter phase function for participating media.
 * Convention: positive g = forward scattering, negative g = back scattering,
 * g = 0 = isotropic (uniform sphere). Same convention as Cycles' Principled
 * Volume `anisotropy` slider.
 *
 *   p(theta) = (1 - g^2) / (4*pi * (1 + g^2 - 2*g*cos_theta)^(3/2))
 *
 * Normalized over the unit sphere — i.e. eval and pdf are identical.
 *
 * Sampling (PBRT §11.3): inverse-CDF on cos_theta, uniform on azimuth.
 */

#pragma once

#include "../units/render_units.hpp"
#include "../bsdf/ggx.hpp"   // for buildOrthonormalBasis / localToWorld
#include <cmath>
#include <numbers>
#include <utility>

namespace volume {

constexpr float HG_PI    = std::numbers::pi_v<float>;
constexpr float HG_INV_4PI = 1.0f / (4.0f * std::numbers::pi_v<float>);
constexpr float HG_EPS_G   = 1e-3f;   // below this |g| we treat as isotropic

// Phase value [1/sr]. Same as pdf since HG is normalized over the sphere.
inline float hg_eval(float g, float cos_theta) {
    const float gg = g * g;
    const float denom = 1.0f + gg - 2.0f * g * cos_theta;
    // numerically clamp denom to avoid pow(0, 1.5) explosions at the
    // forward-scatter singularity for |g| -> 1.
    const float d = std::max(denom, 1e-8f);
    return HG_INV_4PI * (1.0f - gg) / (d * std::sqrt(d));
}

inline float hg_pdf(float g, float cos_theta) {
    return hg_eval(g, cos_theta);
}

// Sample a direction wi given wo. Returns (wi, pdf in [1/sr]).
// wi is built in a frame where wo is the +Z axis, then transformed back.
inline std::pair<render::Direction, float>
hg_sample(float g, const render::Direction& wo, float u1, float u2) {
    float cos_theta;
    if (std::fabs(g) < HG_EPS_G) {
        // Isotropic: uniform on sphere -> cos_theta = 1 - 2u
        cos_theta = 1.0f - 2.0f * u1;
    } else {
        // Standard HG inverse-CDF (PBRT §11.3 Eq. 11.16-style)
        const float gg = g * g;
        const float sqr = (1.0f - gg) / (1.0f - g + 2.0f * g * u1);
        cos_theta = (1.0f + gg - sqr * sqr) / (2.0f * g);
        cos_theta = std::clamp(cos_theta, -1.0f, 1.0f);
    }
    const float sin_theta = std::sqrt(std::max(0.0f, 1.0f - cos_theta * cos_theta));
    const float phi = 2.0f * HG_PI * u2;

    // Local frame: +Z = wo. cos_theta is the angle between wi and wo.
    render::Direction tangent, bitangent;
    buildOrthonormalBasis(wo, tangent, bitangent);
    const render::Vec3f local{
        sin_theta * std::cos(phi),
        sin_theta * std::sin(phi),
        cos_theta,
    };
    render::Direction wi = render::direction_from_unit_vector(render::Vec3f{
        tangent.x() * local.x + bitangent.x() * local.y + wo.x() * local.z,
        tangent.y() * local.x + bitangent.y() * local.y + wo.y() * local.z,
        tangent.z() * local.x + bitangent.z() * local.y + wo.z() * local.z,
    });
    return {wi, hg_pdf(g, cos_theta)};
}

}  // namespace volume
