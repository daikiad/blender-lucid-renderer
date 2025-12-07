/**
 * pbr.hpp - Physically Based Rendering (Compatibility Header)
 * ============================================================
 * 
 * This file provides backward compatibility by including all the
 * PBR components of the renderer.
 * 
 * New code should include specific headers:
 * - bsdf/fresnel.hpp, bsdf/ggx.hpp, bsdf/bsdf.hpp
 * - light/light.hpp, light/scene_lights.hpp  
 * - integrator/path_tracer.hpp
 */

#pragma once

// Include all dependencies
#include "renderer.hpp"

// BSDF components
#include "bsdf/fresnel.hpp"
#include "bsdf/ggx.hpp"
#include "bsdf/bsdf.hpp"

// Light components
#include "light/light.hpp"
#include "light/scene_lights.hpp"

// Path tracing integrators
#include "integrator/path_tracer.hpp"

// Re-export constants for backward compatibility
constexpr float PI = GGX_PI;
constexpr float INV_PI = GGX_INV_PI;
constexpr float EPSILON = GGX_EPSILON;
