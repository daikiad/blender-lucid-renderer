/**
 * renderer.hpp - Core Path Tracing Renderer (Compatibility Header)
 * =================================================================
 * 
 * This file provides backward compatibility by including all the
 * modular components of the renderer.
 * 
 * New code should include specific headers:
 * - math/vec3.hpp, math/random.hpp
 * - core/ray.hpp, core/material.hpp, core/scene.hpp
 * - geometry/intersection.hpp, geometry/bvh.hpp
 * - bsdf/fresnel.hpp, bsdf/ggx.hpp, bsdf/bsdf.hpp
 * - light/light.hpp, light/scene_lights.hpp
 * - integrator/path_tracer.hpp
 */

#pragma once

// Math utilities
#include "math/vec3.hpp"
#include "math/random.hpp"

// Core structures
#include "core/ray.hpp"
#include "core/material.hpp"
#include "core/scene.hpp"

// Geometry and intersection
#include "geometry/intersection.hpp"
#include "geometry/bvh.hpp"

// Node evaluator forward declarations (implemented in node_evaluator.cpp)
Vec3 evaluateNode(const NodeTree &tree, const std::string &nodeName, const std::string &socketName, const Vec2 &uv);
Vec3 getAlbedoFromNodeTree(const NodeTree &tree, const Vec2 &uv);
Vec3 getEmissionFromNodeTree(const NodeTree &tree, const Vec2 &uv);
float getTransmissionFromNodeTree(const NodeTree &tree, const Vec2 &uv);
float getIORFromNodeTree(const NodeTree &tree, const Vec2 &uv);
float getMetallicFromNodeTree(const NodeTree &tree, const Vec2 &uv);
float getRoughnessFromNodeTree(const NodeTree &tree, const Vec2 &uv);

// ========== Debug Rendering Functions ==========

// Debug mode: return normal as color
inline Vec3 traceNormal(const Scene &scene, const Ray &ray, bool useAABB = true) {
    Hit hit = intersectScene(scene, ray, useAABB);
    if (hit.hit) {
        return hit.normal;
    }
    return Vec3{0, 0, 0};
}

// Debug mode: return albedo from material
inline Vec3 traceAlbedo(const Scene &scene, const Ray &ray, bool useAABB = true) {
    Hit hit = intersectScene(scene, ray, useAABB);
    if (hit.hit) {
        if (hit.material.useNodes && hit.material.nodeTree.valid) {
            return getAlbedoFromNodeTree(hit.material.nodeTree, hit.uv);
        }
        return hit.material.albedo;
    }
    return Vec3{0, 0, 0};
}

// Debug mode: return emission from material
inline Vec3 traceEmission(const Scene &scene, const Ray &ray, bool useAABB = true) {
    Hit hit = intersectScene(scene, ray, useAABB);
    if (hit.hit) {
        if (hit.material.useNodes && hit.material.nodeTree.valid) {
            return getEmissionFromNodeTree(hit.material.nodeTree, hit.uv);
        }
        return hit.material.emission;
    }
    return Vec3{0, 0, 0};
}

// ========== Legacy Path Tracing ==========

/**
 * Legacy trace function for backward compatibility
 * Uses simple path tracing with cosine-weighted sampling
 */
inline Vec3 trace(const Scene &scene, const Ray &ray, int depth, bool useAABB = true) {
    if (depth <= 0) return Vec3{0, 0, 0};
    
    // Russian Roulette
    float rrProbability = 1.0f;
    if (depth < 3) {
        rrProbability = 0.9f;
        if (randf() > rrProbability) {
            return Vec3{0, 0, 0};
        }
    }
    
    Hit hit = intersectScene(scene, ray, useAABB);
    
    if (!hit.hit) {
        return getEnvironmentColor(ray);
    }
    
    // Evaluate material properties
    Vec3 albedo = hit.material.albedo;
    Vec3 emission = hit.material.emission;
    float transmission = hit.material.transmission;
    float ior = hit.material.ior;
    float metallic = hit.material.metallic;
    float roughness = hit.material.roughness;
    
    if (hit.material.useNodes && hit.material.nodeTree.valid) {
        albedo = getAlbedoFromNodeTree(hit.material.nodeTree, hit.uv);
        emission = getEmissionFromNodeTree(hit.material.nodeTree, hit.uv);
        transmission = getTransmissionFromNodeTree(hit.material.nodeTree, hit.uv);
        ior = getIORFromNodeTree(hit.material.nodeTree, hit.uv);
        metallic = getMetallicFromNodeTree(hit.material.nodeTree, hit.uv);
        roughness = getRoughnessFromNodeTree(hit.material.nodeTree, hit.uv);
    }
    
    Vec3 result = emission;
    
    Ray scattered;
    Vec3 attenuation = albedo;
    
    // Glass/Transparent material handling
    if (transmission > 0.0f) {
        if (randf() >= transmission) {
            // Diffuse path
            Vec3 normal = hit.normal;
            if (Vec3::dot(ray.d, normal) > 0) {
                normal = normal * -1.0f;
            }
            Vec3 scatterDir = randomCosineDirection(normal);
            scattered.o = hit.point + normal * 0.001f;
            scattered.d = scatterDir;
        } else {
            // Glass path
            bool frontFace = Vec3::dot(ray.d, hit.normal) < 0;
            Vec3 n = frontFace ? hit.normal : hit.normal * -1.0f;
            float refraction_ratio = frontFace ? (1.0f / ior) : ior;
            
            Vec3 unit_direction = ray.d;
            unit_direction.normalize();
            
            float cos_theta = std::min(-Vec3::dot(unit_direction, n), 1.0f);
            float sin_theta = std::sqrt(1.0f - cos_theta * cos_theta);
            
            bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
            
            float reflectance;
            if (cannot_refract) {
                reflectance = 1.0f;
            } else {
                float cos_for_fresnel = cos_theta;
                if (!frontFace) {
                    float sin_refracted = refraction_ratio * sin_theta;
                    cos_for_fresnel = std::sqrt(1.0f - sin_refracted * sin_refracted);
                }
                float r0 = (1.0f - ior) / (1.0f + ior);
                r0 = r0 * r0;
                reflectance = r0 + (1.0f - r0) * std::pow((1.0f - cos_for_fresnel), 5.0f);
            }
            
            Vec3 direction;
            if (cannot_refract || randf() < reflectance) {
                direction = reflect(unit_direction, n);
                scattered.o = hit.point + n * 0.001f;
            } else {
                direction = refract(unit_direction, n, refraction_ratio);
                scattered.o = hit.point - n * 0.001f;
            }
            
            scattered.d = direction;
            attenuation = Vec3(1.0f, 1.0f, 1.0f);
        }
    } else if (metallic > 0.0f) {
        // Metallic material
        Vec3 normal = hit.normal;
        if (Vec3::dot(ray.d, normal) > 0) {
            normal = normal * -1.0f;
        }
        
        Vec3 unit_direction = ray.d;
        unit_direction.normalize();
        
        Vec3 reflected = reflect(unit_direction, normal);
        
        if (roughness > 0.001f) {
            Vec3 random_scatter = randomUnitVector() * roughness;
            reflected = reflected + random_scatter;
            reflected.normalize();
            if (Vec3::dot(reflected, normal) < 0) {
                reflected = reflect(unit_direction, normal);
            }
        }
        
        scattered.o = hit.point + normal * 0.001f;
        scattered.d = reflected;
        
        if (randf() < metallic) {
            attenuation = albedo;
        } else {
            Vec3 scatterDir = randomCosineDirection(normal);
            scattered.d = scatterDir;
            attenuation = albedo;
        }
    } else {
        // Diffuse material
        Vec3 normal = hit.normal;
        if (Vec3::dot(ray.d, normal) > 0) {
            normal = normal * -1.0f;
        }
        Vec3 scatterDir = randomCosineDirection(normal);
        scattered.o = hit.point + normal * 0.001f;
        scattered.d = scatterDir;
    }
    
    Vec3 incomingLight = trace(scene, scattered, depth - 1, useAABB);
    result = result + attenuation * incomingLight;
    
    if (rrProbability < 1.0f) {
        result = result * (1.0f / rrProbability);
    }
    
    return result;
}
