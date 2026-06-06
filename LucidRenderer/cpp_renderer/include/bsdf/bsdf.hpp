/**
 * bsdf.hpp - BSDF Evaluation and Sampling (Unit-Safe)
 * ====================================================
 * 
 * BSDF implementation with render:: namespace types:
 * - BSDFRGB [1/sr] for reflectance per solid angle
 * - PdfW [1/sr] for probability densities
 * - Direction for all direction vectors
 * - AttenuationRGB (dimensionless [0,1]) for material coefficients
 * - ThroughputRGB (dimensionless [0,∞)) for path weights
 */

#pragma once
#include "../units/render_units.hpp"
#include "../math/random.hpp"
#include "../core/ray.hpp"
#include "fresnel.hpp"
#include "ggx.hpp"

#include <variant>

// ========== Material Parameters ==========

struct MaterialParams {
    render::AttenuationRGB albedo;   // Base color (dimensionless [0,1])
    float metallic;
    float roughness;
    float transmission;
    float ior;
    
    MaterialParams() 
        : albedo(0.8f, 0.8f, 0.8f)
        , metallic(0.0f)
        , roughness(0.5f)
        , transmission(0.0f)
        , ior(1.45f) {}
};

// ========== BSDF Sample Result ==========

/**
 * Evaluated BSDF: explicit f and pdf, throughput weight computed as f × |cosθ| / pdf.
 */
struct EvaluatedBSDF {
    render::BSDFRGB f;     // BSDF value [1/sr]
    render::PdfW    pdf;   // Probability density [1/sr]
};

/**
 * Precomputed weight: f × |cosθ| / pdf already absorbed (e.g. GGX VNDF transmission,
 * near-delta specular). MIS treats this as a delta-like sample (light-sampling PDF
 * is unreliable), so the next-bounce MIS weight is forced to 1.0.
 */
struct PrecomputedWeight {
    render::ThroughputRGB weight;  // Direct throughput = f × |NdotL| / pdf
};

/**
 * BSDFSample - Result of sampling a direction from BSDF (unit-safe).
 *
 * The variant alternative encodes how the sample contributes to throughput:
 *   - EvaluatedBSDF: standard f/pdf path, MIS uses real pdf.
 *   - PrecomputedWeight: delta-like, MIS uses 0 pdf (weight = 1.0 against light sampling).
 */
struct BSDFSample {
    render::Direction wi;                                    // Sampled direction
    std::variant<EvaluatedBSDF, PrecomputedWeight> result;   // Throughput contribution
    bool isDelta;                                            // True for pure delta distributions

    enum Type { DIFFUSE, SPECULAR, TRANSMISSION } type;

    BSDFSample()
        : wi(render::direction_from_unit_vector(render::Vec3f(0.0f, 0.0f, 1.0f)))
        , result(EvaluatedBSDF{render::zero_bsdf_rgb(), render::zero_pdf_w()})
        , isDelta(false)
        , type(DIFFUSE) {}

    [[nodiscard]] bool isValid() const {
        return std::visit([](auto const& r) {
            using R = std::decay_t<decltype(r)>;
            if constexpr (std::is_same_v<R, PrecomputedWeight>) return true;
            else return r.pdf > render::MIN_PDF;
        }, result);
    }

    [[nodiscard]] bool hasPrecomputedWeight() const {
        return std::holds_alternative<PrecomputedWeight>(result);
    }
};

// Compute throughput delta for a sample; returns nullopt to terminate the path.
inline std::optional<render::ThroughputRGB>
compute_throughput_update(const BSDFSample& s, render::Direction sampleNormal) {
    return std::visit([&](auto const& r) -> std::optional<render::ThroughputRGB> {
        using R = std::decay_t<decltype(r)>;
        if constexpr (std::is_same_v<R, PrecomputedWeight>) {
            return r.weight;
        } else {
            float c = std::abs(render::dot(sampleNormal, s.wi));
            if (c <= 1e-6f || r.pdf < render::MIN_PDF) return std::nullopt;
            return render::bsdf_sample_weight(r.f, c, r.pdf);
        }
    }, s.result);
}

// PDF value to store as lastBsdfPdf for next-bounce MIS.
// PrecomputedWeight is delta-like: returning 0 makes the next MIS weight = 1.0.
inline render::PdfW mis_pdf_for_next_bounce(const BSDFSample& s) {
    if (auto* eval = std::get_if<EvaluatedBSDF>(&s.result)) {
        return eval->pdf;
    }
    return render::zero_pdf_w();
}

// Compact layout, trivially destructible — variant<2 alternatives> codegen check.
static_assert(sizeof(BSDFSample) <= 48,
              "BSDFSample should fit in a single cache line section");
static_assert(std::is_trivially_destructible_v<BSDFSample>,
              "BSDFSample must be trivially destructible to keep hot path cheap");

// ========== BSDF Evaluation ==========

/**
 * Evaluate Lambertian diffuse BSDF [1/sr]
 */
inline render::BSDFRGB evalDiffuse(const MaterialParams& mat, float NdotL, float NdotV) {
    if (NdotL <= 0.0f || NdotV <= 0.0f) return render::zero_bsdf_rgb();
    // albedo / π → [1/sr] (uses boundary helper)
    return render::diffuse_bsdf_from_albedo(mat.albedo);
}

/**
 * Evaluate GGX specular BSDF [1/sr]
 */
inline render::BSDFRGB evalSpecular(const MaterialParams& mat, const render::Direction& wo, 
                                const render::Direction& wi, const render::Direction& n, 
                                float NdotL, float NdotV) {
    if (NdotL <= 0.0f || NdotV <= 0.0f) return render::zero_bsdf_rgb();

    auto h = render::half_vector_or_default(wo, wi);
    float NdotH = std::max(render::dot(n, h), 0.0f);
    float VdotH = std::max(render::dot(wo, h), 0.0f);

    float roughness = std::max(mat.roughness, MIN_ROUGHNESS);
    float D = ggxD(NdotH, roughness);
    
    float alpha = roughness * roughness;
    float a2 = alpha * alpha;
    float lambdaL = ggx_safe_sqrt(a2 + (1.0f - a2) * NdotL * NdotL);
    float lambdaV = ggx_safe_sqrt(a2 + (1.0f - a2) * NdotV * NdotV);
    float G2_over_denom = 0.5f / (NdotV * lambdaL + NdotL * lambdaV + GGX_EPSILON);
    
    // Compute F0 using RGB3f (plain floats) — boundary into Fresnel math
    render::RGB3f f0 = render::to_rgb3f(mat.albedo) * mat.metallic +
                       render::RGB3f{0.04f, 0.04f, 0.04f} * (1.0f - mat.metallic);
    render::RGB3f F = fresnelSchlickColor(VdotH, f0);
    
    float spec = D * G2_over_denom;
    // BSDFRGB * RGB3f -> BSDFRGB (modulated by Fresnel)
    return render::BSDFRGB{spec, spec, spec} * F;
}

/**
 * Evaluate combined BSDF [1/sr]
 */
inline render::BSDFRGB evalBSDF(const MaterialParams& mat, const render::Direction& wo, 
                            const render::Direction& wi, const render::Direction& n) {
    float NdotL = render::dot(n, wi);
    float NdotV = render::dot(n, wo);

    if (NdotL <= 0.0f || NdotV <= 0.0f) return render::zero_bsdf_rgb();

    render::BSDFRGB specular = evalSpecular(mat, wo, wi, n, NdotL, NdotV);

    auto h = render::half_vector_or_default(wo, wi);
    float VdotH = std::max(render::dot(wo, h), 0.0f);
    
    // Use RGB3f for Fresnel calculation (plain floats)
    render::RGB3f f0{0.04f, 0.04f, 0.04f};
    render::RGB3f F = fresnelSchlickColor(VdotH, f0);
    
    render::BSDFRGB diffuse = evalDiffuse(mat, NdotL, NdotV);
    render::RGB3f kd{
        (1.0f - F.r) * (1.0f - mat.metallic),
        (1.0f - F.g) * (1.0f - mat.metallic),
        (1.0f - F.b) * (1.0f - mat.metallic)};
    diffuse = diffuse * kd;  // BSDFRGB * RGB3f -> BSDFRGB
    
    return diffuse + specular;  // BSDFRGB + BSDFRGB -> BSDFRGB
}

// ========== BSDF Sampling ==========

/**
 * Sample BSDF direction based on material properties
 * Returns BSDFSample with render:: typed members
 */
inline BSDFSample sampleBSDF(const MaterialParams& mat, const render::Direction& wo, const render::Direction& n, 
                              float u1, float u2, float u3) {
    BSDFSample sample;
    
    float roughness = std::max(mat.roughness, MIN_ROUGHNESS);
    
    // Handle glass/transmission
    if (mat.transmission > 0.0f && u3 < mat.transmission) {
        float wo_dot_n = render::dot(wo, n);
        bool frontFace = wo_dot_n > 0.0f;
        render::Direction faceNormal = frontFace ? n : -n;
        float eta = frontFace ? (1.0f / mat.ior) : mat.ior;

        // Sample microfacet normal using VNDF
        render::Direction t, b;
        buildOrthonormalBasis(faceNormal, t, b);
        render::Direction h = sampleGGXVNDF(wo, roughness, u1, u2, faceNormal, t, b);

        float cosThetaI = std::abs(render::dot(wo, h));
        float F = fresnelDielectric(cosThetaI, eta);

        render::Direction incident = -wo;

        if (randf() < F) {
            // Fresnel reflection
            sample.wi = render::reflect(incident, h.as_normal());

            float NdotL = render::dot(faceNormal, sample.wi);
            float NdotV = render::dot(faceNormal, wo);
            if (NdotL <= 0.0f || NdotV <= 0.0f) {
                sample.result = EvaluatedBSDF{render::zero_bsdf_rgb(), render::zero_pdf_w()};
                return sample;
            }

            float G2 = ggxG2(NdotL, NdotV, roughness);
            float G1 = ggxG1(NdotV, roughness);
            float w = G2 / (G1 + GGX_EPSILON);

            sample.result = PrecomputedWeight{render::make_throughput_rgb(w, w, w)};
            sample.isDelta = false;
            sample.type = BSDFSample::SPECULAR;
        } else {
            // Refraction
            auto refracted_opt = render::refract(incident, h.as_normal(), eta);
            if (!refracted_opt) {
                // Total internal reflection
                sample.wi = render::reflect(incident, h.as_normal());

                float NdotL = render::dot(faceNormal, sample.wi);
                float NdotV = render::dot(faceNormal, wo);
                if (NdotL <= 0.0f || NdotV <= 0.0f) {
                    sample.result = EvaluatedBSDF{render::zero_bsdf_rgb(), render::zero_pdf_w()};
                    return sample;
                }

                float G2 = ggxG2(NdotL, NdotV, roughness);
                float G1 = ggxG1(NdotV, roughness);
                float w = G2 / (G1 + GGX_EPSILON);

                sample.result = PrecomputedWeight{render::make_throughput_rgb(w, w, w)};
                sample.isDelta = false;
                sample.type = BSDFSample::SPECULAR;
            } else {
                sample.wi = *refracted_opt;

                float NdotL = std::abs(render::dot(sample.wi, faceNormal));
                float NdotV = std::abs(render::dot(wo, faceNormal));

                float G2 = ggxG2(NdotL, NdotV, roughness);
                float G1 = ggxG1(NdotV, roughness);
                float w = G2 / (G1 + GGX_EPSILON);

                sample.result = PrecomputedWeight{render::make_throughput_rgb(w, w, w)};
                sample.isDelta = false;
                sample.type = BSDFSample::TRANSMISSION;
            }
        }
        return sample;
    }
    
    // Opaque material: diffuse + specular
    float specProb;
    if (mat.metallic > 0.99f) {
        specProb = 1.0f;
    } else {
        specProb = 0.5f * (1.0f + mat.metallic) * (1.0f - mat.roughness * 0.5f);
        specProb = std::clamp(specProb, 0.1f, 0.9f);
    }
    
    render::Direction t, b;
    buildOrthonormalBasis(n, t, b);
    float NdotV = std::max(render::dot(n, wo), GGX_EPSILON);

    if (u1 < specProb) {
        // Specular sampling using GGX VNDF
        render::Direction h = sampleGGXVNDF(wo, roughness, u1 / specProb, u2, n, t, b);
        sample.wi = render::reflect(-wo, h.as_normal());

        float NdotL = render::dot(n, sample.wi);
        if (NdotL <= 0.0f) {
            sample.result = EvaluatedBSDF{render::zero_bsdf_rgb(), render::zero_pdf_w()};
            return sample;
        }

        render::BSDFRGB f = evalBSDF(mat, wo, sample.wi, n);

        float pdfSpec = pdfGGXVNDF(wo, h, roughness, n);
        float pdfDiff = pdfCosineHemisphere(NdotL);
        float pdf_raw = (specProb * pdfSpec + (1.0f - specProb) * pdfDiff) * (1.0f - mat.transmission);
        render::PdfW pdf = pdf_raw * render::per_sr;

        // For near-delta specular (very low roughness), use weight-based sampling.
        // This ensures MIS weight = 1.0 because light sampling cannot efficiently
        // sample the narrow specular lobe.
        constexpr float DELTA_ROUGHNESS_THRESHOLD = 0.05f;
        if (roughness < DELTA_ROUGHNESS_THRESHOLD && mat.metallic > 0.5f) {
            sample.result = PrecomputedWeight{render::bsdf_sample_weight(f, NdotL, pdf)};
        } else {
            sample.result = EvaluatedBSDF{f, pdf};
        }
        sample.isDelta = false;
        sample.type = BSDFSample::SPECULAR;
    } else {
        // Diffuse sampling
        float u1Adj = (u1 - specProb) / (1.0f - specProb);
        sample.wi = sampleCosineHemisphere(u1Adj, u2, n);

        float NdotL = render::dot(n, sample.wi);
        if (NdotL <= 0.0f) {
            sample.result = EvaluatedBSDF{render::zero_bsdf_rgb(), render::zero_pdf_w()};
            return sample;
        }

        render::BSDFRGB f = evalBSDF(mat, wo, sample.wi, n);

        auto h = render::half_vector_or_default(wo, sample.wi);
        float pdfSpec = pdfGGXVNDF(wo, h, roughness, n);
        float pdfDiff = pdfCosineHemisphere(NdotL);
        float pdf_raw = (specProb * pdfSpec + (1.0f - specProb) * pdfDiff) * (1.0f - mat.transmission);
        render::PdfW pdf = pdf_raw * render::per_sr;

        sample.result = EvaluatedBSDF{f, pdf};
        sample.isDelta = false;
        sample.type = BSDFSample::DIFFUSE;
    }

    return sample;
}

// ========== PDF Evaluation ==========

/**
 * Get PDF for a given direction [1/sr]
 */
inline render::PdfW pdfBSDF(const MaterialParams& mat, 
                            const render::Direction& wo, 
                            const render::Direction& wi, 
                            const render::Direction& n) {
    float NdotL = render::dot(n, wi);
    if (NdotL <= 0.0f) return render::PdfW::zero();

    float specProb;
    if (mat.metallic > 0.99f) {
        specProb = 1.0f;
    } else {
        specProb = 0.5f * (1.0f + mat.metallic) * (1.0f - mat.roughness * 0.5f);
        specProb = std::clamp(specProb, 0.1f, 0.9f);
    }

    float roughness = std::max(mat.roughness, MIN_ROUGHNESS);

    auto h = render::half_vector_or_default(wo, wi);

    float pdfSpec = pdfGGXVNDF(wo, h, roughness, n);
    float pdfDiff = pdfCosineHemisphere(NdotL);
    
    float pdf_raw = (specProb * pdfSpec + (1.0f - specProb) * pdfDiff) * (1.0f - mat.transmission);
    return pdf_raw * render::per_sr;
}