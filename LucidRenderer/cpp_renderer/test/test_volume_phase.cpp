// Unit tests for the Henyey-Greenstein phase function.
//
// Checks:
//   - hg_eval is normalized over the sphere for a range of g.
//   - hg_sample produces directions whose cos(theta_wo, wi) distribution
//     matches hg_pdf within a coarse histogram tolerance.

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "math/random.hpp"
#include "volume/phase.hpp"

namespace {

// Monte-Carlo integral of hg_eval over the unit sphere via uniform sampling.
// E[ p(cos_theta) * 4pi ] should be 1.
double mc_normalization(float g, int n_samples) {
    double acc = 0.0;
    for (int i = 0; i < n_samples; ++i) {
        const float z   = 1.0f - 2.0f * randf();
        const float p   = volume::hg_eval(g, z);
        acc += p;
    }
    // pdf = 1/(4pi) under uniform sphere sampling -> integral = E[p] * 4pi
    return (acc / static_cast<double>(n_samples)) * (4.0 * 3.14159265358979323846);
}

}  // namespace

TEST(VolumePhaseHG, NormalizationIsotropic) {
    const double integral = mc_normalization(0.0f, 200000);
    EXPECT_NEAR(integral, 1.0, 0.05);
}

TEST(VolumePhaseHG, NormalizationForward) {
    const double integral = mc_normalization(0.5f, 200000);
    EXPECT_NEAR(integral, 1.0, 0.10);  // slightly looser; HG has long tails
}

TEST(VolumePhaseHG, NormalizationBackward) {
    const double integral = mc_normalization(-0.5f, 200000);
    EXPECT_NEAR(integral, 1.0, 0.10);
}

TEST(VolumePhaseHG, NormalizationStrongForward) {
    const double integral = mc_normalization(0.8f, 400000);
    EXPECT_NEAR(integral, 1.0, 0.15);
}

TEST(VolumePhaseHG, SampleMatchesPdf) {
    // Histogram the sampled cos_theta and compare against analytic pdf.
    const float g = 0.5f;
    const int   N = 100000;
    const int   B = 20;          // bins
    std::vector<int> hist(B, 0);
    const render::Direction wo = render::axis_z();

    for (int i = 0; i < N; ++i) {
        auto [wi, pdf] = volume::hg_sample(g, wo, randf(), randf());
        const float cos_theta = std::clamp(render::dot(wo, wi), -1.0f, 1.0f);
        int bin = static_cast<int>((cos_theta + 1.0f) * 0.5f * B);
        if (bin == B) bin = B - 1;
        ++hist[bin];
    }

    // Expected count per bin: integral over the bin of pdf * 2pi (azimuth)
    // times sample count. Use a Riemann sum: cos_theta width is 2/B.
    for (int b = 0; b < B; ++b) {
        const float c_lo = -1.0f + 2.0f * b / B;
        const float c_hi = -1.0f + 2.0f * (b + 1) / B;
        const float c_mid = 0.5f * (c_lo + c_hi);
        const float pdf_mid = volume::hg_pdf(g, c_mid);  // [1/sr]
        const float expected = pdf_mid * (2.0f * 3.14159265358979323846f) * (c_hi - c_lo) * N;
        const float observed = static_cast<float>(hist[b]);
        // Loose tolerance for sparse bins near anti-forward direction.
        const float tol = std::max(50.0f, 0.30f * expected);
        EXPECT_NEAR(observed, expected, tol)
            << "  bin " << b << " (cos in [" << c_lo << "," << c_hi << "])";
    }
}
