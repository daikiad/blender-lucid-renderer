/**
 * test_random.cpp - Random Number Generator Tests
 *
 * Quality tests for PCG random number generator:
 * - Distribution uniformity
 * - Statistical tests (chi-square)
 * - Correlation tests
 * - Seeding quality
 */

#include "math/random.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

constexpr float kEps = 1e-4f;

// ============================================================================
// Basic RNG Tests
// ============================================================================

TEST(RNGBasicTest, RandfInRange) {
    seed_random(12345);
    
    for (int i = 0; i < 10000; ++i) {
        float r = randf();
        EXPECT_GE(r, 0.0f);
        EXPECT_LT(r, 1.0f);  // Should be [0, 1)
    }
}

TEST(RNGBasicTest, PCG32_DifferentValues) {
    seed_random(12345);
    
    uint32_t prev = pcg32();
    int consecutiveEqual = 0;
    
    for (int i = 0; i < 1000; ++i) {
        uint32_t curr = pcg32();
        if (curr == prev) {
            consecutiveEqual++;
        }
        prev = curr;
    }
    
    // Extremely unlikely to have many consecutive equal values
    EXPECT_LT(consecutiveEqual, 5);
}

TEST(RNGBasicTest, DifferentSeeds_DifferentSequences) {
    seed_random(111);
    std::vector<float> seq1(100);
    for (int i = 0; i < 100; ++i) seq1[i] = randf();
    
    seed_random(222);
    std::vector<float> seq2(100);
    for (int i = 0; i < 100; ++i) seq2[i] = randf();
    
    // Sequences should differ
    int matches = 0;
    for (int i = 0; i < 100; ++i) {
        if (std::abs(seq1[i] - seq2[i]) < kEps) matches++;
    }
    
    EXPECT_LT(matches, 10);  // Very few matches expected
}

TEST(RNGBasicTest, SameSeed_SameSequence) {
    seed_random(999);
    std::vector<float> seq1(100);
    for (int i = 0; i < 100; ++i) seq1[i] = randf();
    
    seed_random(999);
    std::vector<float> seq2(100);
    for (int i = 0; i < 100; ++i) seq2[i] = randf();
    
    // Sequences should be identical
    for (int i = 0; i < 100; ++i) {
        EXPECT_NEAR(seq1[i], seq2[i], kEps);
    }
}

// ============================================================================
// Distribution Uniformity Tests
// ============================================================================

TEST(RNGUniformityTest, MeanApproximately0_5) {
    seed_random(12345);
    
    double sum = 0.0;
    const int n = 100000;
    
    for (int i = 0; i < n; ++i) {
        sum += randf();
    }
    
    double mean = sum / n;
    // Mean should be close to 0.5 for uniform [0,1)
    EXPECT_NEAR(mean, 0.5, 0.01);
}

TEST(RNGUniformityTest, VarianceCorrect) {
    seed_random(54321);
    
    const int n = 100000;
    std::vector<float> samples(n);
    
    for (int i = 0; i < n; ++i) {
        samples[i] = randf();
    }
    
    // Compute mean
    double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / n;
    
    // Compute variance
    double variance = 0.0;
    for (float s : samples) {
        double diff = s - mean;
        variance += diff * diff;
    }
    variance /= n;
    
    // Variance of U(0,1) is 1/12 ≈ 0.0833
    EXPECT_NEAR(variance, 1.0 / 12.0, 0.005);
}

TEST(RNGUniformityTest, ChiSquareTest) {
    seed_random(11111);
    
    const int numBins = 10;
    const int numSamples = 100000;
    std::vector<int> bins(numBins, 0);
    
    for (int i = 0; i < numSamples; ++i) {
        float r = randf();
        int bin = static_cast<int>(r * numBins);
        if (bin >= numBins) bin = numBins - 1;  // Handle edge case
        bins[bin]++;
    }
    
    // Expected count per bin for uniform distribution
    double expected = numSamples / static_cast<double>(numBins);
    
    // Compute chi-square statistic
    double chiSquare = 0.0;
    for (int count : bins) {
        double diff = count - expected;
        chiSquare += (diff * diff) / expected;
    }
    
    // Chi-square critical value for df=9, p=0.01 is ~21.67
    // Chi-square critical value for df=9, p=0.99 is ~2.09
    // We use a generous range to avoid flaky tests
    EXPECT_LT(chiSquare, 30.0);  // Should not be too high (non-uniform)
    EXPECT_GT(chiSquare, 1.0);   // Should not be too low (suspicious)
}

// ============================================================================
// Bit Distribution Tests
// ============================================================================

TEST(RNGBitDistributionTest, AllBitsUsed) {
    seed_random(22222);
    
    uint32_t orAccum = 0;
    uint32_t andAccum = 0xFFFFFFFF;
    
    for (int i = 0; i < 10000; ++i) {
        uint32_t r = pcg32();
        orAccum |= r;
        andAccum &= r;
    }
    
    // All bits should be set at least once (OR accumulates to all 1s)
    EXPECT_EQ(orAccum, 0xFFFFFFFF);
    // All bits should be clear at least once (AND accumulates to all 0s)
    EXPECT_EQ(andAccum, 0x00000000);
}

TEST(RNGBitDistributionTest, EachBitApproximately50Percent) {
    seed_random(33333);
    
    const int numSamples = 50000;
    std::vector<int> bitCounts(32, 0);
    
    for (int i = 0; i < numSamples; ++i) {
        uint32_t r = pcg32();
        for (int bit = 0; bit < 32; ++bit) {
            if (r & (1u << bit)) {
                bitCounts[bit]++;
            }
        }
    }
    
    // Each bit should be set approximately 50% of the time
    for (int bit = 0; bit < 32; ++bit) {
        double ratio = bitCounts[bit] / static_cast<double>(numSamples);
        EXPECT_NEAR(ratio, 0.5, 0.02);  // ±2% tolerance
    }
}

// ============================================================================
// Seeding Quality Tests
// ============================================================================

TEST(RNGSeedingTest, XYZSeeding_DifferentPixels) {
    // Different pixels should produce different sequences
    seed_random_xyz(0, 0, 0);
    float r00 = randf();
    
    seed_random_xyz(1, 0, 0);
    float r10 = randf();
    
    seed_random_xyz(0, 1, 0);
    float r01 = randf();
    
    // All should be different
    EXPECT_NE(r00, r10);
    EXPECT_NE(r00, r01);
    EXPECT_NE(r10, r01);
}

TEST(RNGSeedingTest, XYZSeeding_DifferentSamples) {
    // Different sample indices should produce different sequences
    std::vector<float> samples(10);
    
    for (int s = 0; s < 10; ++s) {
        seed_random_xyz(100, 100, s);
        samples[s] = randf();
    }
    
    // Check uniqueness (should be no duplicates for 10 samples)
    for (int i = 0; i < 10; ++i) {
        for (int j = i + 1; j < 10; ++j) {
            EXPECT_NE(samples[i], samples[j]);
        }
    }
}

TEST(RNGSeedingTest, XYZSeeding_Reproducible) {
    seed_random_xyz(123, 456, 789);
    float r1 = randf();
    float r2 = randf();
    
    seed_random_xyz(123, 456, 789);
    float r1_repeat = randf();
    float r2_repeat = randf();
    
    EXPECT_NEAR(r1, r1_repeat, kEps);
    EXPECT_NEAR(r2, r2_repeat, kEps);
}

// ============================================================================
// Sampling Utility Tests
// ============================================================================

TEST(SamplingUtilityTest, RandomInUnitSphere_InsideSphere) {
    seed_random(44444);
    
    for (int i = 0; i < 100; ++i) {
        render::Direction d = randomInUnitSphere();
        float len2 = d.x() * d.x() + d.y() * d.y() + d.z() * d.z();
        EXPECT_LT(len2, 1.0f + kEps);
    }
}

TEST(SamplingUtilityTest, RandomUnitVector_IsUnitLength) {
    seed_random(55555);
    
    for (int i = 0; i < 100; ++i) {
        render::Direction d = randomUnitVector();
        float len = std::sqrt(d.x() * d.x() + d.y() * d.y() + d.z() * d.z());
        EXPECT_NEAR(len, 1.0f, kEps);
    }
}

TEST(SamplingUtilityTest, RandomUnitVector_UniformDistribution) {
    seed_random(66666);
    
    const int numSamples = 10000;
    int hemisphereTop = 0;
    int hemisphereBottom = 0;
    
    for (int i = 0; i < numSamples; ++i) {
        render::Direction d = randomUnitVector();
        if (d.z() > 0) hemisphereTop++;
        else hemisphereBottom++;
    }
    
    // Should be roughly 50-50 between hemispheres
    double ratio = hemisphereTop / static_cast<double>(numSamples);
    EXPECT_NEAR(ratio, 0.5, 0.03);
}

TEST(SamplingUtilityTest, RandomCosineDirection_InHemisphere) {
    seed_random(77777);
    
    render::Normal normal = render::normal_from_unit_vector(render::Vec3f(0.0f, 0.0f, 1.0f));
    
    for (int i = 0; i < 100; ++i) {
        render::Direction d = randomCosineDirection(normal);
        
        // Should be in upper hemisphere (positive dot with normal)
        float dot = d.z();  // d · (0,0,1)
        EXPECT_GE(dot, -kEps);  // Allow small numerical error
        
        // Should be approximately unit length
        float len = std::sqrt(d.x() * d.x() + d.y() * d.y() + d.z() * d.z());
        EXPECT_NEAR(len, 1.0f, 0.01f);
    }
}

TEST(SamplingUtilityTest, RandomCosineDirection_CosineWeighted) {
    seed_random(88888);
    
    render::Normal normal = render::normal_from_unit_vector(render::Vec3f(0.0f, 0.0f, 1.0f));
    
    const int numSamples = 10000;
    int nearNormal = 0;  // cos(theta) > 0.5 → theta < 60°
    int farFromNormal = 0;
    
    for (int i = 0; i < numSamples; ++i) {
        render::Direction d = randomCosineDirection(normal);
        float cosTheta = std::abs(d.z());
        
        if (cosTheta > 0.5f) nearNormal++;
        else farFromNormal++;
    }
    
    // Cosine-weighted should have more samples near normal
    // For uniform hemisphere: P(cos > 0.5) = 0.5
    // For cosine-weighted: P(cos > 0.5) = 0.75 (integrate cos from 0.5 to 1)
    double nearRatio = nearNormal / static_cast<double>(numSamples);
    EXPECT_GT(nearRatio, 0.65);  // Should be >65% (closer to 75%)
}

// ============================================================================
// Correlation Tests
// ============================================================================

TEST(RNGCorrelationTest, NoObviousSerialCorrelation) {
    seed_random(99999);
    
    const int n = 10000;
    std::vector<float> samples(n);
    
    for (int i = 0; i < n; ++i) {
        samples[i] = randf();
    }
    
    // Compute lag-1 autocorrelation
    double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / n;
    
    double numerator = 0.0;
    double denominator = 0.0;
    
    for (int i = 0; i < n - 1; ++i) {
        numerator += (samples[i] - mean) * (samples[i + 1] - mean);
    }
    for (int i = 0; i < n; ++i) {
        double diff = samples[i] - mean;
        denominator += diff * diff;
    }
    
    double autocorr = numerator / denominator;
    
    // Autocorrelation should be close to 0 for good RNG
    EXPECT_NEAR(autocorr, 0.0, 0.03);
}

TEST(RNGCorrelationTest, NoPairwiseCorrelation) {
    seed_random(10101);
    
    const int n = 5000;
    
    // Generate pairs of consecutive values
    std::vector<float> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = randf();
        y[i] = randf();
    }
    
    // Compute Pearson correlation
    double meanX = std::accumulate(x.begin(), x.end(), 0.0) / n;
    double meanY = std::accumulate(y.begin(), y.end(), 0.0) / n;
    
    double cov = 0.0, varX = 0.0, varY = 0.0;
    for (int i = 0; i < n; ++i) {
        double dx = x[i] - meanX;
        double dy = y[i] - meanY;
        cov += dx * dy;
        varX += dx * dx;
        varY += dy * dy;
    }
    
    double correlation = cov / std::sqrt(varX * varY);
    
    // Should be near zero
    EXPECT_NEAR(correlation, 0.0, 0.05);
}
