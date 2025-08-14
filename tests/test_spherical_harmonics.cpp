#include "test_framework.h"
#include "math/spherical_harmonics.h"
#include <cmath>

using namespace SplatRender;
using namespace SplatRender::Test;

void test_sh_degree_0() {
    // Degree 0 should be constant regardless of direction
    glm::vec3 dir1(1.0f, 0.0f, 0.0f);
    glm::vec3 dir2(0.0f, 1.0f, 0.0f);
    glm::vec3 dir3(0.0f, 0.0f, 1.0f);
    
    float val1 = SphericalHarmonics::evaluateSH_l0_m0(dir1);
    float val2 = SphericalHarmonics::evaluateSH_l0_m0(dir2);
    float val3 = SphericalHarmonics::evaluateSH_l0_m0(dir3);
    
    ASSERT_NEAR(val1, val2, 1e-6f);
    ASSERT_NEAR(val2, val3, 1e-6f);
    
    // Check the actual value (1 / (2 * sqrt(pi)))
    float expected = 0.28209479177387814f;
    ASSERT_NEAR(val1, expected, 1e-6f);
}

void test_sh_degree_1() {
    // Degree 1 should be linear in x, y, z
    glm::vec3 x_dir(1.0f, 0.0f, 0.0f);
    glm::vec3 y_dir(0.0f, 1.0f, 0.0f);
    glm::vec3 z_dir(0.0f, 0.0f, 1.0f);
    
    // Y component
    float y_val = SphericalHarmonics::evaluateSH_l1_m_neg1(y_dir);
    ASSERT_TRUE(y_val > 0.0f);
    ASSERT_NEAR(SphericalHarmonics::evaluateSH_l1_m_neg1(x_dir), 0.0f, 1e-6f);
    ASSERT_NEAR(SphericalHarmonics::evaluateSH_l1_m_neg1(z_dir), 0.0f, 1e-6f);
    
    // Z component
    float z_val = SphericalHarmonics::evaluateSH_l1_m0(z_dir);
    ASSERT_TRUE(z_val > 0.0f);
    ASSERT_NEAR(SphericalHarmonics::evaluateSH_l1_m0(x_dir), 0.0f, 1e-6f);
    ASSERT_NEAR(SphericalHarmonics::evaluateSH_l1_m0(y_dir), 0.0f, 1e-6f);
    
    // X component
    float x_val = SphericalHarmonics::evaluateSH_l1_m_pos1(x_dir);
    ASSERT_TRUE(x_val > 0.0f);
    ASSERT_NEAR(SphericalHarmonics::evaluateSH_l1_m_pos1(y_dir), 0.0f, 1e-6f);
    ASSERT_NEAR(SphericalHarmonics::evaluateSH_l1_m_pos1(z_dir), 0.0f, 1e-6f);
}

void test_sh_orthogonality() {
    // Test that different SH basis functions are orthogonal
    // We'll test with a few sample directions
    std::vector<glm::vec3> test_dirs = {
        glm::normalize(glm::vec3(1.0f, 0.0f, 0.0f)),
        glm::normalize(glm::vec3(0.0f, 1.0f, 0.0f)),
        glm::normalize(glm::vec3(0.0f, 0.0f, 1.0f)),
        glm::normalize(glm::vec3(1.0f, 1.0f, 0.0f)),
        glm::normalize(glm::vec3(1.0f, 0.0f, 1.0f)),
        glm::normalize(glm::vec3(0.0f, 1.0f, 1.0f)),
        glm::normalize(glm::vec3(1.0f, 1.0f, 1.0f))
    };
    
    // This is a simplified test - true orthogonality requires integration over sphere
    // But we can at least check that they're not all the same
    for (const auto& dir : test_dirs) {
        float y0 = SphericalHarmonics::evaluateSH_l0_m0(dir);
        float y1_neg1 = SphericalHarmonics::evaluateSH_l1_m_neg1(dir);
        float y1_0 = SphericalHarmonics::evaluateSH_l1_m0(dir);
        float y1_pos1 = SphericalHarmonics::evaluateSH_l1_m_pos1(dir);
        
        // At least some should be different
        bool all_same = (y0 == y1_neg1) && (y0 == y1_0) && (y0 == y1_pos1);
        ASSERT_FALSE(all_same);
    }
}

void test_sh_basis_deg2() {
    glm::vec3 dir = glm::normalize(glm::vec3(1.0f, 2.0f, 3.0f));
    auto basis = SphericalHarmonics::evaluateSHBasis_deg2(dir);
    
    // Should have 9 coefficients
    ASSERT_EQUAL(9, static_cast<int>(basis.size()));
    
    // Check first few match individual functions
    ASSERT_NEAR(basis[0], SphericalHarmonics::evaluateSH_l0_m0(dir), 1e-6f);
    ASSERT_NEAR(basis[1], SphericalHarmonics::evaluateSH_l1_m_neg1(dir), 1e-6f);
    ASSERT_NEAR(basis[2], SphericalHarmonics::evaluateSH_l1_m0(dir), 1e-6f);
    ASSERT_NEAR(basis[3], SphericalHarmonics::evaluateSH_l1_m_pos1(dir), 1e-6f);
}

void test_sh_evaluate_color_dc_only() {
    std::array<float, 45> coeffs;
    coeffs.fill(0.0f);
    
    // Set DC terms only
    coeffs[0] = 0.5f;   // R
    coeffs[15] = 0.7f;  // G
    coeffs[30] = 0.3f;  // B
    
    // Any direction should give same color with DC only
    glm::vec3 color1 = SphericalHarmonics::evaluateColorFromSH(coeffs, glm::vec3(1, 0, 0), 2);
    glm::vec3 color2 = SphericalHarmonics::evaluateColorFromSH(coeffs, glm::vec3(0, 1, 0), 2);
    glm::vec3 color3 = SphericalHarmonics::evaluateColorFromSH(coeffs, glm::vec3(0, 0, 1), 2);
    
    // DC coefficient needs to be multiplied by Y_0_0
    float y00 = SphericalHarmonics::evaluateSH_l0_m0(glm::vec3(1, 0, 0));
    glm::vec3 expected(0.5f * y00, 0.7f * y00, 0.3f * y00);
    
    ASSERT_VEC_NEAR(color1, expected, 1e-5f);
    ASSERT_VEC_NEAR(color2, expected, 1e-5f);
    ASSERT_VEC_NEAR(color3, expected, 1e-5f);
}

void test_sh_evaluate_color_directional() {
    std::array<float, 45> coeffs;
    coeffs.fill(0.0f);
    
    // Add directional component (degree 1, m=1, corresponds to X direction)
    coeffs[0] = 1.0f;   // R DC
    coeffs[3] = 0.5f;   // R degree 1, m=1 (X direction)
    
    glm::vec3 color_x = SphericalHarmonics::evaluateColorFromSH(coeffs, glm::vec3(1, 0, 0), 2);
    glm::vec3 color_neg_x = SphericalHarmonics::evaluateColorFromSH(coeffs, glm::vec3(-1, 0, 0), 2);
    
    // Should be brighter in +X direction
    ASSERT_TRUE(color_x.r > color_neg_x.r);
    
    // G and B should be zero
    ASSERT_NEAR(color_x.g, 0.0f, 1e-5f);
    ASSERT_NEAR(color_x.b, 0.0f, 1e-5f);
}

void test_sh_convert_coefficients() {
    // Test coefficient conversion
    float input[27]; // 9 coeffs per channel
    for (int i = 0; i < 27; ++i) {
        input[i] = static_cast<float>(i);
    }
    
    std::array<float, 45> output;
    SphericalHarmonics::convertSHCoefficients(input, output, 9);
    
    // Check R channel (0-8 -> 0-8)
    for (int i = 0; i < 9; ++i) {
        ASSERT_NEAR(output[i], static_cast<float>(i), 1e-6f);
    }
    
    // Check G channel (9-17 -> 15-23)
    for (int i = 0; i < 9; ++i) {
        ASSERT_NEAR(output[15 + i], static_cast<float>(9 + i), 1e-6f);
    }
    
    // Check B channel (18-26 -> 30-38)
    for (int i = 0; i < 9; ++i) {
        ASSERT_NEAR(output[30 + i], static_cast<float>(18 + i), 1e-6f);
    }
    
    // Rest should be zero
    for (int i = 9; i < 15; ++i) {
        ASSERT_NEAR(output[i], 0.0f, 1e-6f);
        ASSERT_NEAR(output[15 + i], 0.0f, 1e-6f);
        ASSERT_NEAR(output[30 + i], 0.0f, 1e-6f);
    }
}

void test_sh_normalization() {
    // Test that SH basis functions are properly normalized
    // For real spherical harmonics, integral of Y_l_m^2 over sphere = 1
    // We'll do a simple check that the constant term has the right magnitude
    
    float y00 = SphericalHarmonics::evaluateSH_l0_m0(glm::vec3(1, 0, 0));
    
    // Y_0_0 = 1 / (2 * sqrt(pi))
    // Integral over sphere = 4π * Y_0_0^2 = 1
    float integral_estimate = 4.0f * M_PI * y00 * y00;
    ASSERT_NEAR(integral_estimate, 1.0f, 1e-5f);
}

int main() {
    std::cout << "Running Spherical Harmonics Tests..." << std::endl;
    
    RUN_TEST(test_sh_degree_0);
    RUN_TEST(test_sh_degree_1);
    RUN_TEST(test_sh_orthogonality);
    RUN_TEST(test_sh_basis_deg2);
    RUN_TEST(test_sh_evaluate_color_dc_only);
    RUN_TEST(test_sh_evaluate_color_directional);
    RUN_TEST(test_sh_convert_coefficients);
    RUN_TEST(test_sh_normalization);
    
    TestFramework::getInstance().printSummary();
    
    return 0;
}