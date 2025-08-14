#define GLM_ENABLE_EXPERIMENTAL
#include "test_framework.h"
#include "math/gaussian.h"
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>

using namespace SplatRender;
using namespace SplatRender::Test;

void test_gaussian3d_constructor() {
    Gaussian3D g;
    
    // Check default values
    ASSERT_VEC_NEAR(g.position, glm::vec3(0.0f), 1e-6f);
    ASSERT_VEC_NEAR(g.scale, glm::vec3(1.0f), 1e-6f);
    ASSERT_VEC_NEAR(g.color, glm::vec3(1.0f), 1e-6f);
    ASSERT_NEAR(g.opacity, 1.0f, 1e-6f);
    
    // Check quaternion is identity (w=1, x=y=z=0)
    ASSERT_NEAR(g.rotation.w, 1.0f, 1e-6f);
    ASSERT_NEAR(g.rotation.x, 0.0f, 1e-6f);
    ASSERT_NEAR(g.rotation.y, 0.0f, 1e-6f);
    ASSERT_NEAR(g.rotation.z, 0.0f, 1e-6f);
    
    // Check SH coefficients - DC terms should match color
    ASSERT_NEAR(g.sh_coeffs[0], 1.0f, 1e-6f);   // R channel DC
    ASSERT_NEAR(g.sh_coeffs[15], 1.0f, 1e-6f);  // G channel DC
    ASSERT_NEAR(g.sh_coeffs[30], 1.0f, 1e-6f);  // B channel DC
    
    // Other coefficients should be zero
    for (int i = 1; i < 15; ++i) {
        ASSERT_NEAR(g.sh_coeffs[i], 0.0f, 1e-6f);      // R channel
        ASSERT_NEAR(g.sh_coeffs[15 + i], 0.0f, 1e-6f); // G channel
        ASSERT_NEAR(g.sh_coeffs[30 + i], 0.0f, 1e-6f); // B channel
    }
}

void test_gaussian3d_covariance_identity() {
    Gaussian3D g;
    // Identity rotation and unit scale should give identity covariance
    glm::mat3 cov = g.computeCovariance3D();
    
    glm::mat3 expected = glm::mat3(1.0f);
    ASSERT_MAT_NEAR(cov, expected, 1e-6f);
}

void test_gaussian3d_covariance_scaled() {
    Gaussian3D g;
    g.scale = glm::vec3(2.0f, 3.0f, 4.0f);
    
    glm::mat3 cov = g.computeCovariance3D();
    
    // With identity rotation, covariance should be diagonal with scale^2
    glm::mat3 expected(0.0f);
    expected[0][0] = 4.0f;   // 2^2
    expected[1][1] = 9.0f;   // 3^2
    expected[2][2] = 16.0f;  // 4^2
    
    ASSERT_MAT_NEAR(cov, expected, 1e-6f);
}

void test_gaussian3d_covariance_rotated() {
    Gaussian3D g;
    // 90 degree rotation around Z axis
    g.rotation = glm::angleAxis(glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    g.scale = glm::vec3(2.0f, 1.0f, 1.0f);
    
    glm::mat3 cov = g.computeCovariance3D();
    
    // After 90 degree rotation, the elongation along X should now be along Y
    ASSERT_NEAR(cov[0][0], 1.0f, 1e-5f);  // Was 4, now 1
    ASSERT_NEAR(cov[1][1], 4.0f, 1e-5f);  // Was 1, now 4
    ASSERT_NEAR(cov[2][2], 1.0f, 1e-5f);  // Unchanged
    
    // Off-diagonals should be near zero for axis-aligned rotation
    ASSERT_NEAR(cov[0][1], 0.0f, 1e-5f);
    ASSERT_NEAR(cov[1][0], 0.0f, 1e-5f);
}

void test_gaussian3d_evaluate_color() {
    Gaussian3D g;
    g.color = glm::vec3(0.5f, 0.7f, 0.3f);
    
    // Update SH coefficients to match color
    g.sh_coeffs[0] = g.color.r;
    g.sh_coeffs[15] = g.color.g;
    g.sh_coeffs[30] = g.color.b;
    
    // For any view direction, with only DC terms, color should be scaled by Y_0_0
    glm::vec3 color1 = g.evaluateColor(glm::vec3(0.0f, 0.0f, 1.0f));
    glm::vec3 color2 = g.evaluateColor(glm::vec3(1.0f, 0.0f, 0.0f));
    glm::vec3 color3 = g.evaluateColor(glm::vec3(0.0f, 1.0f, 0.0f));
    
    // Y_0_0 = 0.282094792 (1 / (2 * sqrt(pi)))
    float y00 = 0.28209479177387814f;
    glm::vec3 expected = g.color * y00;
    
    ASSERT_VEC_NEAR(color1, expected, 1e-5f);
    ASSERT_VEC_NEAR(color2, expected, 1e-5f);
    ASSERT_VEC_NEAR(color3, expected, 1e-5f);
}

void test_gaussian2d_compute_alpha() {
    Gaussian2D g;
    g.center = glm::vec2(100.0f, 100.0f);
    g.alpha = 0.8f;
    
    // Simple isotropic Gaussian
    g.cov_2d = glm::mat2(1.0f);
    
    // At center, should get full alpha
    float alpha_center = g.computeAlpha(g.center);
    ASSERT_NEAR(alpha_center, 0.8f, 1e-5f);
    
    // Far away, should get near zero
    float alpha_far = g.computeAlpha(glm::vec2(200.0f, 200.0f));
    ASSERT_TRUE(alpha_far < 0.01f);
    
    // Check symmetry
    float alpha_right = g.computeAlpha(glm::vec2(101.0f, 100.0f));
    float alpha_left = g.computeAlpha(glm::vec2(99.0f, 100.0f));
    float alpha_up = g.computeAlpha(glm::vec2(100.0f, 101.0f));
    float alpha_down = g.computeAlpha(glm::vec2(100.0f, 99.0f));
    
    ASSERT_NEAR(alpha_right, alpha_left, 1e-5f);
    ASSERT_NEAR(alpha_up, alpha_down, 1e-5f);
    ASSERT_NEAR(alpha_right, alpha_up, 1e-5f);
}

void test_gaussian2d_affects_pixel() {
    Gaussian2D g;
    g.center = glm::vec2(100.0f, 100.0f);
    g.radius = 10.0f;
    
    // Within radius
    ASSERT_TRUE(g.affectsPixel(glm::vec2(105.0f, 100.0f)));
    ASSERT_TRUE(g.affectsPixel(glm::vec2(100.0f, 105.0f)));
    ASSERT_TRUE(g.affectsPixel(glm::vec2(107.0f, 107.0f))); // sqrt(49+49) < 10
    
    // Outside radius
    ASSERT_FALSE(g.affectsPixel(glm::vec2(111.0f, 100.0f)));
    ASSERT_FALSE(g.affectsPixel(glm::vec2(100.0f, 111.0f)));
    ASSERT_FALSE(g.affectsPixel(glm::vec2(108.0f, 108.0f))); // sqrt(64+64) > 10
}

void test_sort_by_depth() {
    std::vector<Gaussian2D> gaussians(3);
    gaussians[0].depth = -5.0f;   // Closest
    gaussians[1].depth = -10.0f;  // Middle
    gaussians[2].depth = -20.0f;  // Farthest
    
    GaussianUtils::sortByDepth(gaussians);
    
    // Should be sorted by depth in descending order
    // -5 > -10 > -20, so -5 (closest) comes first
    ASSERT_NEAR(gaussians[0].depth, -5.0f, 1e-6f);
    ASSERT_NEAR(gaussians[1].depth, -10.0f, 1e-6f);
    ASSERT_NEAR(gaussians[2].depth, -20.0f, 1e-6f);
}

int main() {
    std::cout << "Running Gaussian Tests..." << std::endl;
    
    RUN_TEST(test_gaussian3d_constructor);
    RUN_TEST(test_gaussian3d_covariance_identity);
    RUN_TEST(test_gaussian3d_covariance_scaled);
    RUN_TEST(test_gaussian3d_covariance_rotated);
    RUN_TEST(test_gaussian3d_evaluate_color);
    RUN_TEST(test_gaussian2d_compute_alpha);
    RUN_TEST(test_gaussian2d_affects_pixel);
    RUN_TEST(test_sort_by_depth);
    
    TestFramework::getInstance().printSummary();
    
    return 0;
}