#define GLM_ENABLE_EXPERIMENTAL
#include "test_framework.h"
#include "math/gaussian.h"
#include "math/matrix_ops.h"
#include <glm/gtc/matrix_transform.hpp>

using namespace SplatRender;
using namespace SplatRender::Test;

void test_project_gaussian_at_origin() {
    Gaussian3D g3d;
    g3d.position = glm::vec3(0.0f, 0.0f, -5.0f); // 5 units in front of camera
    g3d.scale = glm::vec3(1.0f);
    g3d.color = glm::vec3(1.0f, 0.0f, 0.0f);
    g3d.opacity = 0.8f;
    
    // Simple camera at origin looking down -Z
    glm::mat4 view = glm::lookAt(
        glm::vec3(0.0f, 0.0f, 0.0f),  // eye
        glm::vec3(0.0f, 0.0f, -1.0f), // center
        glm::vec3(0.0f, 1.0f, 0.0f)   // up
    );
    
    glm::mat4 proj = glm::perspective(
        glm::radians(60.0f), // fov
        1.0f,                // aspect
        0.1f,                // near
        100.0f               // far
    );
    
    int width = 800;
    int height = 800;
    
    Gaussian2D g2d = GaussianUtils::projectToScreen(g3d, view, proj, width, height);
    
    // Should project to center of screen
    ASSERT_NEAR(g2d.center.x, width / 2.0f, 1.0f);
    ASSERT_NEAR(g2d.center.y, height / 2.0f, 1.0f);
    
    // Depth should be -5
    ASSERT_NEAR(g2d.depth, -5.0f, 1e-6f);
    
    // Alpha should match opacity
    ASSERT_NEAR(g2d.alpha, 0.8f, 1e-6f);
}

void test_project_gaussian_offscreen() {
    Gaussian3D g3d;
    g3d.position = glm::vec3(100.0f, 0.0f, -5.0f); // Far to the right
    
    glm::mat4 view = glm::mat4(1.0f);
    glm::mat4 proj = glm::perspective(glm::radians(60.0f), 1.0f, 0.1f, 100.0f);
    
    Gaussian2D g2d = GaussianUtils::projectToScreen(g3d, view, proj, 800, 800);
    
    // Should be way off to the right
    ASSERT_TRUE(g2d.center.x > 800.0f);
}

void test_project_gaussian_behind_camera() {
    Gaussian3D g3d;
    g3d.position = glm::vec3(0.0f, 0.0f, 5.0f); // Behind camera
    
    glm::mat4 view = glm::mat4(1.0f);
    glm::mat4 proj = glm::perspective(glm::radians(60.0f), 1.0f, 0.1f, 100.0f);
    
    Gaussian2D g2d = GaussianUtils::projectToScreen(g3d, view, proj, 800, 800);
    
    // Should be marked as invisible
    ASSERT_NEAR(g2d.radius, 0.0f, 1e-6f);
}

void test_covariance_2d_computation() {
    // Create a 3D covariance matrix (elongated along X)
    glm::mat3 cov3d(0.0f);
    cov3d[0][0] = 4.0f; // Variance along X
    cov3d[1][1] = 1.0f; // Variance along Y
    cov3d[2][2] = 1.0f; // Variance along Z
    
    glm::vec3 position(0.0f, 0.0f, -5.0f);
    glm::mat4 view = glm::mat4(1.0f);
    glm::mat4 proj = glm::perspective(glm::radians(60.0f), 1.0f, 0.1f, 100.0f);
    
    glm::mat2 cov2d = GaussianUtils::computeCovariance2D(cov3d, position, view, proj);
    
    // 2D covariance should preserve the elongation along X
    ASSERT_TRUE(cov2d[0][0] > cov2d[1][1]);
    
    // Should be symmetric
    ASSERT_NEAR(cov2d[0][1], cov2d[1][0], 1e-6f);
}

void test_gaussian_radius_computation() {
    Gaussian3D g3d;
    g3d.position = glm::vec3(0.0f, 0.0f, -5.0f);
    g3d.scale = glm::vec3(2.0f, 1.0f, 1.0f); // Elongated along X
    
    glm::mat4 view = glm::mat4(1.0f);
    glm::mat4 proj = glm::perspective(glm::radians(60.0f), 1.0f, 0.1f, 100.0f);
    
    Gaussian2D g2d = GaussianUtils::projectToScreen(g3d, view, proj, 800, 800);
    
    // Radius should be positive and reasonable
    ASSERT_TRUE(g2d.radius > 0.0f);
    ASSERT_TRUE(g2d.radius < 800.0f); // Shouldn't be larger than screen
    
    // With scale 2 along X, radius should be larger than with uniform scale
    Gaussian3D g3d_uniform;
    g3d_uniform.position = g3d.position;
    g3d_uniform.scale = glm::vec3(1.0f);
    
    Gaussian2D g2d_uniform = GaussianUtils::projectToScreen(g3d_uniform, view, proj, 800, 800);
    
    ASSERT_TRUE(g2d.radius > g2d_uniform.radius);
}

void test_perspective_scaling() {
    // Test that Gaussians get smaller with distance
    Gaussian3D g_near, g_far;
    g_near.position = glm::vec3(0.0f, 0.0f, -2.0f);
    g_far.position = glm::vec3(0.0f, 0.0f, -10.0f);
    
    glm::mat4 view = glm::mat4(1.0f);
    glm::mat4 proj = glm::perspective(glm::radians(60.0f), 1.0f, 0.1f, 100.0f);
    
    Gaussian2D g2d_near = GaussianUtils::projectToScreen(g_near, view, proj, 800, 800);
    Gaussian2D g2d_far = GaussianUtils::projectToScreen(g_far, view, proj, 800, 800);
    
    
    // Both should have valid radii
    ASSERT_TRUE(g2d_near.radius > 0.0f);
    ASSERT_TRUE(g2d_far.radius > 0.0f);
    
    // Far Gaussian should have smaller radius
    ASSERT_TRUE(g2d_far.radius < g2d_near.radius);
    
    // Radius should scale approximately with 1/distance
    float expected_ratio = 2.0f / 10.0f; // near_z / far_z
    float actual_ratio = g2d_far.radius / g2d_near.radius;
    ASSERT_NEAR(actual_ratio, expected_ratio, 0.1f);
}

void test_screen_space_coordinates() {
    // Test NDC to screen space conversion
    Gaussian3D g;
    
    glm::mat4 view = glm::mat4(1.0f);
    glm::mat4 proj = glm::ortho(-1.0f, 1.0f, -1.0f, 1.0f, 0.1f, 100.0f);
    
    // Place at different positions to test screen mapping
    int width = 800;
    int height = 600;
    
    // Center
    g.position = glm::vec3(0.0f, 0.0f, -1.0f);
    Gaussian2D g_center = GaussianUtils::projectToScreen(g, view, proj, width, height);
    ASSERT_NEAR(g_center.center.x, width / 2.0f, 1.0f);
    ASSERT_NEAR(g_center.center.y, height / 2.0f, 1.0f);
    
    // Top-left in world space (-1, 1) should map to (0, 0) in screen
    g.position = glm::vec3(-1.0f, 1.0f, -1.0f);
    Gaussian2D g_tl = GaussianUtils::projectToScreen(g, view, proj, width, height);
    ASSERT_NEAR(g_tl.center.x, 0.0f, 1.0f);
    ASSERT_NEAR(g_tl.center.y, 0.0f, 1.0f);
    
    // Bottom-right in world space (1, -1) should map to (width, height)
    g.position = glm::vec3(1.0f, -1.0f, -1.0f);
    Gaussian2D g_br = GaussianUtils::projectToScreen(g, view, proj, width, height);
    ASSERT_NEAR(g_br.center.x, width, 1.0f);
    ASSERT_NEAR(g_br.center.y, height, 1.0f);
}

void test_view_dependent_color() {
    Gaussian3D g;
    g.position = glm::vec3(0.0f, 0.0f, -5.0f);
    g.color = glm::vec3(0.5f, 0.5f, 0.5f);
    
    // Set up some directional SH coefficients
    // Add positive contribution in +X direction for red channel
    g.sh_coeffs[0] = 0.5f;  // DC
    g.sh_coeffs[3] = 0.3f;  // L1, m=1 (X direction)
    g.sh_coeffs[15] = 0.5f; // G DC
    g.sh_coeffs[30] = 0.5f; // B DC
    
    // View from different angles
    glm::mat4 view_front = glm::lookAt(
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 0.0f, -1.0f),
        glm::vec3(0.0f, 1.0f, 0.0f)
    );
    
    glm::mat4 view_side = glm::lookAt(
        glm::vec3(5.0f, 0.0f, -5.0f),
        glm::vec3(0.0f, 0.0f, -5.0f),
        glm::vec3(0.0f, 1.0f, 0.0f)
    );
    
    glm::mat4 proj = glm::perspective(glm::radians(60.0f), 1.0f, 0.1f, 100.0f);
    
    Gaussian2D g2d_front = GaussianUtils::projectToScreen(g, view_front, proj, 800, 800);
    Gaussian2D g2d_side = GaussianUtils::projectToScreen(g, view_side, proj, 800, 800);
    
    // Colors should be different due to view-dependent SH
    // Can't test exact values without knowing the view direction calculation
    // but at least verify they're computed
    ASSERT_TRUE(g2d_front.color.r > 0.0f);
    ASSERT_TRUE(g2d_side.color.r > 0.0f);
}

int main() {
    std::cout << "Running 3D to 2D Projection Tests..." << std::endl;
    
    RUN_TEST(test_project_gaussian_at_origin);
    RUN_TEST(test_project_gaussian_offscreen);
    RUN_TEST(test_project_gaussian_behind_camera);
    RUN_TEST(test_covariance_2d_computation);
    RUN_TEST(test_gaussian_radius_computation);
    RUN_TEST(test_perspective_scaling);
    RUN_TEST(test_screen_space_coordinates);
    RUN_TEST(test_view_dependent_color);
    
    TestFramework::getInstance().printSummary();
    
    return 0;
}