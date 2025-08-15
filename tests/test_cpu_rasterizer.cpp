#include "test_framework.h"
#include "renderer/cpu_rasterizer.h"
#include "core/camera.h"
#include "math/gaussian.h"
#include <glm/glm.hpp>
#include <vector>
#include <cmath>

using namespace SplatRender;
using namespace SplatRender::Test;

void test_cpu_rasterizer_initialization() {
    CPURasterizer rasterizer;
    
    RenderSettings settings;
    settings.width = 800;
    settings.height = 600;
    settings.tile_size = 16;
    
    rasterizer.initialize(settings);
    
    const RenderSettings& current = rasterizer.getSettings();
    ASSERT_EQUAL(current.width, 800);
    ASSERT_EQUAL(current.height, 600);
    ASSERT_EQUAL(current.tile_size, 16);
}

void test_cpu_rasterizer_empty_scene() {
    CPURasterizer rasterizer;
    
    RenderSettings settings;
    settings.width = 100;
    settings.height = 100;
    rasterizer.initialize(settings);
    
    Camera camera;
    std::vector<Gaussian3D> gaussians;
    std::vector<float> output_buffer;
    
    rasterizer.render(gaussians, camera, output_buffer);
    
    // Check buffer is correct size and all zeros
    ASSERT_EQUAL(output_buffer.size(), 100 * 100 * 4);
    
    for (float val : output_buffer) {
        ASSERT_NEAR(val, 0.0f, 1e-6f);
    }
}

void test_cpu_rasterizer_single_gaussian() {
    CPURasterizer rasterizer;
    
    RenderSettings settings;
    settings.width = 100;
    settings.height = 100;
    rasterizer.initialize(settings);
    
    // Create camera looking down -Z axis
    Camera camera(glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(0.0f, 1.0f, 0.0f), -90.0f, 0.0f);
    
    // Create a single Gaussian at origin
    std::vector<Gaussian3D> gaussians;
    Gaussian3D g;
    g.position = glm::vec3(0.0f, 0.0f, 0.0f);
    g.scale = glm::vec3(2.0f, 2.0f, 2.0f);  // Larger scale for better coverage
    g.rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    g.opacity = 1.0f;
    g.color = glm::vec3(1.0f, 0.0f, 0.0f); // Red
    
    // Set DC term in SH coefficients
    // SH coefficient layout: R[0-14], G[15-29], B[30-44]
    g.sh_coeffs[0] = 1.0f;   // R channel DC
    g.sh_coeffs[15] = 0.0f;  // G channel DC
    g.sh_coeffs[30] = 0.0f;  // B channel DC
    
    gaussians.push_back(g);
    
    std::vector<float> output_buffer;
    rasterizer.render(gaussians, camera, output_buffer);
    
    // Check that center pixel has some red color
    int center_x = 50;
    int center_y = 50;
    int pixel_idx = (center_y * 100 + center_x) * 4;
    
    ASSERT_TRUE(output_buffer[pixel_idx] > 0.0f);     // Red channel
    ASSERT_NEAR(output_buffer[pixel_idx + 1], 0.0f, 1e-6f); // Green channel
    ASSERT_NEAR(output_buffer[pixel_idx + 2], 0.0f, 1e-6f); // Blue channel
    ASSERT_TRUE(output_buffer[pixel_idx + 3] > 0.0f); // Alpha channel
    
    // Check stats
    const auto& stats = rasterizer.getStats();
    ASSERT_EQUAL(stats.visible_gaussians, 1);
    ASSERT_EQUAL(stats.culled_gaussians, 0);
}

void test_cpu_rasterizer_culling() {
    CPURasterizer rasterizer;
    
    RenderSettings settings;
    settings.width = 100;
    settings.height = 100;
    settings.enable_culling = true;
    rasterizer.initialize(settings);
    
    Camera camera(glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(0.0f, 1.0f, 0.0f), -90.0f, 0.0f);
    
    std::vector<Gaussian3D> gaussians;
    
    // Gaussian behind camera
    Gaussian3D g1;
    g1.position = glm::vec3(0.0f, 0.0f, 10.0f);
    gaussians.push_back(g1);
    
    // Gaussian far to the side (outside frustum)
    Gaussian3D g2;
    g2.position = glm::vec3(100.0f, 0.0f, 0.0f);
    gaussians.push_back(g2);
    
    // Gaussian with very low opacity
    Gaussian3D g3;
    g3.position = glm::vec3(0.0f, 0.0f, 0.0f);
    g3.opacity = 0.001f;
    gaussians.push_back(g3);
    
    std::vector<float> output_buffer;
    rasterizer.render(gaussians, camera, output_buffer);
    
    const auto& stats = rasterizer.getStats();
    ASSERT_EQUAL(stats.visible_gaussians, 0);
    ASSERT_TRUE(stats.culled_gaussians >= 3);
}

void test_cpu_rasterizer_tile_assignment() {
    CPURasterizer rasterizer;
    
    RenderSettings settings;
    settings.width = 64;  // 4 tiles
    settings.height = 64; // 4 tiles
    settings.tile_size = 16;
    rasterizer.initialize(settings);
    
    Camera camera(glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(0.0f, 1.0f, 0.0f), -90.0f, 0.0f);
    
    std::vector<Gaussian3D> gaussians;
    
    // Create Gaussians in different tiles
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            Gaussian3D g;
            // Position Gaussians in a grid
            float x = (i - 1.5f) * 0.5f;
            float y = (j - 1.5f) * 0.5f;
            g.position = glm::vec3(x, y, 0.0f);
            g.scale = glm::vec3(1.0f, 1.0f, 1.0f); // Larger Gaussians
            g.opacity = 1.0f;
            // SH coefficient layout: R[0-14], G[15-29], B[30-44]
            g.sh_coeffs[0] = 1.0f;   // R channel DC
            g.sh_coeffs[15] = 1.0f;  // G channel DC
            g.sh_coeffs[30] = 1.0f;  // B channel DC
            gaussians.push_back(g);
        }
    }
    
    std::vector<float> output_buffer;
    rasterizer.render(gaussians, camera, output_buffer);
    
    const auto& stats = rasterizer.getStats();
    ASSERT_EQUAL(stats.visible_gaussians, 16);
    ASSERT_EQUAL(stats.culled_gaussians, 0);
    
    // Check that we have non-zero pixels
    bool hasNonZero = false;
    for (size_t i = 0; i < output_buffer.size(); i += 4) {
        if (output_buffer[i + 3] > 0.0f) { // Check alpha
            hasNonZero = true;
            break;
        }
    }
    ASSERT_TRUE(hasNonZero);
}

void test_cpu_rasterizer_depth_sorting() {
    CPURasterizer rasterizer;
    
    RenderSettings settings;
    settings.width = 100;
    settings.height = 100;
    rasterizer.initialize(settings);
    
    Camera camera(glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(0.0f, 1.0f, 0.0f), -90.0f, 0.0f);
    
    std::vector<Gaussian3D> gaussians;
    
    // Create two overlapping Gaussians at different depths
    // Front Gaussian (red)
    Gaussian3D g1;
    g1.position = glm::vec3(0.0f, 0.0f, 1.0f);
    g1.scale = glm::vec3(2.0f, 2.0f, 2.0f);
    g1.opacity = 0.5f;
    // SH coefficient layout: R[0-14], G[15-29], B[30-44]
    g1.sh_coeffs[0] = 1.0f;   // R channel DC
    g1.sh_coeffs[15] = 0.0f;  // G channel DC
    g1.sh_coeffs[30] = 0.0f;  // B channel DC
    gaussians.push_back(g1);
    
    // Back Gaussian (green)
    Gaussian3D g2;
    g2.position = glm::vec3(0.0f, 0.0f, -1.0f);
    g2.scale = glm::vec3(2.0f, 2.0f, 2.0f);
    g2.opacity = 1.0f;
    // SH coefficient layout: R[0-14], G[15-29], B[30-44]
    g2.sh_coeffs[0] = 0.0f;   // R channel DC
    g2.sh_coeffs[15] = 1.0f;  // G channel DC
    g2.sh_coeffs[30] = 0.0f;  // B channel DC
    gaussians.push_back(g2);
    
    std::vector<float> output_buffer;
    rasterizer.render(gaussians, camera, output_buffer);
    
    // Check center pixel - should be blend of red (front) and green (back)
    int center_x = 50;
    int center_y = 50;
    int pixel_idx = (center_y * 100 + center_x) * 4;
    
    // With front-to-back blending, we should see red contribution
    ASSERT_TRUE(output_buffer[pixel_idx] > 0.0f);     // Red channel
    ASSERT_TRUE(output_buffer[pixel_idx + 1] > 0.0f); // Green channel (from back)
}

void test_cpu_rasterizer_alpha_blending() {
    CPURasterizer rasterizer;
    
    RenderSettings settings;
    settings.width = 10;
    settings.height = 10;
    rasterizer.initialize(settings);
    
    Camera camera(glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(0.0f, 1.0f, 0.0f), -90.0f, 0.0f);
    
    std::vector<Gaussian3D> gaussians;
    
    // Create a semi-transparent white Gaussian
    Gaussian3D g;
    g.position = glm::vec3(0.0f, 0.0f, 0.0f);
    g.scale = glm::vec3(2.0f, 2.0f, 2.0f);
    g.opacity = 0.5f;
    // SH coefficient layout: R[0-14], G[15-29], B[30-44]
    g.sh_coeffs[0] = 1.0f;   // R channel DC
    g.sh_coeffs[15] = 1.0f;  // G channel DC
    g.sh_coeffs[30] = 1.0f;  // B channel DC
    gaussians.push_back(g);
    
    std::vector<float> output_buffer;
    rasterizer.render(gaussians, camera, output_buffer);
    
    // Check center pixel
    int center_x = 5;
    int center_y = 5;
    int pixel_idx = (center_y * 10 + center_x) * 4;
    
    // Alpha should be less than 1.0 due to opacity
    ASSERT_TRUE(output_buffer[pixel_idx + 3] > 0.0f);
    ASSERT_TRUE(output_buffer[pixel_idx + 3] < 1.0f);
}

void test_cpu_rasterizer_performance_stats() {
    CPURasterizer rasterizer;
    
    RenderSettings settings;
    settings.width = 200;
    settings.height = 200;
    rasterizer.initialize(settings);
    
    Camera camera;
    
    // Create many Gaussians
    std::vector<Gaussian3D> gaussians;
    for (int i = 0; i < 100; ++i) {
        Gaussian3D g;
        g.position = glm::vec3(
            (i % 10 - 5) * 0.2f,
            (i / 10 - 5) * 0.2f,
            0.0f
        );
        g.scale = glm::vec3(0.1f, 0.1f, 0.1f);
        g.opacity = 0.8f;
        gaussians.push_back(g);
    }
    
    std::vector<float> output_buffer;
    rasterizer.render(gaussians, camera, output_buffer);
    
    const auto& stats = rasterizer.getStats();
    
    // Check that timing stats are populated
    ASSERT_TRUE(stats.projection_time_ms >= 0.0f);
    ASSERT_TRUE(stats.sorting_time_ms >= 0.0f);
    ASSERT_TRUE(stats.rasterization_time_ms >= 0.0f);
    ASSERT_TRUE(stats.total_time_ms >= 0.0f);
    
    // Total time should be at least sum of parts
    float sum = stats.projection_time_ms + stats.sorting_time_ms + stats.rasterization_time_ms;
    ASSERT_TRUE(stats.total_time_ms >= sum * 0.9f); // Allow 10% margin
}

int main() {
    std::cout << "Running CPU Rasterizer Tests..." << std::endl;
    
    RUN_TEST(test_cpu_rasterizer_initialization);
    RUN_TEST(test_cpu_rasterizer_empty_scene);
    RUN_TEST(test_cpu_rasterizer_single_gaussian);
    RUN_TEST(test_cpu_rasterizer_culling);
    RUN_TEST(test_cpu_rasterizer_tile_assignment);
    RUN_TEST(test_cpu_rasterizer_depth_sorting);
    RUN_TEST(test_cpu_rasterizer_alpha_blending);
    RUN_TEST(test_cpu_rasterizer_performance_stats);
    
    TestFramework::getInstance().printSummary();
    
    return TestFramework::getInstance().getFailureCount() > 0 ? 1 : 0;
}