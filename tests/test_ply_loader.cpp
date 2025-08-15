#include "test_framework.h"
#include "io/ply_loader.h"
#include <fstream>
#include <cstdio>

using namespace SplatRender;
using namespace SplatRender::Test;

// Helper to create test PLY files
void createTestPLYFile(const std::string& filename, bool binary = false) {
    std::ofstream file;
    
    if (binary) {
        file.open(filename, std::ios::binary);
        file << "ply\n";
        file << "format binary_little_endian 1.0\n";
    } else {
        file.open(filename);
        file << "ply\n";
        file << "format ascii 1.0\n";
    }
    
    file << "element vertex 2\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property float scale_0\n";
    file << "property float scale_1\n";
    file << "property float scale_2\n";
    file << "property float rot_0\n";
    file << "property float rot_1\n";
    file << "property float rot_2\n";
    file << "property float rot_3\n";
    file << "property float f_dc_0\n";
    file << "property float f_dc_1\n";
    file << "property float f_dc_2\n";
    file << "property float opacity\n";
    file << "end_header\n";
    
    if (binary) {
        // Write binary data
        float vertex1[] = {
            1.0f, 2.0f, 3.0f,           // position
            -1.0f, -1.5f, -2.0f,        // scale (log)
            1.0f, 0.0f, 0.0f, 0.0f,     // rotation (w,x,y,z)
            0.5f, 0.6f, 0.7f,           // f_dc
            2.0f                        // opacity (logit)
        };
        
        float vertex2[] = {
            -1.0f, -2.0f, -3.0f,        // position
            -0.5f, -1.0f, -1.5f,        // scale (log)
            0.707f, 0.707f, 0.0f, 0.0f, // rotation (w,x,y,z)
            0.8f, 0.9f, 1.0f,           // f_dc
            -1.0f                       // opacity (logit)
        };
        
        file.write(reinterpret_cast<char*>(vertex1), sizeof(vertex1));
        file.write(reinterpret_cast<char*>(vertex2), sizeof(vertex2));
    } else {
        // Write ASCII data
        file << "1.0 2.0 3.0 -1.0 -1.5 -2.0 1.0 0.0 0.0 0.0 0.5 0.6 0.7 2.0\n";
        file << "-1.0 -2.0 -3.0 -0.5 -1.0 -1.5 0.707 0.707 0.0 0.0 0.8 0.9 1.0 -1.0\n";
    }
    
    file.close();
}

void test_ply_loader_ascii() {
    const std::string test_file = "test_ascii.ply";
    createTestPLYFile(test_file, false);
    
    PLYLoader loader;
    std::vector<Gaussian3D> gaussians;
    
    bool result = loader.load(test_file, gaussians);
    ASSERT_TRUE(result);
    ASSERT_EQUAL(gaussians.size(), 2);
    
    // Check first Gaussian
    const Gaussian3D& g1 = gaussians[0];
    ASSERT_VEC_NEAR(g1.position, glm::vec3(1.0f, 2.0f, 3.0f), 1e-6f);
    
    // Scale should be exponential of input
    ASSERT_NEAR(g1.scale.x, std::exp(-1.0f), 1e-6f);
    ASSERT_NEAR(g1.scale.y, std::exp(-1.5f), 1e-6f);
    ASSERT_NEAR(g1.scale.z, std::exp(-2.0f), 1e-6f);
    
    // Rotation should be normalized
    ASSERT_NEAR(glm::length(g1.rotation), 1.0f, 1e-6f);
    
    // SH DC coefficients
    ASSERT_NEAR(g1.sh_coeffs[0], 0.5f, 1e-6f);
    ASSERT_NEAR(g1.sh_coeffs[1], 0.6f, 1e-6f);
    ASSERT_NEAR(g1.sh_coeffs[2], 0.7f, 1e-6f);
    
    // Opacity should be sigmoid of input
    float expected_opacity = 1.0f / (1.0f + std::exp(-2.0f));
    ASSERT_NEAR(g1.opacity, expected_opacity, 1e-6f);
    
    // Clean up
    std::remove(test_file.c_str());
}

void test_ply_loader_binary() {
    const std::string test_file = "test_binary.ply";
    createTestPLYFile(test_file, true);
    
    PLYLoader loader;
    std::vector<Gaussian3D> gaussians;
    
    bool result = loader.load(test_file, gaussians);
    ASSERT_TRUE(result);
    ASSERT_EQUAL(gaussians.size(), 2);
    
    // Check second Gaussian
    const Gaussian3D& g2 = gaussians[1];
    ASSERT_VEC_NEAR(g2.position, glm::vec3(-1.0f, -2.0f, -3.0f), 1e-6f);
    
    // Check rotation normalization
    ASSERT_NEAR(glm::length(g2.rotation), 1.0f, 1e-6f);
    
    // Clean up
    std::remove(test_file.c_str());
}

void test_ply_loader_invalid_file() {
    PLYLoader loader;
    std::vector<Gaussian3D> gaussians;
    
    bool result = loader.load("nonexistent_file.ply", gaussians);
    ASSERT_FALSE(result);
    ASSERT_TRUE(gaussians.empty());
    
    // Error message should be set
    ASSERT_TRUE(!loader.getLastError().empty());
}

void test_ply_loader_invalid_format() {
    const std::string test_file = "test_invalid.ply";
    
    std::ofstream file(test_file);
    file << "not_ply\n";
    file << "format ascii 1.0\n";
    file << "end_header\n";
    file.close();
    
    PLYLoader loader;
    std::vector<Gaussian3D> gaussians;
    
    bool result = loader.load(test_file, gaussians);
    ASSERT_FALSE(result);
    
    // Clean up
    std::remove(test_file.c_str());
}

void test_ply_loader_progress_callback() {
    const std::string test_file = "test_progress.ply";
    
    // Create larger file for progress testing
    std::ofstream file(test_file);
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex 100\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property float scale_0\n";
    file << "property float scale_1\n";
    file << "property float scale_2\n";
    file << "property float rot_0\n";
    file << "property float rot_1\n";
    file << "property float rot_2\n";
    file << "property float rot_3\n";
    file << "property float f_dc_0\n";
    file << "property float f_dc_1\n";
    file << "property float f_dc_2\n";
    file << "property float opacity\n";
    file << "end_header\n";
    
    for (int i = 0; i < 100; ++i) {
        file << "0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.5 0.5 0.5 0.0\n";
    }
    file.close();
    
    PLYLoader loader;
    std::vector<Gaussian3D> gaussians;
    
    // Track progress
    float last_progress = -1.0f;
    int callback_count = 0;
    
    loader.setProgressCallback([&](float progress) {
        ASSERT_TRUE(progress >= 0.0f && progress <= 1.0f);
        ASSERT_TRUE(progress >= last_progress); // Progress should never go backwards
        last_progress = progress;
        callback_count++;
    });
    
    bool result = loader.load(test_file, gaussians);
    ASSERT_TRUE(result);
    ASSERT_TRUE(callback_count > 0); // Progress callback should have been called
    ASSERT_NEAR(last_progress, 1.0f, 1e-6f); // Final progress should be 1.0
    
    // Clean up
    std::remove(test_file.c_str());
}

void test_ply_loader_sh_coefficients() {
    const std::string test_file = "test_sh.ply";
    
    // Create file with SH coefficients
    std::ofstream file(test_file);
    if (!file.is_open()) {
        ASSERT_TRUE(false);
        return;
    }
    
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex 1\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property float scale_0\n";
    file << "property float scale_1\n";
    file << "property float scale_2\n";
    file << "property float rot_0\n";
    file << "property float rot_1\n";
    file << "property float rot_2\n";
    file << "property float rot_3\n";
    file << "property float f_dc_0\n";
    file << "property float f_dc_1\n";
    file << "property float f_dc_2\n";
    file << "property float opacity\n";
    
    // Add some f_rest coefficients
    for (int i = 0; i < 9; ++i) {
        file << "property float f_rest_" << i << "\n";
    }
    
    file << "end_header\n";
    
    // Write vertex with SH coefficients
    file << "0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 ";
    file << "0.1 0.2 0.3 0.0 "; // DC terms
    for (int i = 0; i < 9; ++i) {
        file << (0.4f + i * 0.1f) << " "; // f_rest values
    }
    file << "\n";
    file.close();
    
    PLYLoader loader;
    std::vector<Gaussian3D> gaussians;
    
    bool result = loader.load(test_file, gaussians);
    ASSERT_TRUE(result);
    ASSERT_EQUAL(gaussians.size(), 1);
    
    const Gaussian3D& g = gaussians[0];
    
    // Check DC coefficients
    ASSERT_NEAR(g.sh_coeffs[0], 0.1f, 1e-6f);
    ASSERT_NEAR(g.sh_coeffs[1], 0.2f, 1e-6f);
    ASSERT_NEAR(g.sh_coeffs[2], 0.3f, 1e-6f);
    
    // Check f_rest coefficients (ensure we don't overflow the array)
    for (int i = 0; i < 9 && (3 + i) < 45; ++i) {
        ASSERT_NEAR(g.sh_coeffs[3 + i], 0.4f + i * 0.1f, 1e-6f);
    }
    
    // Clean up
    std::remove(test_file.c_str());
}

void test_ply_real_file() {
    // Test loading the simple test file we created
    PLYLoader loader;
    std::vector<Gaussian3D> gaussians;
    
    bool result = loader.load("../test_data/simple_gaussian.ply", gaussians);
    if (!result) {
        std::cout << "SKIPPED: Test file not found" << std::endl;
        return;
    }
    
    ASSERT_EQUAL(gaussians.size(), 3);
    
    // Check positions
    ASSERT_VEC_NEAR(gaussians[0].position, glm::vec3(0.0f, 0.0f, 0.0f), 1e-6f);
    ASSERT_VEC_NEAR(gaussians[1].position, glm::vec3(1.0f, 0.0f, 0.0f), 1e-6f);
    ASSERT_VEC_NEAR(gaussians[2].position, glm::vec3(0.0f, 1.0f, 0.0f), 1e-6f);
    
    // Check colors (SH DC coefficients)
    ASSERT_VEC_NEAR(glm::vec3(gaussians[0].sh_coeffs[0], gaussians[0].sh_coeffs[1], gaussians[0].sh_coeffs[2]),
                    glm::vec3(1.0f, 0.0f, 0.0f), 1e-6f); // Red
    ASSERT_VEC_NEAR(glm::vec3(gaussians[1].sh_coeffs[0], gaussians[1].sh_coeffs[1], gaussians[1].sh_coeffs[2]),
                    glm::vec3(0.0f, 1.0f, 0.0f), 1e-6f); // Green
    ASSERT_VEC_NEAR(glm::vec3(gaussians[2].sh_coeffs[0], gaussians[2].sh_coeffs[1], gaussians[2].sh_coeffs[2]),
                    glm::vec3(0.0f, 0.0f, 1.0f), 1e-6f); // Blue
}

int main() {
    std::cout << "Running PLY Loader Tests..." << std::endl;
    
    RUN_TEST(test_ply_loader_ascii);
    RUN_TEST(test_ply_loader_binary);
    RUN_TEST(test_ply_loader_invalid_file);
    RUN_TEST(test_ply_loader_invalid_format);
    RUN_TEST(test_ply_loader_progress_callback);
    RUN_TEST(test_ply_loader_sh_coefficients);
    // RUN_TEST(test_ply_real_file);  // Requires test data file
    
    TestFramework::getInstance().printSummary();
    
    return TestFramework::getInstance().getFailureCount() > 0 ? 1 : 0;
}