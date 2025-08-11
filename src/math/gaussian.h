#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <array>
#include <vector>

namespace SplatRender {

struct Gaussian3D {
    // Spatial properties
    glm::vec3 position;      // World space position
    glm::vec3 scale;         // Scale factors along axes
    glm::quat rotation;      // Orientation quaternion
    
    // Appearance properties
    glm::vec3 color;         // Base RGB color (DC component)
    float opacity;           // Alpha value
    
    // View-dependent appearance (Spherical Harmonics)
    // For RGB channels, up to degree 3
    // Layout: 3 color channels × (1 + 3 + 5 + 7) coefficients = 48 total
    // But typically only first 15 per channel are used (up to degree 2)
    std::array<float, 45> sh_coeffs;  // 15 coefficients × 3 channels
    
    // Constructor
    Gaussian3D();
    
    // Compute 3D covariance matrix from scale and rotation
    glm::mat3 computeCovariance3D() const;
    
    // Evaluate color for given view direction
    glm::vec3 evaluateColor(const glm::vec3& view_direction) const;
};

struct Gaussian2D {
    // Screen space properties
    glm::vec2 center;        // Pixel coordinates
    glm::mat2 cov_2d;        // 2x2 covariance matrix in screen space
    
    // Rendering properties
    glm::vec3 color;         // View-dependent RGB after SH evaluation
    float alpha;             // Opacity after projection
    float depth;             // Z-depth for sorting
    
    // Optimization data
    uint32_t tile_id;        // Which tile(s) this Gaussian affects
    float radius;            // Bounding radius in pixels (3-sigma)
    
    // Compute alpha value at given pixel
    float computeAlpha(const glm::vec2& pixel_pos) const;
    
    // Check if Gaussian affects given pixel (within 3-sigma)
    bool affectsPixel(const glm::vec2& pixel_pos) const;
};

// Utility functions
namespace GaussianUtils {
    // Project 3D Gaussian to 2D screen space
    Gaussian2D projectToScreen(const Gaussian3D& gaussian3d,
                              const glm::mat4& view_matrix,
                              const glm::mat4& projection_matrix,
                              int screen_width,
                              int screen_height);
    
    // Compute 2D covariance from 3D covariance and projection
    glm::mat2 computeCovariance2D(const glm::mat3& cov3d,
                                  const glm::vec3& position,
                                  const glm::mat4& view_matrix,
                                  const glm::mat4& projection_matrix);
    
    // Sort Gaussians by depth (front-to-back)
    void sortByDepth(std::vector<Gaussian2D>& gaussians);
}

} // namespace SplatRender