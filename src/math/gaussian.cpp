#define GLM_ENABLE_EXPERIMENTAL
#include "math/gaussian.h"
#include "math/spherical_harmonics.h"
#include "math/matrix_ops.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/norm.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace SplatRender {

// Constructor - Initialize with sensible defaults
Gaussian3D::Gaussian3D() 
    : position(0.0f, 0.0f, 0.0f),
      scale(1.0f, 1.0f, 1.0f),
      rotation(1.0f, 0.0f, 0.0f, 0.0f),  // Identity quaternion (w, x, y, z)
      color(1.0f, 1.0f, 1.0f),
      opacity(1.0f) {
    // Initialize spherical harmonics coefficients to zero
    // The first 3 coefficients (DC term) are set to the base color
    std::fill(sh_coeffs.begin(), sh_coeffs.end(), 0.0f);
    
    // Set DC terms (l=0, m=0) for each color channel
    // SH coefficient layout: R[0-14], G[15-29], B[30-44]
    sh_coeffs[0] = color.r;   // R channel DC
    sh_coeffs[15] = color.g;  // G channel DC  
    sh_coeffs[30] = color.b;  // B channel DC
}

// Compute 3D covariance matrix from scale and rotation
// Σ = R * S * S^T * R^T where S is diagonal scale matrix
glm::mat3 Gaussian3D::computeCovariance3D() const {
    // Create scale matrix
    glm::mat3 S = glm::mat3(0.0f);
    S[0][0] = scale.x;
    S[1][1] = scale.y;
    S[2][2] = scale.z;
    
    // Convert quaternion to rotation matrix
    glm::mat3 R = glm::mat3_cast(rotation);
    
    // Compute S * S^T (which is diagonal with scale^2 values)
    glm::mat3 S_squared = glm::mat3(0.0f);
    S_squared[0][0] = scale.x * scale.x;
    S_squared[1][1] = scale.y * scale.y;
    S_squared[2][2] = scale.z * scale.z;
    
    // Compute R * S * S^T * R^T
    glm::mat3 covariance = R * S_squared * glm::transpose(R);
    
    return covariance;
}

// Evaluate color for given view direction using spherical harmonics
glm::vec3 Gaussian3D::evaluateColor(const glm::vec3& view_direction) const {
    // Use spherical harmonics to compute view-dependent color
    // Default to degree 2 (9 coefficients per channel)
    glm::vec3 color = SphericalHarmonics::evaluateColorFromSH(sh_coeffs, view_direction, 2);
    
    
    return color;
}

// Gaussian2D implementation

// Compute alpha value at given pixel position
float Gaussian2D::computeAlpha(const glm::vec2& pixel_pos) const {
    // Compute difference from center
    glm::vec2 diff = pixel_pos - center;
    
    // Compute inverse of covariance matrix
    float det = cov_2d[0][0] * cov_2d[1][1] - cov_2d[0][1] * cov_2d[1][0];
    
    // If determinant is too small, Gaussian is degenerate
    if (std::abs(det) < 1e-6f) {
        return 0.0f;
    }
    
    glm::mat2 inv_cov;
    inv_cov[0][0] = cov_2d[1][1] / det;
    inv_cov[1][1] = cov_2d[0][0] / det;
    inv_cov[0][1] = -cov_2d[0][1] / det;
    inv_cov[1][0] = -cov_2d[1][0] / det;
    
    // Compute Gaussian weight: exp(-0.5 * (x - μ)^T * Σ^(-1) * (x - μ))
    float exponent = -0.5f * glm::dot(diff, inv_cov * diff);
    
    // Clamp to avoid numerical issues
    exponent = std::max(exponent, -10.0f);
    
    // Compute final alpha
    float gaussian_weight = std::exp(exponent);
    return alpha * gaussian_weight;
}

// Check if Gaussian affects given pixel (within 3-sigma radius)
bool Gaussian2D::affectsPixel(const glm::vec2& pixel_pos) const {
    glm::vec2 diff = pixel_pos - center;
    float dist_sq = glm::length2(diff);
    
    // Check if within bounding radius
    return dist_sq <= (radius * radius);
}

namespace GaussianUtils {

// Project 3D Gaussian to 2D screen space
Gaussian2D projectToScreen(const Gaussian3D& gaussian3d,
                          const glm::mat4& view_matrix,
                          const glm::mat4& projection_matrix,
                          int screen_width,
                          int screen_height) {
    Gaussian2D result;
    
    // Transform position to view space
    glm::vec4 pos_view = view_matrix * glm::vec4(gaussian3d.position, 1.0f);
    
    // Check if behind camera
    if (pos_view.z >= 0.0f) {
        result.radius = 0.0f;  // Mark as invisible
        return result;
    }
    
    // Project to clip space
    glm::vec4 pos_clip = projection_matrix * pos_view;
    
    // Perspective division to NDC
    glm::vec3 pos_ndc = glm::vec3(pos_clip) / pos_clip.w;
    
    // Convert to screen space
    result.center.x = (pos_ndc.x * 0.5f + 0.5f) * (screen_width - 1);
    result.center.y = (0.5f - pos_ndc.y * 0.5f) * (screen_height - 1);  // Flip Y
    
    // Store depth for sorting
    result.depth = pos_view.z;
    
    // Compute 2D covariance matrix
    glm::mat3 cov3d = gaussian3d.computeCovariance3D();
    result.cov_2d = computeCovariance2D(cov3d, gaussian3d.position, 
                                       view_matrix, projection_matrix);
    
    // Evaluate color (for now, just use base color)
    result.color = gaussian3d.evaluateColor(-glm::vec3(pos_view));
    result.alpha = gaussian3d.opacity;
    
    // Compute bounding radius (3-sigma)
    // Eigenvalues of 2x2 covariance matrix
    float a = result.cov_2d[0][0];
    float b = result.cov_2d[0][1];
    float c = result.cov_2d[1][1];
    
    float trace = a + c;
    float det = a * c - b * b;
    float discriminant = trace * trace - 4.0f * det;
    
    // Handle numerical precision issues - treat very small negative discriminants as zero
    if (discriminant >= -1e-6f) {
        float sqrt_disc = std::sqrt(std::max(0.0f, discriminant));
        float lambda1 = 0.5f * (trace + sqrt_disc);
        float lambda2 = 0.5f * (trace - sqrt_disc);
        
        // 3-sigma radius based on largest eigenvalue
        result.radius = 3.0f * std::sqrt(std::max(lambda1, lambda2));
    } else {
        result.radius = 0.0f;
    }
    
    return result;
}

// Compute 2D covariance from 3D covariance and projection
glm::mat2 computeCovariance2D(const glm::mat3& cov3d,
                              const glm::vec3& position,
                              const glm::mat4& view_matrix,
                              const glm::mat4& projection_matrix) {
    // Transform position to view space
    glm::vec4 pos_view = view_matrix * glm::vec4(position, 1.0f);
    glm::vec3 pos_view3 = glm::vec3(pos_view);
    
    // Compute Jacobian of projection at this point
    float focal_x = projection_matrix[0][0];
    float focal_y = projection_matrix[1][1];
    
    // Use the matrix ops function to compute Jacobian
    glm::mat3x2 J = MatrixOps::computeProjectionJacobian(pos_view3, focal_x, focal_y);
    
    // Transform covariance to view space
    glm::mat3 view3x3 = glm::mat3(view_matrix);
    glm::mat3 cov_view = view3x3 * cov3d * glm::transpose(view3x3);
    
    // Project to 2D: Σ_2D = J^T * Σ_view * J
    // Use a simpler approach by extracting vectors
    
    // Extract columns of J (which is 3x2)
    glm::vec3 j_col0(J[0][0], J[1][0], J[2][0]);
    glm::vec3 j_col1(J[0][1], J[1][1], J[2][1]);
    
    // Compute J^T * cov_view * J element by element
    glm::mat2 cov2d;
    
    // Element (0,0): j_col0^T * cov_view * j_col0
    glm::vec3 temp0 = cov_view * j_col0;
    cov2d[0][0] = glm::dot(j_col0, temp0);
    
    // Element (0,1) and (1,0): j_col0^T * cov_view * j_col1
    glm::vec3 temp1 = cov_view * j_col1;
    cov2d[0][1] = glm::dot(j_col0, temp1);
    cov2d[1][0] = cov2d[0][1]; // Symmetric
    
    // Element (1,1): j_col1^T * cov_view * j_col1
    cov2d[1][1] = glm::dot(j_col1, temp1);
    
    // Add small epsilon to diagonal for numerical stability
    float epsilon = 0.1f;
    cov2d[0][0] += epsilon;
    cov2d[1][1] += epsilon;
    
    return cov2d;
}

// Sort Gaussians by depth (front-to-back)
void sortByDepth(std::vector<Gaussian2D>& gaussians) {
    std::sort(gaussians.begin(), gaussians.end(),
              [](const Gaussian2D& a, const Gaussian2D& b) {
                  return a.depth > b.depth;  // Larger depth = farther = render first
              });
}

} // namespace GaussianUtils

} // namespace SplatRender