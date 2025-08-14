#pragma once

#include <glm/glm.hpp>
#include <array>

namespace SplatRender {

namespace SphericalHarmonics {

// Maximum degree supported (3 for 3D Gaussian Splatting)
constexpr int MAX_DEGREE = 3;

// Number of coefficients per color channel for each degree
// Degree 0: 1 coefficient  (DC term)
// Degree 1: 3 coefficients (directional)
// Degree 2: 5 coefficients 
// Degree 3: 7 coefficients
// Total: 1 + 3 + 5 + 7 = 16, but typically use up to degree 2 (1 + 3 + 5 = 9)
constexpr int COEFFS_PER_CHANNEL_DEG2 = 9;
constexpr int COEFFS_PER_CHANNEL_DEG3 = 16;

// Total coefficients for RGB (3 channels * 15 = 45 for degree 2)
constexpr int TOTAL_COEFFS_DEG2 = COEFFS_PER_CHANNEL_DEG2 * 3;
constexpr int TOTAL_COEFFS_DEG3 = COEFFS_PER_CHANNEL_DEG3 * 3;

// SH basis function evaluation
float evaluateSH_l0_m0(const glm::vec3& dir);

// Degree 1
float evaluateSH_l1_m_neg1(const glm::vec3& dir);
float evaluateSH_l1_m0(const glm::vec3& dir);
float evaluateSH_l1_m_pos1(const glm::vec3& dir);

// Degree 2
float evaluateSH_l2_m_neg2(const glm::vec3& dir);
float evaluateSH_l2_m_neg1(const glm::vec3& dir);
float evaluateSH_l2_m0(const glm::vec3& dir);
float evaluateSH_l2_m_pos1(const glm::vec3& dir);
float evaluateSH_l2_m_pos2(const glm::vec3& dir);

// Degree 3 (optional, for higher quality)
float evaluateSH_l3_m_neg3(const glm::vec3& dir);
float evaluateSH_l3_m_neg2(const glm::vec3& dir);
float evaluateSH_l3_m_neg1(const glm::vec3& dir);
float evaluateSH_l3_m0(const glm::vec3& dir);
float evaluateSH_l3_m_pos1(const glm::vec3& dir);
float evaluateSH_l3_m_pos2(const glm::vec3& dir);
float evaluateSH_l3_m_pos3(const glm::vec3& dir);

// Evaluate all SH basis functions up to given degree
std::array<float, 9> evaluateSHBasis_deg2(const glm::vec3& dir);
std::array<float, 16> evaluateSHBasis_deg3(const glm::vec3& dir);

// Evaluate color from SH coefficients
glm::vec3 evaluateColorFromSH(const std::array<float, 45>& coefficients,
                              const glm::vec3& view_direction,
                              int max_degree = 2);

// Convert from standard SH coefficient ordering to our storage format
void convertSHCoefficients(const float* input_coeffs, 
                          std::array<float, 45>& output_coeffs,
                          int num_coeffs_per_channel);

} // namespace SphericalHarmonics

} // namespace SplatRender