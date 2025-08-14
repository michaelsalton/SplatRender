#include "math/spherical_harmonics.h"
#include <cmath>

namespace SplatRender {

namespace SphericalHarmonics {

// Constants for SH evaluation
constexpr float C0 = 0.28209479177387814f;  // 1 / (2 * sqrt(pi))
constexpr float C1 = 0.4886025119029199f;   // sqrt(3 / (4 * pi))
constexpr float C2[] = {
    1.0925484305920792f,   // sqrt(15 / (4 * pi))
    -1.0925484305920792f,  // -sqrt(15 / (4 * pi))
    0.31539156525252005f,  // sqrt(5 / (16 * pi))
    -1.0925484305920792f,  // -sqrt(15 / (4 * pi))
    0.5462742152960396f    // sqrt(15 / (16 * pi))
};
constexpr float C3[] = {
    -0.5900435899266435f,  // -sqrt(35 / (2 * pi)) / 4
    2.890611442640554f,    // sqrt(105 / (4 * pi))
    -0.4570457994644658f,  // -sqrt(21 / (32 * pi))
    0.3731763325901154f,   // sqrt(7 / (16 * pi))
    -0.4570457994644658f,  // -sqrt(21 / (32 * pi))
    1.445305721320277f,    // sqrt(105 / (16 * pi))
    -0.5900435899266435f   // -sqrt(35 / (32 * pi))
};

// Degree 0 (constant)
float evaluateSH_l0_m0(const glm::vec3& /*dir*/) {
    return C0;
}

// Degree 1
float evaluateSH_l1_m_neg1(const glm::vec3& dir) {
    return C1 * dir.y;
}

float evaluateSH_l1_m0(const glm::vec3& dir) {
    return C1 * dir.z;
}

float evaluateSH_l1_m_pos1(const glm::vec3& dir) {
    return C1 * dir.x;
}

// Degree 2
float evaluateSH_l2_m_neg2(const glm::vec3& dir) {
    return C2[0] * dir.x * dir.y;
}

float evaluateSH_l2_m_neg1(const glm::vec3& dir) {
    return C2[1] * dir.y * dir.z;
}

float evaluateSH_l2_m0(const glm::vec3& dir) {
    return C2[2] * (2.0f * dir.z * dir.z - dir.x * dir.x - dir.y * dir.y);
}

float evaluateSH_l2_m_pos1(const glm::vec3& dir) {
    return C2[3] * dir.x * dir.z;
}

float evaluateSH_l2_m_pos2(const glm::vec3& dir) {
    return C2[4] * (dir.x * dir.x - dir.y * dir.y);
}

// Degree 3
float evaluateSH_l3_m_neg3(const glm::vec3& dir) {
    return C3[0] * dir.y * (3.0f * dir.x * dir.x - dir.y * dir.y);
}

float evaluateSH_l3_m_neg2(const glm::vec3& dir) {
    return C3[1] * dir.x * dir.y * dir.z;
}

float evaluateSH_l3_m_neg1(const glm::vec3& dir) {
    return C3[2] * dir.y * (4.0f * dir.z * dir.z - dir.x * dir.x - dir.y * dir.y);
}

float evaluateSH_l3_m0(const glm::vec3& dir) {
    return C3[3] * dir.z * (2.0f * dir.z * dir.z - 3.0f * dir.x * dir.x - 3.0f * dir.y * dir.y);
}

float evaluateSH_l3_m_pos1(const glm::vec3& dir) {
    return C3[4] * dir.x * (4.0f * dir.z * dir.z - dir.x * dir.x - dir.y * dir.y);
}

float evaluateSH_l3_m_pos2(const glm::vec3& dir) {
    return C3[5] * dir.z * (dir.x * dir.x - dir.y * dir.y);
}

float evaluateSH_l3_m_pos3(const glm::vec3& dir) {
    return C3[6] * dir.x * (dir.x * dir.x - 3.0f * dir.y * dir.y);
}

// Evaluate all SH basis functions up to degree 2
std::array<float, 9> evaluateSHBasis_deg2(const glm::vec3& dir) {
    std::array<float, 9> basis;
    
    // Degree 0
    basis[0] = evaluateSH_l0_m0(dir);
    
    // Degree 1
    basis[1] = evaluateSH_l1_m_neg1(dir);
    basis[2] = evaluateSH_l1_m0(dir);
    basis[3] = evaluateSH_l1_m_pos1(dir);
    
    // Degree 2
    basis[4] = evaluateSH_l2_m_neg2(dir);
    basis[5] = evaluateSH_l2_m_neg1(dir);
    basis[6] = evaluateSH_l2_m0(dir);
    basis[7] = evaluateSH_l2_m_pos1(dir);
    basis[8] = evaluateSH_l2_m_pos2(dir);
    
    return basis;
}

// Evaluate all SH basis functions up to degree 3
std::array<float, 16> evaluateSHBasis_deg3(const glm::vec3& dir) {
    std::array<float, 16> basis;
    
    // Copy degree 0-2
    auto basis_deg2 = evaluateSHBasis_deg2(dir);
    std::copy(basis_deg2.begin(), basis_deg2.end(), basis.begin());
    
    // Degree 3
    basis[9] = evaluateSH_l3_m_neg3(dir);
    basis[10] = evaluateSH_l3_m_neg2(dir);
    basis[11] = evaluateSH_l3_m_neg1(dir);
    basis[12] = evaluateSH_l3_m0(dir);
    basis[13] = evaluateSH_l3_m_pos1(dir);
    basis[14] = evaluateSH_l3_m_pos2(dir);
    basis[15] = evaluateSH_l3_m_pos3(dir);
    
    return basis;
}

// Evaluate color from SH coefficients
glm::vec3 evaluateColorFromSH(const std::array<float, 45>& coefficients,
                              const glm::vec3& view_direction,
                              int max_degree) {
    // Normalize view direction
    glm::vec3 dir = glm::normalize(view_direction);
    
    // Evaluate basis functions
    std::array<float, 16> basis;
    int num_basis = 0;
    
    if (max_degree == 2) {
        auto basis_deg2 = evaluateSHBasis_deg2(dir);
        std::copy(basis_deg2.begin(), basis_deg2.end(), basis.begin());
        num_basis = 9;
    } else if (max_degree == 3) {
        basis = evaluateSHBasis_deg3(dir);
        num_basis = 16;
    } else {
        // Just DC component
        basis[0] = evaluateSH_l0_m0(dir);
        num_basis = 1;
    }
    
    // Accumulate color from SH coefficients
    glm::vec3 color(0.0f);
    
    // R channel (coefficients 0-14 for degree 2, 0-15 for degree 3)
    for (int i = 0; i < num_basis && i < 15; ++i) {
        color.r += coefficients[i] * basis[i];
    }
    
    // G channel (coefficients 15-29 for degree 2, 15-30 for degree 3)
    for (int i = 0; i < num_basis && i < 15; ++i) {
        color.g += coefficients[15 + i] * basis[i];
    }
    
    // B channel (coefficients 30-44 for degree 2, 30-45 for degree 3)
    for (int i = 0; i < num_basis && i < 15; ++i) {
        color.b += coefficients[30 + i] * basis[i];
    }
    
    // SH can produce negative values, clamp to valid range
    color = glm::max(color, glm::vec3(0.0f));
    
    return color;
}

// Convert from standard SH coefficient ordering to our storage format
void convertSHCoefficients(const float* input_coeffs, 
                          std::array<float, 45>& output_coeffs,
                          int num_coeffs_per_channel) {
    // Input format is typically:
    // [R0, R1, ..., Rn, G0, G1, ..., Gn, B0, B1, ..., Bn]
    // Where n = num_coeffs_per_channel - 1
    
    // Zero initialize
    output_coeffs.fill(0.0f);
    
    // Copy coefficients
    int num_to_copy = std::min(num_coeffs_per_channel, 15);
    
    // R channel
    for (int i = 0; i < num_to_copy; ++i) {
        output_coeffs[i] = input_coeffs[i];
    }
    
    // G channel
    for (int i = 0; i < num_to_copy; ++i) {
        output_coeffs[15 + i] = input_coeffs[num_coeffs_per_channel + i];
    }
    
    // B channel
    for (int i = 0; i < num_to_copy; ++i) {
        output_coeffs[30 + i] = input_coeffs[2 * num_coeffs_per_channel + i];
    }
}

} // namespace SphericalHarmonics

} // namespace SplatRender