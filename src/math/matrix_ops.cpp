#include "math/matrix_ops.h"
#include <cmath>
#include <algorithm>

namespace SplatRender {

namespace MatrixOps {

// Matrix-matrix multiplication helpers
glm::mat3 multiply3x3(const glm::mat3& a, const glm::mat3& b) {
    return a * b;
}

// Matrix-vector multiplication for 3x2 matrices
glm::vec2 multiply3x2(const glm::mat3x2& matrix, const glm::vec3& vector) {
    glm::vec2 result;
    result.x = matrix[0][0] * vector.x + matrix[1][0] * vector.y + matrix[2][0] * vector.z;
    result.y = matrix[0][1] * vector.x + matrix[1][1] * vector.y + matrix[2][1] * vector.z;
    return result;
}

// Quaternion operations
glm::mat3 quaternionToRotationMatrix(const glm::quat& q) {
    return glm::mat3_cast(q);
}

glm::quat normalizeQuaternion(const glm::quat& q) {
    return glm::normalize(q);
}

// Matrix operations
glm::mat3 transpose3x3(const glm::mat3& m) {
    return glm::transpose(m);
}

glm::mat3 inverse3x3(const glm::mat3& m) {
    return glm::inverse(m);
}

float determinant3x3(const glm::mat3& m) {
    return glm::determinant(m);
}

glm::mat2 transpose2x2(const glm::mat2& m) {
    return glm::transpose(m);
}

glm::mat2 inverse2x2(const glm::mat2& m) {
    float det = determinant2x2(m);
    if (std::abs(det) < 1e-6f) {
        // Return identity if not invertible
        return glm::mat2(1.0f);
    }
    
    glm::mat2 result;
    result[0][0] = m[1][1] / det;
    result[1][1] = m[0][0] / det;
    result[0][1] = -m[0][1] / det;
    result[1][0] = -m[1][0] / det;
    
    return result;
}

float determinant2x2(const glm::mat2& m) {
    return m[0][0] * m[1][1] - m[0][1] * m[1][0];
}

// Eigenvalue computation for symmetric 2x2 matrices
Eigenvalues2D computeEigenvalues2x2(const glm::mat2& m) {
    // For a 2x2 matrix [[a, b], [c, d]]
    // Eigenvalues are solutions to: det(M - λI) = 0
    // Which gives: λ² - (a+d)λ + (ad-bc) = 0
    
    float a = m[0][0];
    float b = m[0][1];
    float c = m[1][0];
    float d = m[1][1];
    
    float trace = a + d;
    float det = a * d - b * c;
    
    // Quadratic formula
    float discriminant = trace * trace - 4.0f * det;
    
    Eigenvalues2D result;
    
    if (discriminant < 0.0f) {
        // Complex eigenvalues (shouldn't happen for covariance matrices)
        result.lambda1 = result.lambda2 = 0.0f;
    } else {
        float sqrt_disc = std::sqrt(discriminant);
        result.lambda1 = 0.5f * (trace + sqrt_disc);
        result.lambda2 = 0.5f * (trace - sqrt_disc);
    }
    
    // Ensure lambda1 >= lambda2
    if (result.lambda1 < result.lambda2) {
        std::swap(result.lambda1, result.lambda2);
    }
    
    return result;
}

// View/Projection matrix helpers
glm::mat4 createViewMatrix(const glm::vec3& eye, const glm::vec3& center, const glm::vec3& up) {
    return glm::lookAt(eye, center, up);
}

glm::mat4 createProjectionMatrix(float fov_radians, float aspect_ratio, float near, float far) {
    return glm::perspective(fov_radians, aspect_ratio, near, far);
}

// Extract 3x3 rotation from 4x4 matrix
glm::mat3 extractRotation(const glm::mat4& m) {
    return glm::mat3(m);
}

// Compute Jacobian of perspective projection
glm::mat3x2 computeProjectionJacobian(const glm::vec3& pos_view, 
                                     float focal_x, 
                                     float focal_y) {
    float z = pos_view.z;
    float z2 = z * z;
    
    // Jacobian matrix for perspective projection
    // Maps 3D view space to 2D screen space
    glm::mat3x2 J;
    
    // dx/dx, dy/dx
    J[0][0] = focal_x / (-z);  // Use -z since camera looks down -Z
    J[0][1] = 0.0f;
    
    // dx/dy, dy/dy
    J[1][0] = 0.0f;
    J[1][1] = focal_y / (-z);  // Use -z since camera looks down -Z
    
    // dx/dz, dy/dz
    J[2][0] = -focal_x * pos_view.x / z2;
    J[2][1] = -focal_y * pos_view.y / z2;
    
    return J;
}

} // namespace MatrixOps

} // namespace SplatRender