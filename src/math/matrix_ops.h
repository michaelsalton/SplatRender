#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace SplatRender {

namespace MatrixOps {

// Matrix-matrix multiplication helpers
glm::mat3 multiply3x3(const glm::mat3& a, const glm::mat3& b);

// Matrix-vector multiplication for 3x2 matrices
glm::vec2 multiply3x2(const glm::mat3x2& matrix, const glm::vec3& vector);

// Quaternion operations
glm::mat3 quaternionToRotationMatrix(const glm::quat& q);
glm::quat normalizeQuaternion(const glm::quat& q);

// Matrix operations
glm::mat3 transpose3x3(const glm::mat3& m);
glm::mat3 inverse3x3(const glm::mat3& m);
float determinant3x3(const glm::mat3& m);

glm::mat2 transpose2x2(const glm::mat2& m);
glm::mat2 inverse2x2(const glm::mat2& m);
float determinant2x2(const glm::mat2& m);

// Eigenvalue computation for symmetric 2x2 matrices
struct Eigenvalues2D {
    float lambda1;
    float lambda2;
};
Eigenvalues2D computeEigenvalues2x2(const glm::mat2& m);

// View/Projection matrix helpers
glm::mat4 createViewMatrix(const glm::vec3& eye, const glm::vec3& center, const glm::vec3& up);
glm::mat4 createProjectionMatrix(float fov_radians, float aspect_ratio, float near, float far);

// Extract 3x3 rotation from 4x4 matrix
glm::mat3 extractRotation(const glm::mat4& m);

// Compute Jacobian of perspective projection
glm::mat3x2 computeProjectionJacobian(const glm::vec3& pos_view, 
                                     float focal_x, 
                                     float focal_y);

} // namespace MatrixOps

} // namespace SplatRender