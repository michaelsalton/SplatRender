#include "test_framework.h"
#include "math/matrix_ops.h"
#include <glm/gtc/quaternion.hpp>

using namespace SplatRender;
using namespace SplatRender::Test;

void test_quaternion_to_rotation_identity() {
    glm::quat q(1.0f, 0.0f, 0.0f, 0.0f); // Identity quaternion
    glm::mat3 rot = MatrixOps::quaternionToRotationMatrix(q);
    
    glm::mat3 expected = glm::mat3(1.0f);
    ASSERT_MAT_NEAR(rot, expected, 1e-6f);
}

void test_quaternion_to_rotation_90deg_z() {
    // 90 degree rotation around Z axis
    glm::quat q = glm::angleAxis(glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    glm::mat3 rot = MatrixOps::quaternionToRotationMatrix(q);
    
    // Should rotate X axis to Y axis
    glm::vec3 x_axis(1.0f, 0.0f, 0.0f);
    glm::vec3 rotated = rot * x_axis;
    ASSERT_VEC_NEAR(rotated, glm::vec3(0.0f, 1.0f, 0.0f), 1e-5f);
    
    // Should rotate Y axis to -X axis
    glm::vec3 y_axis(0.0f, 1.0f, 0.0f);
    rotated = rot * y_axis;
    ASSERT_VEC_NEAR(rotated, glm::vec3(-1.0f, 0.0f, 0.0f), 1e-5f);
}

void test_normalize_quaternion() {
    glm::quat q(2.0f, 0.0f, 0.0f, 0.0f); // Not normalized
    glm::quat normalized = MatrixOps::normalizeQuaternion(q);
    
    float length = glm::length(normalized);
    ASSERT_NEAR(length, 1.0f, 1e-6f);
}

void test_determinant_2x2() {
    glm::mat2 m(1.0f, 2.0f, 3.0f, 4.0f);
    // det = 1*4 - 2*3 = 4 - 6 = -2
    float det = MatrixOps::determinant2x2(m);
    ASSERT_NEAR(det, -2.0f, 1e-6f);
}

void test_inverse_2x2() {
    glm::mat2 m(4.0f, 2.0f, 1.0f, 3.0f);
    glm::mat2 inv = MatrixOps::inverse2x2(m);
    
    // Check that m * inv = identity
    glm::mat2 product = m * inv;
    glm::mat2 identity(1.0f);
    ASSERT_MAT_NEAR(product, identity, 1e-5f);
}

void test_inverse_2x2_singular() {
    // Singular matrix (determinant = 0)
    glm::mat2 m(1.0f, 2.0f, 2.0f, 4.0f);
    glm::mat2 inv = MatrixOps::inverse2x2(m);
    
    // Should return identity for non-invertible matrix
    glm::mat2 identity(1.0f);
    ASSERT_MAT_NEAR(inv, identity, 1e-6f);
}

void test_eigenvalues_2x2_identity() {
    glm::mat2 m(1.0f);
    auto eigenvals = MatrixOps::computeEigenvalues2x2(m);
    
    ASSERT_NEAR(eigenvals.lambda1, 1.0f, 1e-6f);
    ASSERT_NEAR(eigenvals.lambda2, 1.0f, 1e-6f);
}

void test_eigenvalues_2x2_diagonal() {
    glm::mat2 m(0.0f);
    m[0][0] = 3.0f;
    m[1][1] = 5.0f;
    
    auto eigenvals = MatrixOps::computeEigenvalues2x2(m);
    
    ASSERT_NEAR(eigenvals.lambda1, 5.0f, 1e-6f);
    ASSERT_NEAR(eigenvals.lambda2, 3.0f, 1e-6f);
}

void test_eigenvalues_2x2_symmetric() {
    // Symmetric matrix
    glm::mat2 m(4.0f, 1.0f, 1.0f, 4.0f);
    
    auto eigenvals = MatrixOps::computeEigenvalues2x2(m);
    
    // Eigenvalues should be 5 and 3
    ASSERT_NEAR(eigenvals.lambda1, 5.0f, 1e-5f);
    ASSERT_NEAR(eigenvals.lambda2, 3.0f, 1e-5f);
}

void test_projection_jacobian() {
    glm::vec3 pos_view(0.0f, 0.0f, -5.0f); // 5 units in front of camera
    float focal_x = 1.0f;
    float focal_y = 1.0f;
    
    glm::mat3x2 J = MatrixOps::computeProjectionJacobian(pos_view, focal_x, focal_y);
    
    // At center, dx/dx should be focal_x / |z|
    // Since z is negative, we need focal_x / (-z)
    ASSERT_NEAR(J[0][0], 1.0f / 5.0f, 1e-6f);
    ASSERT_NEAR(J[1][1], 1.0f / 5.0f, 1e-6f);
    
    // Off-diagonal should be zero at center
    ASSERT_NEAR(J[0][1], 0.0f, 1e-6f);
    ASSERT_NEAR(J[1][0], 0.0f, 1e-6f);
    
    // dz terms should be zero at center
    ASSERT_NEAR(J[2][0], 0.0f, 1e-6f);
    ASSERT_NEAR(J[2][1], 0.0f, 1e-6f);
}

void test_view_matrix() {
    glm::vec3 eye(0.0f, 0.0f, 5.0f);
    glm::vec3 center(0.0f, 0.0f, 0.0f);
    glm::vec3 up(0.0f, 1.0f, 0.0f);
    
    glm::mat4 view = MatrixOps::createViewMatrix(eye, center, up);
    
    // Transform world origin to view space
    glm::vec4 origin_world(0.0f, 0.0f, 0.0f, 1.0f);
    glm::vec4 origin_view = view * origin_world;
    
    // Should be at (0, 0, -5) in view space
    ASSERT_NEAR(origin_view.x, 0.0f, 1e-6f);
    ASSERT_NEAR(origin_view.y, 0.0f, 1e-6f);
    ASSERT_NEAR(origin_view.z, -5.0f, 1e-6f);
}

void test_projection_matrix() {
    float fov = glm::radians(60.0f);
    float aspect = 16.0f / 9.0f;
    float near = 0.1f;
    float far = 100.0f;
    
    glm::mat4 proj = MatrixOps::createProjectionMatrix(fov, aspect, near, far);
    
    // Test that a point at the center of near plane projects to NDC z=-1
    glm::vec4 near_center(0.0f, 0.0f, -near, 1.0f);
    glm::vec4 projected = proj * near_center;
    projected /= projected.w; // Perspective divide
    
    ASSERT_NEAR(projected.z, -1.0f, 1e-5f);
}

void test_matrix_multiply_3x2() {
    glm::mat3x2 mat;
    mat[0][0] = 1.0f; mat[0][1] = 2.0f;
    mat[1][0] = 3.0f; mat[1][1] = 4.0f;
    mat[2][0] = 5.0f; mat[2][1] = 6.0f;
    
    glm::vec3 vec(1.0f, 2.0f, 3.0f);
    
    glm::vec2 result = MatrixOps::multiply3x2(mat, vec);
    
    // result.x = 1*1 + 3*2 + 5*3 = 1 + 6 + 15 = 22
    // result.y = 2*1 + 4*2 + 6*3 = 2 + 8 + 18 = 28
    ASSERT_NEAR(result.x, 22.0f, 1e-6f);
    ASSERT_NEAR(result.y, 28.0f, 1e-6f);
}

int main() {
    std::cout << "Running Matrix Operations Tests..." << std::endl;
    
    RUN_TEST(test_quaternion_to_rotation_identity);
    RUN_TEST(test_quaternion_to_rotation_90deg_z);
    RUN_TEST(test_normalize_quaternion);
    RUN_TEST(test_determinant_2x2);
    RUN_TEST(test_inverse_2x2);
    RUN_TEST(test_inverse_2x2_singular);
    RUN_TEST(test_eigenvalues_2x2_identity);
    RUN_TEST(test_eigenvalues_2x2_diagonal);
    RUN_TEST(test_eigenvalues_2x2_symmetric);
    RUN_TEST(test_projection_jacobian);
    RUN_TEST(test_view_matrix);
    RUN_TEST(test_projection_matrix);
    RUN_TEST(test_matrix_multiply_3x2);
    
    TestFramework::getInstance().printSummary();
    
    return 0;
}