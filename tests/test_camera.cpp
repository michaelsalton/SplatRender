#include "test_framework.h"
#include "core/camera.h"
#include <cstdio>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/epsilon.hpp>

using namespace SplatRender;
using namespace SplatRender::Test;

void test_camera_constructor() {
    Camera camera;
    
    // Check default position
    glm::vec3 pos = camera.getPosition();
    ASSERT_NEAR(pos.x, 0.0f, 1e-6f);
    ASSERT_NEAR(pos.y, 0.0f, 1e-6f);
    ASSERT_NEAR(pos.z, 3.0f, 1e-6f);
    
    // Check default orientation
    ASSERT_NEAR(camera.getYaw(), -90.0f, 1e-6f);
    ASSERT_NEAR(camera.getPitch(), 0.0f, 1e-6f);
    
    // Check default settings
    ASSERT_NEAR(camera.getFOV(), 60.0f, 1e-6f);
    ASSERT_NEAR(camera.getNearPlane(), 0.1f, 1e-6f);
    ASSERT_NEAR(camera.getFarPlane(), 1000.0f, 1e-6f);
}

void test_camera_custom_constructor() {
    glm::vec3 position(5.0f, 10.0f, 15.0f);
    glm::vec3 up(0.0f, 1.0f, 0.0f);
    float yaw = -45.0f;
    float pitch = 30.0f;
    
    Camera camera(position, up, yaw, pitch);
    
    glm::vec3 pos = camera.getPosition();
    ASSERT_NEAR(pos.x, 5.0f, 1e-6f);
    ASSERT_NEAR(pos.y, 10.0f, 1e-6f);
    ASSERT_NEAR(pos.z, 15.0f, 1e-6f);
    
    ASSERT_NEAR(camera.getYaw(), -45.0f, 1e-6f);
    ASSERT_NEAR(camera.getPitch(), 30.0f, 1e-6f);
}

void test_camera_movement() {
    Camera camera;
    camera.setPosition(glm::vec3(0.0f, 0.0f, 0.0f));
    
    float delta_time = 0.1f;
    
    // Test forward movement
    camera.processKeyboard(CameraMovement::FORWARD, delta_time);
    glm::vec3 pos = camera.getPosition();
    ASSERT_TRUE(pos.z < 0.0f); // Should move in -Z direction (forward)
    
    // Test backward movement
    camera.setPosition(glm::vec3(0.0f, 0.0f, 0.0f));
    camera.processKeyboard(CameraMovement::BACKWARD, delta_time);
    pos = camera.getPosition();
    ASSERT_TRUE(pos.z > 0.0f); // Should move in +Z direction (backward)
    
    // Test left movement
    camera.setPosition(glm::vec3(0.0f, 0.0f, 0.0f));
    camera.processKeyboard(CameraMovement::LEFT, delta_time);
    pos = camera.getPosition();
    ASSERT_TRUE(pos.x < 0.0f); // Should move in -X direction (left)
    
    // Test right movement
    camera.setPosition(glm::vec3(0.0f, 0.0f, 0.0f));
    camera.processKeyboard(CameraMovement::RIGHT, delta_time);
    pos = camera.getPosition();
    ASSERT_TRUE(pos.x > 0.0f); // Should move in +X direction (right)
    
    // Test up movement
    camera.setPosition(glm::vec3(0.0f, 0.0f, 0.0f));
    camera.processKeyboard(CameraMovement::UP, delta_time);
    pos = camera.getPosition();
    ASSERT_TRUE(pos.y > 0.0f); // Should move in +Y direction (up)
    
    // Test down movement
    camera.setPosition(glm::vec3(0.0f, 0.0f, 0.0f));
    camera.processKeyboard(CameraMovement::DOWN, delta_time);
    pos = camera.getPosition();
    ASSERT_TRUE(pos.y < 0.0f); // Should move in -Y direction (down)
}

void test_camera_rotation() {
    Camera camera;
    
    // Test yaw rotation
    camera.processMouseMovement(100.0f, 0.0f);
    ASSERT_TRUE(camera.getYaw() > -90.0f); // Should rotate right
    
    camera.processMouseMovement(-200.0f, 0.0f);
    ASSERT_TRUE(camera.getYaw() < -90.0f); // Should rotate left
    
    // Test pitch rotation
    camera.processMouseMovement(0.0f, 100.0f);
    ASSERT_TRUE(camera.getPitch() > 0.0f); // Should look up
    
    camera.processMouseMovement(0.0f, -200.0f);
    ASSERT_TRUE(camera.getPitch() < 0.0f); // Should look down
    
    // Test pitch clamping
    camera.processMouseMovement(0.0f, 10000.0f);
    ASSERT_TRUE(camera.getPitch() <= 89.0f); // Should be clamped
    
    camera.processMouseMovement(0.0f, -10000.0f);
    ASSERT_TRUE(camera.getPitch() >= -89.0f); // Should be clamped
}

void test_camera_zoom() {
    Camera camera;
    float initial_fov = camera.getFOV();
    
    // Zoom in (positive scroll)
    camera.processMouseScroll(5.0f);
    ASSERT_TRUE(camera.getFOV() < initial_fov);
    
    // Zoom out (negative scroll)
    camera.processMouseScroll(-10.0f);
    ASSERT_TRUE(camera.getFOV() > initial_fov);
    
    // Test FOV clamping
    camera.processMouseScroll(1000.0f);
    ASSERT_TRUE(camera.getFOV() >= 1.0f); // Should be clamped at minimum
    
    camera.processMouseScroll(-1000.0f);
    ASSERT_TRUE(camera.getFOV() <= 120.0f); // Should be clamped at maximum
}

void test_view_matrix() {
    Camera camera;
    camera.setPosition(glm::vec3(0.0f, 0.0f, 5.0f));
    
    glm::mat4 view = camera.getViewMatrix();
    
    // The view matrix should transform world space to camera space
    // Camera at (0,0,5) looking down -Z should see origin at (0,0,-5) in view space
    glm::vec4 origin_world(0.0f, 0.0f, 0.0f, 1.0f);
    glm::vec4 origin_view = view * origin_world;
    
    ASSERT_NEAR(origin_view.x, 0.0f, 1e-6f);
    ASSERT_NEAR(origin_view.y, 0.0f, 1e-6f);
    ASSERT_NEAR(origin_view.z, -5.0f, 1e-6f);
}

void test_projection_matrix() {
    Camera camera;
    float aspect = 16.0f / 9.0f;
    
    glm::mat4 proj = camera.getProjectionMatrix(aspect);
    
    // Test that it's a valid perspective matrix
    // Bottom-right element should be 0 for perspective projection
    ASSERT_NEAR(proj[3][3], 0.0f, 1e-6f);
    
    // Test aspect ratio influence
    glm::mat4 proj_square = camera.getProjectionMatrix(1.0f);
    ASSERT_FALSE(glm::all(glm::epsilonEqual(proj[0], proj_square[0], 1e-6f)));
}

void test_view_projection_matrix() {
    Camera camera;
    camera.setPosition(glm::vec3(0.0f, 0.0f, 5.0f));
    
    glm::mat4 view = camera.getViewMatrix();
    glm::mat4 proj = camera.getProjectionMatrix(1.0f);
    glm::mat4 view_proj = camera.getViewProjectionMatrix(1.0f);
    
    // View-projection should be proj * view
    glm::mat4 expected = proj * view;
    
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            ASSERT_NEAR(view_proj[i][j], expected[i][j], 1e-6f);
        }
    }
}

void test_camera_front_vector() {
    Camera camera;
    
    // Default front vector (looking down -Z)
    glm::vec3 front = camera.getFront();
    ASSERT_NEAR(front.x, 0.0f, 1e-6f);
    ASSERT_NEAR(front.y, 0.0f, 1e-6f);
    ASSERT_NEAR(front.z, -1.0f, 1e-6f);
    
    // After rotation, front vector should change
    camera.processMouseMovement(90.0f / 0.1f, 0.0f); // Rotate 90 degrees right
    front = camera.getFront();
    ASSERT_NEAR(front.x, 1.0f, 1e-3f); // Now looking along +X
    ASSERT_NEAR(front.z, 0.0f, 1e-3f);
}

void test_camera_vectors_orthogonal() {
    Camera camera;
    
    // Test at various orientations
    for (float yaw = -180.0f; yaw <= 180.0f; yaw += 45.0f) {
        for (float pitch = -80.0f; pitch <= 80.0f; pitch += 40.0f) {
            camera.processMouseMovement(yaw / 0.1f, pitch / 0.1f);
            
            glm::vec3 front = camera.getFront();
            glm::vec3 right = camera.getRight();
            glm::vec3 up = camera.getUp();
            
            // All vectors should be normalized
            ASSERT_NEAR(glm::length(front), 1.0f, 1e-6f);
            ASSERT_NEAR(glm::length(right), 1.0f, 1e-6f);
            ASSERT_NEAR(glm::length(up), 1.0f, 1e-6f);
            
            // All vectors should be orthogonal
            ASSERT_NEAR(glm::dot(front, right), 0.0f, 1e-6f);
            ASSERT_NEAR(glm::dot(front, up), 0.0f, 1e-6f);
            ASSERT_NEAR(glm::dot(right, up), 0.0f, 1e-6f);
        }
    }
}

void test_camera_settings() {
    Camera camera;
    
    // Test movement speed
    camera.setMovementSpeed(10.0f);
    glm::vec3 initial_pos = camera.getPosition();
    camera.processKeyboard(CameraMovement::FORWARD, 1.0f);
    glm::vec3 new_pos = camera.getPosition();
    float distance = glm::length(new_pos - initial_pos);
    ASSERT_NEAR(distance, 10.0f, 1e-6f);
    
    // Test mouse sensitivity
    camera.setMouseSensitivity(2.0f);
    float initial_yaw = camera.getYaw();
    camera.processMouseMovement(10.0f, 0.0f);
    float yaw_change = camera.getYaw() - initial_yaw;
    ASSERT_NEAR(yaw_change, 20.0f, 1e-6f); // 10 * 2.0 sensitivity
    
    // Test FOV setting
    camera.setFOV(45.0f);
    ASSERT_NEAR(camera.getFOV(), 45.0f, 1e-6f);
    
    // Test clip planes
    camera.setClipPlanes(1.0f, 500.0f);
    ASSERT_NEAR(camera.getNearPlane(), 1.0f, 1e-6f);
    ASSERT_NEAR(camera.getFarPlane(), 500.0f, 1e-6f);
}

void test_camera_save_load_state() {
    Camera camera;
    
    // Set specific camera state
    camera.setPosition(glm::vec3(10.0f, 20.0f, 30.0f));
    camera.processMouseMovement(45.0f, -30.0f); // Change yaw and pitch
    camera.setFOV(90.0f);
    camera.setMovementSpeed(15.0f);
    camera.setMouseSensitivity(0.3f);
    camera.setClipPlanes(0.5f, 2000.0f);
    
    // Save state
    const std::string test_file = "test_camera_state.txt";
    camera.saveState(test_file);
    
    // Create new camera with default values
    Camera loaded_camera;
    
    // Load state
    bool result = loaded_camera.loadState(test_file);
    ASSERT_TRUE(result);
    
    // Verify loaded state matches saved state
    ASSERT_VEC_NEAR(loaded_camera.getPosition(), glm::vec3(10.0f, 20.0f, 30.0f), 1e-6f);
    ASSERT_NEAR(loaded_camera.getFOV(), 90.0f, 1e-6f);
    // Note: Can't directly verify movement_speed_ and mouse_sensitivity_ as they're private
    // But we can verify the values were loaded by checking camera behavior
    ASSERT_NEAR(loaded_camera.getNearPlane(), 0.5f, 1e-6f);
    ASSERT_NEAR(loaded_camera.getFarPlane(), 2000.0f, 1e-6f);
    
    // Clean up test file
    std::remove(test_file.c_str());
}

void test_camera_load_nonexistent_file() {
    Camera camera;
    
    // Try to load non-existent file
    bool result = camera.loadState("nonexistent_file.txt");
    ASSERT_FALSE(result);
    
    // Camera should remain unchanged
    ASSERT_VEC_NEAR(camera.getPosition(), glm::vec3(0.0f, 0.0f, 3.0f), 1e-6f);
}

int main() {
    std::cout << "Running Camera Tests..." << std::endl;
    
    RUN_TEST(test_camera_constructor);
    RUN_TEST(test_camera_custom_constructor);
    RUN_TEST(test_camera_movement);
    RUN_TEST(test_camera_rotation);
    RUN_TEST(test_camera_zoom);
    RUN_TEST(test_view_matrix);
    RUN_TEST(test_projection_matrix);
    RUN_TEST(test_view_projection_matrix);
    RUN_TEST(test_camera_front_vector);
    RUN_TEST(test_camera_vectors_orthogonal);
    RUN_TEST(test_camera_settings);
    RUN_TEST(test_camera_save_load_state);
    RUN_TEST(test_camera_load_nonexistent_file);
    
    TestFramework::getInstance().printSummary();
    
    return TestFramework::getInstance().getFailureCount() > 0 ? 1 : 0;
}