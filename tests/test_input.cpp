#include "test_framework.h"
#include "core/input.h"
#include "core/camera.h"
#include <GLFW/glfw3.h>

using namespace SplatRender;
using namespace SplatRender::Test;

// Mock window for testing
class MockWindow {
public:
    MockWindow() : initialized_(false), window_(nullptr) {}
    
    bool initialize() {
        if (!glfwInit()) {
            return false;
        }
        
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        window_ = glfwCreateWindow(640, 480, "Test", nullptr, nullptr);
        if (!window_) {
            glfwTerminate();
            return false;
        }
        
        glfwMakeContextCurrent(window_);
        initialized_ = true;
        return true;
    }
    
    ~MockWindow() {
        if (window_) {
            glfwDestroyWindow(window_);
        }
        if (initialized_) {
            glfwTerminate();
        }
    }
    
    GLFWwindow* getWindow() { return window_; }
    
private:
    bool initialized_;
    GLFWwindow* window_;
};

void test_input_handler_creation() {
    MockWindow mock_window;
    if (!mock_window.initialize()) {
        std::cout << "SKIPPED: GLFW not available" << std::endl;
        return;
    }
    
    Camera camera;
    InputHandler handler(mock_window.getWindow(), &camera);
    
    // Should start with mouse captured
    ASSERT_TRUE(handler.isMouseCaptured());
}

void test_key_state_tracking() {
    MockWindow mock_window;
    if (!mock_window.initialize()) {
        std::cout << "SKIPPED: GLFW not available" << std::endl;
        return;
    }
    
    Camera camera;
    InputHandler handler(mock_window.getWindow(), &camera);
    
    // Initially no keys pressed
    ASSERT_FALSE(handler.isKeyPressed(GLFW_KEY_W));
    ASSERT_FALSE(handler.isKeyPressed(GLFW_KEY_A));
    
    // Simulate key press
    handler.onKeyCallback(GLFW_KEY_W, 0, GLFW_PRESS, 0);
    ASSERT_TRUE(handler.isKeyPressed(GLFW_KEY_W));
    
    // Simulate key release
    handler.onKeyCallback(GLFW_KEY_W, 0, GLFW_RELEASE, 0);
    ASSERT_FALSE(handler.isKeyPressed(GLFW_KEY_W));
}

void test_mouse_button_state() {
    MockWindow mock_window;
    if (!mock_window.initialize()) {
        std::cout << "SKIPPED: GLFW not available" << std::endl;
        return;
    }
    
    Camera camera;
    InputHandler handler(mock_window.getWindow(), &camera);
    
    // Initially no buttons pressed
    ASSERT_FALSE(handler.isMouseButtonPressed(GLFW_MOUSE_BUTTON_LEFT));
    
    // Simulate button press
    handler.onMouseButtonCallback(GLFW_MOUSE_BUTTON_LEFT, GLFW_PRESS, 0);
    ASSERT_TRUE(handler.isMouseButtonPressed(GLFW_MOUSE_BUTTON_LEFT));
    
    // Simulate button release
    handler.onMouseButtonCallback(GLFW_MOUSE_BUTTON_LEFT, GLFW_RELEASE, 0);
    ASSERT_FALSE(handler.isMouseButtonPressed(GLFW_MOUSE_BUTTON_LEFT));
}

void test_mouse_capture_toggle() {
    MockWindow mock_window;
    if (!mock_window.initialize()) {
        std::cout << "SKIPPED: GLFW not available" << std::endl;
        return;
    }
    
    Camera camera;
    InputHandler handler(mock_window.getWindow(), &camera);
    
    // Should start captured
    ASSERT_TRUE(handler.isMouseCaptured());
    
    // Toggle with TAB
    handler.onKeyCallback(GLFW_KEY_TAB, 0, GLFW_PRESS, 0);
    ASSERT_FALSE(handler.isMouseCaptured());
    
    // Toggle again
    handler.onKeyCallback(GLFW_KEY_TAB, 0, GLFW_PRESS, 0);
    ASSERT_TRUE(handler.isMouseCaptured());
}

void test_camera_movement_input() {
    MockWindow mock_window;
    if (!mock_window.initialize()) {
        std::cout << "SKIPPED: GLFW not available" << std::endl;
        return;
    }
    
    Camera camera;
    InputHandler handler(mock_window.getWindow(), &camera);
    
    glm::vec3 initial_pos = camera.getPosition();
    
    // Simulate W key press and process input
    handler.onKeyCallback(GLFW_KEY_W, 0, GLFW_PRESS, 0);
    handler.processInput(0.1f); // 100ms delta time
    
    glm::vec3 new_pos = camera.getPosition();
    
    // Camera should have moved forward
    ASSERT_TRUE(glm::length(new_pos - initial_pos) > 0.0f);
    
    handler.onKeyCallback(GLFW_KEY_W, 0, GLFW_RELEASE, 0);
}

void test_mouse_movement() {
    MockWindow mock_window;
    if (!mock_window.initialize()) {
        std::cout << "SKIPPED: GLFW not available" << std::endl;
        return;
    }
    
    Camera camera;
    InputHandler handler(mock_window.getWindow(), &camera);
    
    float initial_yaw = camera.getYaw();
    float initial_pitch = camera.getPitch();
    
    // Simulate mouse movement
    handler.onMouseMoveCallback(100.0, 100.0); // First call sets initial position
    handler.onMouseMoveCallback(200.0, 150.0); // Move right and down
    
    // Camera should have rotated
    ASSERT_TRUE(camera.getYaw() != initial_yaw);
    ASSERT_TRUE(camera.getPitch() != initial_pitch);
}

void test_scroll_fov_adjustment() {
    MockWindow mock_window;
    if (!mock_window.initialize()) {
        std::cout << "SKIPPED: GLFW not available" << std::endl;
        return;
    }
    
    Camera camera;
    InputHandler handler(mock_window.getWindow(), &camera);
    
    float initial_fov = camera.getFOV();
    
    // Scroll up (positive y offset) should decrease FOV
    handler.onScrollCallback(0.0, 5.0);
    ASSERT_TRUE(camera.getFOV() < initial_fov);
    
    // Scroll down (negative y offset) should increase FOV
    handler.onScrollCallback(0.0, -10.0);
    ASSERT_TRUE(camera.getFOV() > initial_fov - 5.0f);
}

void test_input_when_mouse_not_captured() {
    MockWindow mock_window;
    if (!mock_window.initialize()) {
        std::cout << "SKIPPED: GLFW not available" << std::endl;
        return;
    }
    
    Camera camera;
    InputHandler handler(mock_window.getWindow(), &camera);
    
    // Disable mouse capture
    handler.setMouseCapture(false);
    
    glm::vec3 initial_pos = camera.getPosition();
    float initial_yaw = camera.getYaw();
    
    // Try to move - should not work
    handler.onKeyCallback(GLFW_KEY_W, 0, GLFW_PRESS, 0);
    handler.processInput(0.1f);
    
    // Try mouse movement - should not work
    handler.onMouseMoveCallback(100.0, 100.0);
    handler.onMouseMoveCallback(200.0, 200.0);
    
    // Position and rotation should not change
    ASSERT_TRUE(glm::all(glm::equal(camera.getPosition(), initial_pos)));
    ASSERT_EQUAL(camera.getYaw(), initial_yaw);
    
    handler.onKeyCallback(GLFW_KEY_W, 0, GLFW_RELEASE, 0);
}

int main() {
    std::cout << "Running Input Handler Tests..." << std::endl;
    std::cout << "Note: These tests require GLFW" << std::endl;
    
    RUN_TEST(test_input_handler_creation);
    RUN_TEST(test_key_state_tracking);
    RUN_TEST(test_mouse_button_state);
    RUN_TEST(test_mouse_capture_toggle);
    RUN_TEST(test_camera_movement_input);
    RUN_TEST(test_mouse_movement);
    RUN_TEST(test_scroll_fov_adjustment);
    RUN_TEST(test_input_when_mouse_not_captured);
    
    TestFramework::getInstance().printSummary();
    
    return TestFramework::getInstance().getFailureCount() > 0 ? 1 : 0;
}