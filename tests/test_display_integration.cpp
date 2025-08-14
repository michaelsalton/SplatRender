#include "test_framework.h"
#include "renderer/opengl_display.h"
#include <GLFW/glfw3.h>
#include <vector>
#include <iostream>

using namespace SplatRender;
using namespace SplatRender::Test;

// Helper class to manage GLFW context for tests
class GLFWTestContext {
public:
    GLFWTestContext() : window_(nullptr), initialized_(false) {}
    
    bool initialize() {
        if (!glfwInit()) {
            std::cerr << "Failed to initialize GLFW for tests" << std::endl;
            return false;
        }
        
        // Create hidden window for OpenGL context
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        
#ifdef __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
        
        window_ = glfwCreateWindow(640, 480, "Test Window", nullptr, nullptr);
        if (!window_) {
            std::cerr << "Failed to create GLFW window for tests" << std::endl;
            glfwTerminate();
            return false;
        }
        
        glfwMakeContextCurrent(window_);
        initialized_ = true;
        return true;
    }
    
    ~GLFWTestContext() {
        if (window_) {
            glfwDestroyWindow(window_);
        }
        if (initialized_) {
            glfwTerminate();
        }
    }
    
    GLFWwindow* getWindow() { return window_; }
    bool isInitialized() { return initialized_; }
    
private:
    GLFWwindow* window_;
    bool initialized_;
};

// Global test context
static GLFWTestContext g_test_context;

void test_opengl_display_initialization() {
    if (!g_test_context.isInitialized()) {
        std::cout << "SKIPPED: OpenGL context not available" << std::endl;
        return;
    }
    
    OpenGLDisplay display;
    bool result = display.initialize(640, 480);
    ASSERT_TRUE(result);
    
    // Check that we can get the texture ID
    GLuint tex_id = display.getTextureID();
    ASSERT_TRUE(tex_id > 0);
    
    display.shutdown();
}

void test_opengl_display_clear() {
    if (!g_test_context.isInitialized()) {
        std::cout << "SKIPPED: OpenGL context not available" << std::endl;
        return;
    }
    
    OpenGLDisplay display;
    display.initialize(640, 480);
    
    // Clear to red
    display.clear(1.0f, 0.0f, 0.0f, 1.0f);
    
    // Read back pixel to verify (requires framebuffer)
    // For now, just verify no OpenGL errors
    GLenum error = glGetError();
    ASSERT_TRUE(error == GL_NO_ERROR);
    
    display.shutdown();
}

void test_opengl_display_texture_upload() {
    if (!g_test_context.isInitialized()) {
        std::cout << "SKIPPED: OpenGL context not available" << std::endl;
        return;
    }
    
    OpenGLDisplay display;
    display.initialize(320, 240);
    
    // Create test pattern
    std::vector<float> buffer(320 * 240 * 4);
    for (int y = 0; y < 240; ++y) {
        for (int x = 0; x < 320; ++x) {
            int idx = (y * 320 + x) * 4;
            buffer[idx + 0] = static_cast<float>(x) / 320.0f; // R
            buffer[idx + 1] = static_cast<float>(y) / 240.0f; // G
            buffer[idx + 2] = 0.5f; // B
            buffer[idx + 3] = 1.0f; // A
        }
    }
    
    // Display the texture
    display.displayTexture(buffer, 320, 240);
    
    // Verify no OpenGL errors
    GLenum error = glGetError();
    ASSERT_TRUE(error == GL_NO_ERROR);
    
    display.shutdown();
}

void test_opengl_display_resize() {
    if (!g_test_context.isInitialized()) {
        std::cout << "SKIPPED: OpenGL context not available" << std::endl;
        return;
    }
    
    OpenGLDisplay display;
    display.initialize(640, 480);
    
    // Resize
    display.resize(800, 600);
    
    // Create and display buffer with new size
    std::vector<float> buffer(800 * 600 * 4, 0.5f);
    display.displayTexture(buffer, 800, 600);
    
    // Verify no OpenGL errors
    GLenum error = glGetError();
    ASSERT_TRUE(error == GL_NO_ERROR);
    
    display.shutdown();
}

void test_opengl_display_multiple_frames() {
    if (!g_test_context.isInitialized()) {
        std::cout << "SKIPPED: OpenGL context not available" << std::endl;
        return;
    }
    
    OpenGLDisplay display;
    display.initialize(256, 256);
    
    // Render multiple frames with different patterns
    for (int frame = 0; frame < 10; ++frame) {
        std::vector<float> buffer(256 * 256 * 4);
        
        // Create animated pattern
        float t = frame / 10.0f;
        for (int y = 0; y < 256; ++y) {
            for (int x = 0; x < 256; ++x) {
                int idx = (y * 256 + x) * 4;
                buffer[idx + 0] = t; // R
                buffer[idx + 1] = 1.0f - t; // G
                buffer[idx + 2] = 0.5f; // B
                buffer[idx + 3] = 1.0f; // A
            }
        }
        
        display.displayTexture(buffer, 256, 256);
        
        // Swap buffers to complete frame
        glfwSwapBuffers(g_test_context.getWindow());
    }
    
    // Verify no OpenGL errors
    GLenum error = glGetError();
    ASSERT_TRUE(error == GL_NO_ERROR);
    
    display.shutdown();
}

void test_opengl_shader_compilation() {
    if (!g_test_context.isInitialized()) {
        std::cout << "SKIPPED: OpenGL context not available" << std::endl;
        return;
    }
    
    // Test shader compilation directly
    const char* vertex_source = R"(
        #version 330 core
        layout (location = 0) in vec2 aPos;
        void main() {
            gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
        }
    )";
    
    const char* fragment_source = R"(
        #version 330 core
        out vec4 FragColor;
        void main() {
            FragColor = vec4(1.0, 0.0, 0.0, 1.0);
        }
    )";
    
    // Compile vertex shader
    GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex_shader, 1, &vertex_source, nullptr);
    glCompileShader(vertex_shader);
    
    GLint success;
    glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &success);
    ASSERT_TRUE(success);
    
    // Compile fragment shader
    GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader, 1, &fragment_source, nullptr);
    glCompileShader(fragment_shader);
    
    glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &success);
    ASSERT_TRUE(success);
    
    // Link program
    GLuint program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);
    
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    ASSERT_TRUE(success);
    
    // Cleanup
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
    glDeleteProgram(program);
}

void test_opengl_state_management() {
    if (!g_test_context.isInitialized()) {
        std::cout << "SKIPPED: OpenGL context not available" << std::endl;
        return;
    }
    
    // Test that OpenGL state is properly managed
    OpenGLDisplay display;
    display.initialize(640, 480);
    
    // Get initial viewport
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    ASSERT_TRUE(viewport[2] == 640);
    ASSERT_TRUE(viewport[3] == 480);
    
    // Resize should update viewport
    display.resize(800, 600);
    glGetIntegerv(GL_VIEWPORT, viewport);
    ASSERT_TRUE(viewport[2] == 800);
    ASSERT_TRUE(viewport[3] == 600);
    
    display.shutdown();
}

void test_window_event_handling() {
    if (!g_test_context.isInitialized()) {
        std::cout << "SKIPPED: OpenGL context not available" << std::endl;
        return;
    }
    
    // Create a visible window for this test
    glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
    GLFWwindow* test_window = glfwCreateWindow(640, 480, "Event Test", nullptr, nullptr);
    ASSERT_TRUE(test_window != nullptr);
    
    glfwMakeContextCurrent(test_window);
    
    // Set up event tracking
    bool key_pressed = false;
    bool mouse_moved = false;
    bool window_resized = false;
    
    struct EventData {
        bool* key_pressed;
        bool* mouse_moved;
        bool* window_resized;
    } event_data = { &key_pressed, &mouse_moved, &window_resized };
    
    glfwSetWindowUserPointer(test_window, &event_data);
    
    // Set callbacks
    glfwSetKeyCallback(test_window, [](GLFWwindow* window, int /*key*/, int /*scancode*/, int action, int /*mods*/) {
        EventData* data = static_cast<EventData*>(glfwGetWindowUserPointer(window));
        if (action == GLFW_PRESS) {
            *data->key_pressed = true;
        }
    });
    
    glfwSetCursorPosCallback(test_window, [](GLFWwindow* window, double /*xpos*/, double /*ypos*/) {
        EventData* data = static_cast<EventData*>(glfwGetWindowUserPointer(window));
        *data->mouse_moved = true;
    });
    
    glfwSetFramebufferSizeCallback(test_window, [](GLFWwindow* window, int /*width*/, int /*height*/) {
        EventData* data = static_cast<EventData*>(glfwGetWindowUserPointer(window));
        *data->window_resized = true;
    });
    
    // Process events
    glfwPollEvents();
    
    // In a real test environment, we would simulate events
    // For now, just verify the window was created
    ASSERT_TRUE(test_window != nullptr);
    
    glfwDestroyWindow(test_window);
}

int main() {
    std::cout << "Running Display Integration Tests..." << std::endl;
    std::cout << "Note: These tests require an OpenGL context" << std::endl;
    
    // Initialize test context
    if (!g_test_context.initialize()) {
        std::cerr << "WARNING: Could not initialize OpenGL context for testing" << std::endl;
        std::cerr << "Display integration tests will be skipped" << std::endl;
        return 0; // Don't fail if no display available (e.g., CI environment)
    }
    
    RUN_TEST(test_opengl_display_initialization);
    RUN_TEST(test_opengl_display_clear);
    RUN_TEST(test_opengl_display_texture_upload);
    RUN_TEST(test_opengl_display_resize);
    RUN_TEST(test_opengl_display_multiple_frames);
    RUN_TEST(test_opengl_shader_compilation);
    RUN_TEST(test_opengl_state_management);
    RUN_TEST(test_window_event_handling);
    
    TestFramework::getInstance().printSummary();
    
    return TestFramework::getInstance().getFailureCount() > 0 ? 1 : 0;
}