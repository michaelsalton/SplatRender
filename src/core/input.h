#pragma once

#include <functional>
#include <unordered_map>

struct GLFWwindow;

namespace SplatRender {

class Camera;

class InputHandler {
public:
    InputHandler(GLFWwindow* window, Camera* camera);
    ~InputHandler();
    
    void processInput(float delta_time);
    
    // Mouse state
    void setMouseCapture(bool capture);
    bool isMouseCaptured() const { return mouse_captured_; }
    
    // Key state queries
    bool isKeyPressed(int key) const;
    bool isMouseButtonPressed(int button) const;
    
    // Callbacks (called by GLFW)
    void onKeyCallback(int key, int scancode, int action, int mods);
    void onMouseButtonCallback(int button, int action, int mods);
    void onMouseMoveCallback(double xpos, double ypos);
    void onScrollCallback(double xoffset, double yoffset);
    void onWindowSizeCallback(int width, int height);
    
private:
    GLFWwindow* window_;
    Camera* camera_;
    
    // Mouse state
    bool mouse_captured_;
    bool first_mouse_;
    double last_mouse_x_;
    double last_mouse_y_;
    
    // Key states
    std::unordered_map<int, bool> key_states_;
    std::unordered_map<int, bool> mouse_button_states_;
    
    // Static callbacks for GLFW
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
};

} // namespace SplatRender