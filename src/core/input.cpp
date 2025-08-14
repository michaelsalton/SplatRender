#include "core/input.h"
#include "core/camera.h"
#include <GLFW/glfw3.h>

namespace SplatRender {

// Initialize static member
InputHandler* InputHandler::instance_ = nullptr;

InputHandler::InputHandler(GLFWwindow* window, Camera* camera)
    : window_(window)
    , camera_(camera)
    , mouse_captured_(false)
    , first_mouse_(true)
    , last_mouse_x_(640.0)
    , last_mouse_y_(360.0) {
    
    // Set static instance
    instance_ = this;
    
    glfwSetKeyCallback(window, keyCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    
    // Capture mouse by default
    setMouseCapture(true);
}

InputHandler::~InputHandler() {
    // Clear static instance
    if (instance_ == this) {
        instance_ = nullptr;
    }
}

void InputHandler::processInput(float delta_time) {
    // Process continuous key input
    if (camera_ && mouse_captured_) {
        if (isKeyPressed(GLFW_KEY_W)) {
            camera_->processKeyboard(CameraMovement::FORWARD, delta_time);
        }
        if (isKeyPressed(GLFW_KEY_S)) {
            camera_->processKeyboard(CameraMovement::BACKWARD, delta_time);
        }
        if (isKeyPressed(GLFW_KEY_A)) {
            camera_->processKeyboard(CameraMovement::LEFT, delta_time);
        }
        if (isKeyPressed(GLFW_KEY_D)) {
            camera_->processKeyboard(CameraMovement::RIGHT, delta_time);
        }
        if (isKeyPressed(GLFW_KEY_SPACE)) {
            camera_->processKeyboard(CameraMovement::UP, delta_time);
        }
        if (isKeyPressed(GLFW_KEY_LEFT_SHIFT)) {
            camera_->processKeyboard(CameraMovement::DOWN, delta_time);
        }
    }
}

void InputHandler::setMouseCapture(bool capture) {
    mouse_captured_ = capture;
    glfwSetInputMode(window_, GLFW_CURSOR, capture ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
    
    if (capture) {
        first_mouse_ = true;
    }
}

bool InputHandler::isKeyPressed(int key) const {
    auto it = key_states_.find(key);
    return it != key_states_.end() && it->second;
}

bool InputHandler::isMouseButtonPressed(int button) const {
    auto it = mouse_button_states_.find(button);
    return it != mouse_button_states_.end() && it->second;
}

void InputHandler::onKeyCallback(int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
        key_states_[key] = true;
        
        // Toggle mouse capture with TAB
        if (key == GLFW_KEY_TAB) {
            setMouseCapture(!mouse_captured_);
        }
    } else if (action == GLFW_RELEASE) {
        key_states_[key] = false;
    }
}

void InputHandler::onMouseButtonCallback(int button, int action, int mods) {
    if (action == GLFW_PRESS) {
        mouse_button_states_[button] = true;
    } else if (action == GLFW_RELEASE) {
        mouse_button_states_[button] = false;
    }
}

void InputHandler::onMouseMoveCallback(double xpos, double ypos) {
    if (!camera_ || !mouse_captured_) {
        return;
    }
    
    if (first_mouse_) {
        last_mouse_x_ = xpos;
        last_mouse_y_ = ypos;
        first_mouse_ = false;
    }
    
    float xoffset = static_cast<float>(xpos - last_mouse_x_);
    float yoffset = static_cast<float>(last_mouse_y_ - ypos); // Reversed
    
    last_mouse_x_ = xpos;
    last_mouse_y_ = ypos;
    
    camera_->processMouseMovement(xoffset, yoffset);
}

void InputHandler::onScrollCallback(double xoffset, double yoffset) {
    if (camera_ && mouse_captured_) {
        camera_->processMouseScroll(static_cast<float>(yoffset));
    }
}

void InputHandler::onWindowSizeCallback(int width, int height) {
    // Window size changes are handled by the engine
}

// Static callback implementations
void InputHandler::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (instance_) {
        instance_->onKeyCallback(key, scancode, action, mods);
    }
}

void InputHandler::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (instance_) {
        instance_->onMouseButtonCallback(button, action, mods);
    }
}

void InputHandler::cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    if (instance_) {
        instance_->onMouseMoveCallback(xpos, ypos);
    }
}

void InputHandler::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    if (instance_) {
        instance_->onScrollCallback(xoffset, yoffset);
    }
}

void InputHandler::framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    if (instance_) {
        instance_->onWindowSizeCallback(width, height);
    }
}

} // namespace SplatRender