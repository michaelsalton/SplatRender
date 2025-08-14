#pragma once

#include <memory>
#include <string>
#include <chrono>

namespace SplatRender {

class Camera;
class InputHandler;
class OpenGLDisplay;

class Engine {
public:
    Engine();
    ~Engine();

    bool initialize(int width, int height, const std::string& title);
    void run();
    void shutdown();

    Camera* getCamera() { return camera_.get(); }
    InputHandler* getInputHandler() { return input_handler_.get(); }
    
    int getWindowWidth() const { return window_width_; }
    int getWindowHeight() const { return window_height_; }

private:
    void update(float delta_time);
    void render();
    
    int window_width_;
    int window_height_;
    std::string window_title_;
    
    struct GLFWwindow* window_;
    
    std::unique_ptr<Camera> camera_;
    std::unique_ptr<InputHandler> input_handler_;
    std::unique_ptr<OpenGLDisplay> display_;
    
    std::chrono::steady_clock::time_point last_frame_time_;
    float fps_;
    int frame_count_;
    float fps_update_timer_;
    
    bool is_initialized_;
    bool should_close_;
};

} // namespace SplatRender