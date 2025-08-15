#include "core/engine.h"
#include "core/camera.h"
#include "core/input.h"
#include "renderer/opengl_display.h"
#include "renderer/cpu_rasterizer.h"
#include "renderer/text_renderer.h"
#include "renderer/axis_renderer.h"
#include "io/ply_loader.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include <sstream>
#include <iomanip>

namespace SplatRender {

// Error callback for GLFW
static void error_callback(int error, const char* description) {
    std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}


Engine::Engine()
    : window_width_(1280)
    , window_height_(720)
    , window_title_("SplatRender")
    , window_(nullptr)
    , last_frame_time_(std::chrono::steady_clock::now())
    , fps_(0.0f)
    , frame_count_(0)
    , fps_update_timer_(0.0f)
    , is_initialized_(false)
    , should_close_(false) {
}

Engine::~Engine() {
    shutdown();
}

bool Engine::initialize(int width, int height, const std::string& title) {
    if (is_initialized_) {
        return true;
    }
    
    window_width_ = width;
    window_height_ = height;
    window_title_ = title;
    
    // Set error callback
    glfwSetErrorCallback(error_callback);
    
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }
    
    // Configure GLFW
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    
    // Create window
    window_ = glfwCreateWindow(window_width_, window_height_, window_title_.c_str(), nullptr, nullptr);
    if (!window_) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }
    
    // Make OpenGL context current
    glfwMakeContextCurrent(window_);
    
    // Enable V-Sync
    glfwSwapInterval(1);
    
    
    // Initialize OpenGL display
    display_ = std::make_unique<OpenGLDisplay>();
    if (!display_->initialize(window_width_, window_height_)) {
        std::cerr << "Failed to initialize OpenGL display" << std::endl;
        return false;
    }
    
    // Initialize camera
    camera_ = std::make_unique<Camera>();
    camera_->setPosition(glm::vec3(0.0f, 0.0f, 5.0f));
    
    // Print initial camera info
    glm::vec3 pos = camera_->getPosition();
    glm::vec3 front = camera_->getFront();
    std::cout << "Initial camera position: (" << pos.x << ", " << pos.y << ", " << pos.z << ")" << std::endl;
    std::cout << "Initial camera front: (" << front.x << ", " << front.y << ", " << front.z << ")" << std::endl;
    
    // Store pointer to engine for callbacks
    glfwSetWindowUserPointer(window_, this);
    
    // Initialize input handler
    input_handler_ = std::make_unique<InputHandler>(window_, camera_.get());
    
    // Initialize CPU rasterizer
    cpu_rasterizer_ = std::make_unique<CPURasterizer>();
    RenderSettings settings;
    settings.width = window_width_;
    settings.height = window_height_;
    settings.tile_size = 16;
    cpu_rasterizer_->initialize(settings);
    
    // Initialize text renderer
    text_renderer_ = std::make_unique<TextRenderer>();
    if (!text_renderer_->initialize(window_width_, window_height_)) {
        std::cerr << "Failed to initialize text renderer" << std::endl;
        // Continue anyway, just won't show FPS
    }
    
    // Initialize axis renderer
    axis_renderer_ = std::make_unique<AxisRenderer>();
    if (!axis_renderer_->initialize()) {
        std::cerr << "Failed to initialize axis renderer" << std::endl;
        // Continue anyway, just won't show axes
    }
    
    // Set OpenGL state
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    is_initialized_ = true;
    std::cout << "Engine initialized successfully" << std::endl;
    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GLSL Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
    
    // Print controls
    std::cout << "\n=== Controls ===" << std::endl;
    std::cout << "WASD        - Move camera" << std::endl;
    std::cout << "Mouse       - Look around" << std::endl;
    std::cout << "Scroll      - Adjust FOV" << std::endl;
    std::cout << "Space/Shift - Move up/down" << std::endl;
    std::cout << "TAB         - Toggle mouse capture" << std::endl;
    std::cout << "F1          - Print camera info" << std::endl;
    std::cout << "F5          - Save camera state" << std::endl;
    std::cout << "F6          - Load camera state" << std::endl;
    std::cout << "ESC         - Exit" << std::endl;
    std::cout << "===============\n" << std::endl;
    
    return true;
}

void Engine::run() {
    if (!is_initialized_) {
        std::cerr << "Engine not initialized" << std::endl;
        return;
    }
    
    std::cout << "Starting render loop..." << std::endl;
    
    while (!glfwWindowShouldClose(window_) && !should_close_) {
        // Calculate delta time
        auto current_time = std::chrono::steady_clock::now();
        float delta_time = std::chrono::duration<float>(current_time - last_frame_time_).count();
        last_frame_time_ = current_time;
        
        // Poll events
        glfwPollEvents();
        
        // Update
        update(delta_time);
        
        // Render
        render();
        
        // Swap buffers
        glfwSwapBuffers(window_);
        
        // Update FPS counter
        frame_count_++;
        fps_update_timer_ += delta_time;
        if (fps_update_timer_ >= 1.0f) {
            fps_ = frame_count_ / fps_update_timer_;
            frame_count_ = 0;
            fps_update_timer_ = 0.0f;
            
            // Update window title with FPS
            std::stringstream ss;
            ss << window_title_ << " - FPS: " << std::fixed << std::setprecision(1) << fps_;
            glfwSetWindowTitle(window_, ss.str().c_str());
        }
    }
}

void Engine::shutdown() {
    if (!is_initialized_) {
        return;
    }
    
    // Shutdown components
    input_handler_.reset();
    camera_.reset();
    display_.reset();
    
    // Destroy window
    if (window_) {
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }
    
    // Terminate GLFW
    glfwTerminate();
    
    is_initialized_ = false;
    std::cout << "Engine shutdown complete" << std::endl;
}

void Engine::update(float delta_time) {
    // Update input
    input_handler_->processInput(delta_time);
    
    // Check for escape key
    if (input_handler_->isKeyPressed(GLFW_KEY_ESCAPE)) {
        should_close_ = true;
    }
    
    // Check for window resize
    int width, height;
    glfwGetFramebufferSize(window_, &width, &height);
    if (width != window_width_ || height != window_height_) {
        window_width_ = width;
        window_height_ = height;
        display_->resize(width, height);
        
        // Update rasterizer settings
        if (cpu_rasterizer_) {
            RenderSettings settings = cpu_rasterizer_->getSettings();
            settings.width = width;
            settings.height = height;
            cpu_rasterizer_->setSettings(settings);
            
            // Clear render buffer to force resize
            render_buffer_.clear();
        }
        
        // Update text renderer
        if (text_renderer_) {
            text_renderer_->updateScreenSize(width, height);
        }
    }
}

void Engine::render() {
    // Clear the screen
    display_->clear(0.1f, 0.1f, 0.2f, 1.0f);
    
    if (!gaussians_.empty()) {
        // Render Gaussians using CPU rasterizer
        cpu_rasterizer_->render(gaussians_, *camera_, render_buffer_);
        
        // Display rendered image
        display_->displayTexture(render_buffer_, window_width_, window_height_);
        
        // Show stats
        const auto& stats = cpu_rasterizer_->getStats();
        if (frame_count_ % 60 == 0) {  // Print every 60 frames
            std::cout << "Rendered " << stats.visible_gaussians << " Gaussians, "
                      << "culled " << stats.culled_gaussians << ", "
                      << "render time: " << stats.total_time_ms << " ms" << std::endl;
        }
    } else {
        // No Gaussians loaded, show a test pattern
        static std::vector<float> test_buffer;
        if (test_buffer.size() != window_width_ * window_height_ * 4) {
            test_buffer.resize(window_width_ * window_height_ * 4);
            
            // Create a gradient test pattern
            for (int y = 0; y < window_height_; ++y) {
                for (int x = 0; x < window_width_; ++x) {
                    int idx = (y * window_width_ + x) * 4;
                    test_buffer[idx + 0] = static_cast<float>(x) / window_width_;     // R
                    test_buffer[idx + 1] = static_cast<float>(y) / window_height_;    // G
                    test_buffer[idx + 2] = 0.5f;                                      // B
                    test_buffer[idx + 3] = 1.0f;                                      // A
                }
            }
        }
        
        // Display the test pattern
        display_->displayTexture(test_buffer, window_width_, window_height_);
    }
    
    // Draw coordinate axes
    if (axis_renderer_) {
        float aspect_ratio = static_cast<float>(window_width_) / static_cast<float>(window_height_);
        axis_renderer_->render(*camera_, aspect_ratio);
    }
    
    // Draw FPS counter and stats
    if (text_renderer_) {
        std::stringstream fps_text;
        fps_text << "FPS: " << std::fixed << std::setprecision(1) << fps_;
        
        // Draw FPS in top-left corner with a slight offset
        text_renderer_->drawText(fps_text.str(), 10.0f, 10.0f, 2.0f, glm::vec3(1.0f, 1.0f, 0.0f));
        
        // Also show render stats if we have Gaussians
        if (!gaussians_.empty()) {
            const auto& stats = cpu_rasterizer_->getStats();
            std::stringstream stats_text;
            stats_text << "Gaussians: " << stats.visible_gaussians << "/" << gaussians_.size();
            text_renderer_->drawText(stats_text.str(), 10.0f, 35.0f, 1.5f, glm::vec3(0.8f, 1.0f, 0.8f));
            
            stats_text.str("");
            stats_text << "Render: " << std::fixed << std::setprecision(2) << stats.total_time_ms << " ms";
            text_renderer_->drawText(stats_text.str(), 10.0f, 55.0f, 1.5f, glm::vec3(0.8f, 1.0f, 0.8f));
        }
        
        // Show camera position
        glm::vec3 pos = camera_->getPosition();
        std::stringstream pos_text;
        pos_text << "Pos: (" << std::fixed << std::setprecision(1) 
                 << pos.x << ", " << pos.y << ", " << pos.z << ")";
        text_renderer_->drawText(pos_text.str(), 10.0f, 75.0f, 1.5f, glm::vec3(0.8f, 0.8f, 1.0f));
    }
}

bool Engine::loadPLY(const std::string& filename) {
    PLYLoader loader;
    std::vector<Gaussian3D> temp_gaussians;
    
    if (!loader.load(filename, temp_gaussians)) {
        std::cerr << "Failed to load PLY file: " << loader.getLastError() << std::endl;
        return false;
    }
    
    gaussians_ = std::move(temp_gaussians);
    std::cout << "Loaded " << gaussians_.size() << " Gaussians from " << filename << std::endl;
    
    return true;
}

} // namespace SplatRender