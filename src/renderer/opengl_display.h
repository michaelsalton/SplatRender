#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <string>
#include <vector>

namespace SplatRender {

class OpenGLDisplay {
public:
    OpenGLDisplay();
    ~OpenGLDisplay();
    
    bool initialize(int width, int height);
    void shutdown();
    
    // Display a buffer as texture
    void displayTexture(const std::vector<float>& buffer, int width, int height);
    
    // Display with CUDA interop (Linux only)
    void displayCudaTexture(GLuint cuda_texture);
    
    // Clear the display
    void clear(float r = 0.0f, float g = 0.0f, float b = 0.0f, float a = 1.0f);
    
    // Resize handling
    void resize(int width, int height);
    
    // Get texture ID for CUDA interop
    GLuint getTextureID() const { return texture_; }

private:
    bool createShaders();
    bool createQuad();
    void checkGLError(const std::string& context);
    
    int width_;
    int height_;
    
    GLuint texture_;
    GLuint shader_program_;
    GLuint vao_;
    GLuint vbo_;
    
    bool is_initialized_;
    
    // Shader source
    static const char* vertex_shader_source_;
    static const char* fragment_shader_source_;
};

} // namespace SplatRender