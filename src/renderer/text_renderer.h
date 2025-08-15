#pragma once

#include <string>
#include <glm/glm.hpp>
#include <GL/glew.h>

namespace SplatRender {

class TextRenderer {
public:
    TextRenderer();
    ~TextRenderer();
    
    bool initialize(int screen_width, int screen_height);
    void shutdown();
    
    // Draw text at screen coordinates
    void drawText(const std::string& text, float x, float y, float scale = 1.0f, glm::vec3 color = glm::vec3(1.0f));
    
    // Update screen dimensions
    void updateScreenSize(int width, int height);
    
private:
    void createFontTexture();
    void createShaders();
    
    GLuint shader_program_;
    GLuint vao_, vbo_;
    GLuint font_texture_;
    
    int screen_width_;
    int screen_height_;
    
    // Shader uniform locations
    GLint projection_loc_;
    GLint text_color_loc_;
    GLint texture_loc_;
    
    // Simple 8x8 bitmap font (ASCII 32-127)
    static const unsigned char font_bitmap_[];
};

} // namespace SplatRender