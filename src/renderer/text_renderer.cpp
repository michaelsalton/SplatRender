#include "renderer/text_renderer.h"
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <vector>

namespace SplatRender {

// Simple 8x8 monospace bitmap font data (ASCII 32-127)
// Each character is 8 bytes (8x8 pixels, 1 bit per pixel)
const unsigned char TextRenderer::font_bitmap_[] = {
    // Space (32)
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    // ! (33)
    0x18, 0x18, 0x18, 0x18, 0x18, 0x00, 0x18, 0x00,
    // " (34)
    0x36, 0x36, 0x36, 0x00, 0x00, 0x00, 0x00, 0x00,
    // # (35)
    0x36, 0x36, 0x7F, 0x36, 0x7F, 0x36, 0x36, 0x00,
    // $ (36)
    0x18, 0x3E, 0x60, 0x3C, 0x06, 0x7C, 0x18, 0x00,
    // % (37)
    0x60, 0x66, 0x0C, 0x18, 0x30, 0x66, 0x06, 0x00,
    // & (38)
    0x38, 0x6C, 0x38, 0x70, 0xDE, 0xCC, 0x76, 0x00,
    // ' (39)
    0x18, 0x18, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00,
    // ( (40)
    0x0C, 0x18, 0x30, 0x30, 0x30, 0x18, 0x0C, 0x00,
    // ) (41)
    0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x18, 0x30, 0x00,
    // * (42)
    0x00, 0x66, 0x3C, 0xFF, 0x3C, 0x66, 0x00, 0x00,
    // + (43)
    0x00, 0x18, 0x18, 0x7E, 0x18, 0x18, 0x00, 0x00,
    // , (44)
    0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x30,
    // - (45)
    0x00, 0x00, 0x00, 0x7E, 0x00, 0x00, 0x00, 0x00,
    // . (46)
    0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x00,
    // / (47)
    0x06, 0x0C, 0x18, 0x30, 0x60, 0xC0, 0x80, 0x00,
    // 0-9 (48-57)
    0x3C, 0x66, 0x6E, 0x76, 0x66, 0x66, 0x3C, 0x00, // 0
    0x18, 0x38, 0x18, 0x18, 0x18, 0x18, 0x7E, 0x00, // 1
    0x3C, 0x66, 0x06, 0x0C, 0x18, 0x30, 0x7E, 0x00, // 2
    0x3C, 0x66, 0x06, 0x1C, 0x06, 0x66, 0x3C, 0x00, // 3
    0x0C, 0x1C, 0x3C, 0x6C, 0x7E, 0x0C, 0x0C, 0x00, // 4
    0x7E, 0x60, 0x7C, 0x06, 0x06, 0x66, 0x3C, 0x00, // 5
    0x1C, 0x30, 0x60, 0x7C, 0x66, 0x66, 0x3C, 0x00, // 6
    0x7E, 0x06, 0x0C, 0x18, 0x30, 0x30, 0x30, 0x00, // 7
    0x3C, 0x66, 0x66, 0x3C, 0x66, 0x66, 0x3C, 0x00, // 8
    0x3C, 0x66, 0x66, 0x3E, 0x06, 0x0C, 0x38, 0x00, // 9
    // : (58)
    0x00, 0x18, 0x18, 0x00, 0x00, 0x18, 0x18, 0x00,
    // ; (59)
    0x00, 0x18, 0x18, 0x00, 0x00, 0x18, 0x18, 0x30,
    // < (60)
    0x0C, 0x18, 0x30, 0x60, 0x30, 0x18, 0x0C, 0x00,
    // = (61)
    0x00, 0x00, 0x7E, 0x00, 0x7E, 0x00, 0x00, 0x00,
    // > (62)
    0x30, 0x18, 0x0C, 0x06, 0x0C, 0x18, 0x30, 0x00,
    // ? (63)
    0x3C, 0x66, 0x06, 0x0C, 0x18, 0x00, 0x18, 0x00,
    // @ (64)
    0x3C, 0x66, 0x6E, 0x6A, 0x6E, 0x60, 0x3C, 0x00,
    // A-Z (65-90)
    0x3C, 0x66, 0x66, 0x7E, 0x66, 0x66, 0x66, 0x00, // A
    0x7C, 0x66, 0x66, 0x7C, 0x66, 0x66, 0x7C, 0x00, // B
    0x3C, 0x66, 0x60, 0x60, 0x60, 0x66, 0x3C, 0x00, // C
    0x78, 0x6C, 0x66, 0x66, 0x66, 0x6C, 0x78, 0x00, // D
    0x7E, 0x60, 0x60, 0x7C, 0x60, 0x60, 0x7E, 0x00, // E
    0x7E, 0x60, 0x60, 0x7C, 0x60, 0x60, 0x60, 0x00, // F
    0x3C, 0x66, 0x60, 0x6E, 0x66, 0x66, 0x3C, 0x00, // G
    0x66, 0x66, 0x66, 0x7E, 0x66, 0x66, 0x66, 0x00, // H
    0x7E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x7E, 0x00, // I
    0x3E, 0x0C, 0x0C, 0x0C, 0x0C, 0x6C, 0x38, 0x00, // J
    0x66, 0x6C, 0x78, 0x70, 0x78, 0x6C, 0x66, 0x00, // K
    0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x7E, 0x00, // L
    0x63, 0x77, 0x7F, 0x6B, 0x63, 0x63, 0x63, 0x00, // M
    0x66, 0x76, 0x7E, 0x6E, 0x66, 0x66, 0x66, 0x00, // N
    0x3C, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x00, // O
    0x7C, 0x66, 0x66, 0x7C, 0x60, 0x60, 0x60, 0x00, // P
    0x3C, 0x66, 0x66, 0x66, 0x6A, 0x6C, 0x36, 0x00, // Q
    0x7C, 0x66, 0x66, 0x7C, 0x6C, 0x66, 0x66, 0x00, // R
    0x3C, 0x66, 0x60, 0x3C, 0x06, 0x66, 0x3C, 0x00, // S
    0x7E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x00, // T
    0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x00, // U
    0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x18, 0x00, // V
    0x63, 0x63, 0x63, 0x6B, 0x7F, 0x77, 0x63, 0x00, // W
    0x66, 0x66, 0x3C, 0x18, 0x3C, 0x66, 0x66, 0x00, // X
    0x66, 0x66, 0x66, 0x3C, 0x18, 0x18, 0x18, 0x00, // Y
    0x7E, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x7E, 0x00, // Z
    // [ (91)
    0x3C, 0x30, 0x30, 0x30, 0x30, 0x30, 0x3C, 0x00,
    // \ (92)
    0xC0, 0x60, 0x30, 0x18, 0x0C, 0x06, 0x02, 0x00,
    // ] (93)
    0x3C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x3C, 0x00,
    // ^ (94)
    0x18, 0x3C, 0x66, 0x00, 0x00, 0x00, 0x00, 0x00,
    // _ (95)
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0x00,
};

const char* text_vertex_shader = R"(
#version 410 core
layout (location = 0) in vec2 vertex;
layout (location = 1) in vec2 texCoord;

out vec2 TexCoord;

uniform mat4 projection;

void main() {
    gl_Position = projection * vec4(vertex, 0.0, 1.0);
    TexCoord = texCoord;
}
)";

const char* text_fragment_shader = R"(
#version 410 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D text;
uniform vec3 textColor;

void main() {
    float alpha = texture(text, TexCoord).r;
    FragColor = vec4(textColor, alpha);
}
)";

TextRenderer::TextRenderer() 
    : shader_program_(0)
    , vao_(0)
    , vbo_(0)
    , font_texture_(0)
    , screen_width_(800)
    , screen_height_(600) {
}

TextRenderer::~TextRenderer() {
    shutdown();
}

bool TextRenderer::initialize(int screen_width, int screen_height) {
    screen_width_ = screen_width;
    screen_height_ = screen_height;
    
    // Initialize GLEW if not already done
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK && err != GLEW_ERROR_NO_GLX_DISPLAY) {
        std::cerr << "GLEW init error: " << glewGetErrorString(err) << std::endl;
        // Continue anyway, it might already be initialized
    }
    
    createShaders();
    createFontTexture();
    
    // Create VAO and VBO for rendering quads
    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_);
    
    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    
    // Reserve space for 6 vertices per character (2 triangles)
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4 * 256, NULL, GL_DYNAMIC_DRAW);
    
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);
    
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    
    return true;
}

void TextRenderer::shutdown() {
    if (vao_) glDeleteVertexArrays(1, &vao_);
    if (vbo_) glDeleteBuffers(1, &vbo_);
    if (font_texture_) glDeleteTextures(1, &font_texture_);
    if (shader_program_) glDeleteProgram(shader_program_);
    
    vao_ = vbo_ = font_texture_ = shader_program_ = 0;
}

void TextRenderer::createShaders() {
    // Create vertex shader
    GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex_shader, 1, &text_vertex_shader, NULL);
    glCompileShader(vertex_shader);
    
    // Check vertex shader compilation
    GLint success;
    glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char info_log[512];
        glGetShaderInfoLog(vertex_shader, 512, NULL, info_log);
        std::cerr << "Text vertex shader compilation failed: " << info_log << std::endl;
    }
    
    // Create fragment shader
    GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader, 1, &text_fragment_shader, NULL);
    glCompileShader(fragment_shader);
    
    // Check fragment shader compilation
    glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char info_log[512];
        glGetShaderInfoLog(fragment_shader, 512, NULL, info_log);
        std::cerr << "Text fragment shader compilation failed: " << info_log << std::endl;
    }
    
    // Create shader program
    shader_program_ = glCreateProgram();
    glAttachShader(shader_program_, vertex_shader);
    glAttachShader(shader_program_, fragment_shader);
    glLinkProgram(shader_program_);
    
    // Check linking
    glGetProgramiv(shader_program_, GL_LINK_STATUS, &success);
    if (!success) {
        char info_log[512];
        glGetProgramInfoLog(shader_program_, 512, NULL, info_log);
        std::cerr << "Text shader program linking failed: " << info_log << std::endl;
    }
    
    // Clean up shaders
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
    
    // Get uniform locations
    projection_loc_ = glGetUniformLocation(shader_program_, "projection");
    text_color_loc_ = glGetUniformLocation(shader_program_, "textColor");
    texture_loc_ = glGetUniformLocation(shader_program_, "text");
}

void TextRenderer::createFontTexture() {
    // Create texture atlas for ASCII characters 32-127
    const int chars_per_row = 16;
    const int char_width = 8;
    const int char_height = 8;
    const int atlas_width = chars_per_row * char_width;
    const int atlas_height = 6 * char_height; // 96 chars / 16 per row = 6 rows
    
    std::vector<unsigned char> atlas(atlas_width * atlas_height, 0);
    
    // Copy font bitmap data to atlas
    for (int ch = 32; ch <= 127; ++ch) {
        int char_index = ch - 32;
        int row = char_index / chars_per_row;
        int col = char_index % chars_per_row;
        
        for (int y = 0; y < 8; ++y) {
            unsigned char byte = font_bitmap_[char_index * 8 + y];
            for (int x = 0; x < 8; ++x) {
                if (byte & (0x80 >> x)) {
                    int atlas_x = col * char_width + x;
                    int atlas_y = row * char_height + y;
                    atlas[atlas_y * atlas_width + atlas_x] = 255;
                }
            }
        }
    }
    
    // Create OpenGL texture
    glGenTextures(1, &font_texture_);
    glBindTexture(GL_TEXTURE_2D, font_texture_);
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, atlas_width, atlas_height, 0, GL_RED, GL_UNSIGNED_BYTE, atlas.data());
    
    glBindTexture(GL_TEXTURE_2D, 0);
}

void TextRenderer::drawText(const std::string& text, float x, float y, float scale, glm::vec3 color) {
    // Save OpenGL state
    GLint old_blend_src, old_blend_dst;
    glGetIntegerv(GL_BLEND_SRC_ALPHA, &old_blend_src);
    glGetIntegerv(GL_BLEND_DST_ALPHA, &old_blend_dst);
    GLboolean old_depth_test = glIsEnabled(GL_DEPTH_TEST);
    
    // Set up state for text rendering
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_DEPTH_TEST);
    
    // Use shader program
    glUseProgram(shader_program_);
    
    // Set uniforms
    glm::mat4 projection = glm::ortho(0.0f, static_cast<float>(screen_width_), 
                                      static_cast<float>(screen_height_), 0.0f);
    glUniformMatrix4fv(projection_loc_, 1, GL_FALSE, &projection[0][0]);
    glUniform3fv(text_color_loc_, 1, &color[0]);
    glUniform1i(texture_loc_, 0);
    
    // Bind texture and VAO
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, font_texture_);
    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    
    // Build vertex data for all characters
    std::vector<float> vertices;
    vertices.reserve(text.length() * 6 * 4);
    
    float xpos = x;
    const float char_width = 8.0f * scale;
    const float char_height = 8.0f * scale;
    const float atlas_char_width = 8.0f / 128.0f;  // 128 = atlas width
    const float atlas_char_height = 8.0f / 48.0f;  // 48 = atlas height
    
    for (char c : text) {
        if (c < 32 || c > 127) continue;
        
        int char_index = c - 32;
        int row = char_index / 16;
        int col = char_index % 16;
        
        float u0 = col * atlas_char_width;
        float v0 = row * atlas_char_height;
        float u1 = u0 + atlas_char_width;
        float v1 = v0 + atlas_char_height;
        
        // First triangle
        vertices.push_back(xpos);            vertices.push_back(y);              vertices.push_back(u0); vertices.push_back(v0);
        vertices.push_back(xpos + char_width); vertices.push_back(y);              vertices.push_back(u1); vertices.push_back(v0);
        vertices.push_back(xpos);            vertices.push_back(y + char_height); vertices.push_back(u0); vertices.push_back(v1);
        
        // Second triangle
        vertices.push_back(xpos + char_width); vertices.push_back(y);              vertices.push_back(u1); vertices.push_back(v0);
        vertices.push_back(xpos + char_width); vertices.push_back(y + char_height); vertices.push_back(u1); vertices.push_back(v1);
        vertices.push_back(xpos);            vertices.push_back(y + char_height); vertices.push_back(u0); vertices.push_back(v1);
        
        xpos += char_width;
    }
    
    // Upload and draw
    if (!vertices.empty()) {
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.size() * sizeof(float), vertices.data());
        glDrawArrays(GL_TRIANGLES, 0, vertices.size() / 4);
    }
    
    // Restore state
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);
    
    if (old_depth_test) glEnable(GL_DEPTH_TEST);
    glBlendFunc(old_blend_src, old_blend_dst);
}

void TextRenderer::updateScreenSize(int width, int height) {
    screen_width_ = width;
    screen_height_ = height;
}

} // namespace SplatRender