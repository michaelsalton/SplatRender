#include "renderer/axis_renderer.h"
#include "core/camera.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>

namespace SplatRender {

const char* axis_vertex_shader = R"(
#version 330 core
layout (location = 0) in vec3 position;

uniform mat4 mvp;

void main() {
    gl_Position = mvp * vec4(position, 1.0);
}
)";

const char* axis_fragment_shader = R"(
#version 330 core
out vec4 FragColor;

uniform vec3 color;

void main() {
    FragColor = vec4(color, 1.0);
}
)";

AxisRenderer::AxisRenderer() 
    : shader_program_(0)
    , vao_(0)
    , vbo_(0) {
}

AxisRenderer::~AxisRenderer() {
    shutdown();
}

bool AxisRenderer::initialize() {
    // Initialize GLEW if needed
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK && err != GLEW_ERROR_NO_GLX_DISPLAY) {
        // Already initialized, continue
    }
    
    if (!createShaders()) {
        std::cerr << "Failed to create axis shaders" << std::endl;
        return false;
    }
    
    createGeometry();
    
    return true;
}

void AxisRenderer::shutdown() {
    if (vao_) glDeleteVertexArrays(1, &vao_);
    if (vbo_) glDeleteBuffers(1, &vbo_);
    if (shader_program_) glDeleteProgram(shader_program_);
    
    vao_ = vbo_ = shader_program_ = 0;
}

bool AxisRenderer::createShaders() {
    // Create vertex shader
    GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex_shader, 1, &axis_vertex_shader, NULL);
    glCompileShader(vertex_shader);
    
    // Check compilation
    GLint success;
    glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char info_log[512];
        glGetShaderInfoLog(vertex_shader, 512, NULL, info_log);
        std::cerr << "Axis vertex shader compilation failed: " << info_log << std::endl;
    }
    
    // Create fragment shader
    GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader, 1, &axis_fragment_shader, NULL);
    glCompileShader(fragment_shader);
    
    // Check compilation
    glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char info_log[512];
        glGetShaderInfoLog(fragment_shader, 512, NULL, info_log);
        std::cerr << "Axis fragment shader compilation failed: " << info_log << std::endl;
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
        std::cerr << "Axis shader program linking failed: " << info_log << std::endl;
    }
    
    // Clean up shaders
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
    
    // Get uniform locations
    mvp_loc_ = glGetUniformLocation(shader_program_, "mvp");
    color_loc_ = glGetUniformLocation(shader_program_, "color");
    
    if (mvp_loc_ == -1 || color_loc_ == -1) {
        std::cerr << "Failed to get uniform locations!" << std::endl;
        return false;
    }
    
    return true;
}

void AxisRenderer::createGeometry() {
    // Create axis lines
    float vertices[] = {
        // X axis (red)
        -5.0f, 0.0f, 0.0f,
         5.0f, 0.0f, 0.0f,
        // Y axis (green)
        0.0f, -5.0f, 0.0f,
        0.0f,  5.0f, 0.0f,
        // Z axis (blue)
        0.0f, 0.0f, -5.0f,
        0.0f, 0.0f,  5.0f,
    };
    
    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_);
    
    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    
    // Allocate buffer large enough for main axes + markers
    // Main axes: 6 vertices
    // Markers: 20 per axis * 3 axes * 2 vertices per marker = 120 vertices
    // Total: 126 vertices * 3 floats per vertex = 378 floats
    glBufferData(GL_ARRAY_BUFFER, 378 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    
    // Upload initial axis data
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
    
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void AxisRenderer::render(const Camera& camera, float aspect_ratio) {
    // Save OpenGL state
    GLint old_line_width;
    glGetIntegerv(GL_LINE_WIDTH, &old_line_width);
    GLboolean old_depth_test = glIsEnabled(GL_DEPTH_TEST);
    GLboolean old_blend = glIsEnabled(GL_BLEND);
    
    // Set up state for axis rendering
    glDisable(GL_DEPTH_TEST);  // Disable depth test so axes are always visible
    glDisable(GL_BLEND);  // Disable blending
    glLineWidth(10.0f);  // Make lines very thick
    
    // Use shader program
    glUseProgram(shader_program_);
    
    // Calculate MVP matrix
    glm::mat4 view = camera.getViewMatrix();
    glm::mat4 projection = camera.getProjectionMatrix(aspect_ratio);
    glm::mat4 mvp = projection * view;
    
    glUniformMatrix4fv(mvp_loc_, 1, GL_FALSE, glm::value_ptr(mvp));
    
    // Bind VAO
    glBindVertexArray(vao_);
    
    // Draw X axis (red)
    glUniform3f(color_loc_, 1.0f, 0.0f, 0.0f);
    glDrawArrays(GL_LINES, 0, 2);
    
    // Draw Y axis (green)
    glUniform3f(color_loc_, 0.0f, 1.0f, 0.0f);
    glDrawArrays(GL_LINES, 2, 2);
    
    // Draw Z axis (blue)
    glUniform3f(color_loc_, 0.0f, 0.0f, 1.0f);
    glDrawArrays(GL_LINES, 4, 2);
    
    // Skip marker drawing for now
    if (false) {
    // Draw small markers every unit along each axis
    glLineWidth(1.0f);
    
    // Create marker vertices
    float markers[6 * 3 * 20]; // 20 markers per axis, 3 axes, 2 vertices per marker, 3 coords per vertex
    int idx = 0;
    
    // X axis markers
    glUniform3f(color_loc_, 0.5f, 0.0f, 0.0f);
    for (int i = -10; i <= 10; i++) {
        if (i == 0) continue;
        markers[idx++] = i; markers[idx++] = -0.2f; markers[idx++] = 0.0f;
        markers[idx++] = i; markers[idx++] = 0.2f; markers[idx++] = 0.0f;
    }
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferSubData(GL_ARRAY_BUFFER, 0, idx * sizeof(float), markers);
    glDrawArrays(GL_LINES, 0, idx/3);
    
    // Y axis markers
    idx = 0;
    glUniform3f(color_loc_, 0.0f, 0.5f, 0.0f);
    for (int i = -10; i <= 10; i++) {
        if (i == 0) continue;
        markers[idx++] = -0.2f; markers[idx++] = i; markers[idx++] = 0.0f;
        markers[idx++] = 0.2f; markers[idx++] = i; markers[idx++] = 0.0f;
    }
    glBufferSubData(GL_ARRAY_BUFFER, 0, idx * sizeof(float), markers);
    glDrawArrays(GL_LINES, 0, idx/3);
    
    // Z axis markers
    idx = 0;
    glUniform3f(color_loc_, 0.0f, 0.0f, 0.5f);
    for (int i = -10; i <= 10; i++) {
        if (i == 0) continue;
        markers[idx++] = -0.2f; markers[idx++] = 0.0f; markers[idx++] = i;
        markers[idx++] = 0.2f; markers[idx++] = 0.0f; markers[idx++] = i;
    }
    glBufferSubData(GL_ARRAY_BUFFER, 0, idx * sizeof(float), markers);
    glDrawArrays(GL_LINES, 0, idx/3);
    }
    
    // Restore state
    glBindVertexArray(0);
    glUseProgram(0);
    glLineWidth(old_line_width);
    if (old_depth_test) glEnable(GL_DEPTH_TEST);
    if (old_blend) glEnable(GL_BLEND);
}

} // namespace SplatRender