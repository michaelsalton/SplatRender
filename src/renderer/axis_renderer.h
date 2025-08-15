#pragma once

#include <GL/glew.h>
#include <glm/glm.hpp>

namespace SplatRender {

class Camera;

class AxisRenderer {
public:
    AxisRenderer();
    ~AxisRenderer();
    
    bool initialize();
    void shutdown();
    
    // Draw coordinate axes at origin
    void render(const Camera& camera, float aspect_ratio);
    
private:
    GLuint shader_program_;
    GLuint vao_, vbo_;
    
    // Shader uniform locations
    GLint mvp_loc_;
    GLint color_loc_;
    
    void createShaders();
    void createGeometry();
};

} // namespace SplatRender