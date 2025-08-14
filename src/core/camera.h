#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace SplatRender {

enum class CameraMovement {
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT,
    UP,
    DOWN
};

class Camera {
public:
    Camera(glm::vec3 position = glm::vec3(0.0f, 0.0f, 3.0f),
           glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f),
           float yaw = -90.0f,
           float pitch = 0.0f);
    
    // View and projection matrices
    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjectionMatrix(float aspect_ratio) const;
    glm::mat4 getViewProjectionMatrix(float aspect_ratio) const;
    
    // Camera properties
    glm::vec3 getPosition() const { return position_; }
    glm::vec3 getFront() const { return front_; }
    glm::vec3 getUp() const { return up_; }
    glm::vec3 getRight() const { return right_; }
    
    float getYaw() const { return yaw_; }
    float getPitch() const { return pitch_; }
    float getFOV() const { return fov_; }
    float getNearPlane() const { return near_plane_; }
    float getFarPlane() const { return far_plane_; }
    
    // Camera control
    void processKeyboard(CameraMovement direction, float delta_time);
    void processMouseMovement(float xoffset, float yoffset, bool constrain_pitch = true);
    void processMouseScroll(float yoffset);
    
    // Camera settings
    void setPosition(const glm::vec3& position);
    void setFOV(float fov);
    void setClipPlanes(float near_plane, float far_plane);
    void setMovementSpeed(float speed) { movement_speed_ = speed; }
    void setMouseSensitivity(float sensitivity) { mouse_sensitivity_ = sensitivity; }

private:
    void updateCameraVectors();
    
    // Camera attributes
    glm::vec3 position_;
    glm::vec3 front_;
    glm::vec3 up_;
    glm::vec3 right_;
    glm::vec3 world_up_;
    
    // Euler angles
    float yaw_;
    float pitch_;
    
    // Camera options
    float movement_speed_;
    float mouse_sensitivity_;
    float fov_;
    float near_plane_;
    float far_plane_;
};

} // namespace SplatRender