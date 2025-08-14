#include "core/camera.h"
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>

namespace SplatRender {

Camera::Camera(glm::vec3 position, glm::vec3 up, float yaw, float pitch)
    : position_(position)
    , world_up_(up)
    , yaw_(yaw)
    , pitch_(pitch)
    , front_(0.0f, 0.0f, -1.0f)
    , movement_speed_(5.0f)
    , mouse_sensitivity_(0.1f)
    , fov_(60.0f)
    , near_plane_(0.1f)
    , far_plane_(1000.0f) {
    updateCameraVectors();
}

void Camera::setPosition(const glm::vec3& position) {
    position_ = position;
}


void Camera::setFOV(float fov) {
    fov_ = glm::clamp(fov, 1.0f, 120.0f);
}

void Camera::setClipPlanes(float near_plane, float far_plane) {
    near_plane_ = near_plane;
    far_plane_ = far_plane;
}

void Camera::processKeyboard(CameraMovement direction, float delta_time) {
    float velocity = movement_speed_ * delta_time;
    
    switch (direction) {
        case CameraMovement::FORWARD:
            position_ += front_ * velocity;
            break;
        case CameraMovement::BACKWARD:
            position_ -= front_ * velocity;
            break;
        case CameraMovement::LEFT:
            position_ -= right_ * velocity;
            break;
        case CameraMovement::RIGHT:
            position_ += right_ * velocity;
            break;
        case CameraMovement::UP:
            position_ += world_up_ * velocity;
            break;
        case CameraMovement::DOWN:
            position_ -= world_up_ * velocity;
            break;
    }
}

void Camera::processMouseMovement(float xoffset, float yoffset, bool constrain_pitch) {
    xoffset *= mouse_sensitivity_;
    yoffset *= mouse_sensitivity_;
    
    yaw_ += xoffset;
    pitch_ += yoffset;
    
    if (constrain_pitch) {
        pitch_ = glm::clamp(pitch_, -89.0f, 89.0f);
    }
    
    updateCameraVectors();
}

void Camera::processMouseScroll(float yoffset) {
    fov_ -= yoffset;
    fov_ = glm::clamp(fov_, 1.0f, 120.0f);
}

glm::mat4 Camera::getViewMatrix() const {
    return glm::lookAt(position_, position_ + front_, up_);
}

glm::mat4 Camera::getProjectionMatrix(float aspect_ratio) const {
    return glm::perspective(glm::radians(fov_), aspect_ratio, near_plane_, far_plane_);
}

glm::mat4 Camera::getViewProjectionMatrix(float aspect_ratio) const {
    return getProjectionMatrix(aspect_ratio) * getViewMatrix();
}


void Camera::updateCameraVectors() {
    // Calculate new front vector
    glm::vec3 new_front;
    new_front.x = cos(glm::radians(yaw_)) * cos(glm::radians(pitch_));
    new_front.y = sin(glm::radians(pitch_));
    new_front.z = sin(glm::radians(yaw_)) * cos(glm::radians(pitch_));
    front_ = glm::normalize(new_front);
    
    // Recalculate right and up vectors
    right_ = glm::normalize(glm::cross(front_, world_up_));
    up_ = glm::normalize(glm::cross(right_, front_));
}

} // namespace SplatRender