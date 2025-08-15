#include "core/camera.h"
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>

namespace SplatRender {

Camera::Camera(glm::vec3 position, glm::vec3 up, float yaw, float pitch)
    : position_(position)
    , world_up_(up)
    , yaw_(yaw)
    , pitch_(pitch)
    , front_(0.0f, 0.0f, -1.0f)
    , movement_speed_(2.0f)
    , mouse_sensitivity_(0.05f)
    , fov_(60.0f)
    , near_plane_(0.01f)
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

void Camera::saveState(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for saving camera state: " << filename << std::endl;
        return;
    }
    
    file << "# SplatRender Camera State\n";
    file << "position " << position_.x << " " << position_.y << " " << position_.z << "\n";
    file << "yaw " << yaw_ << "\n";
    file << "pitch " << pitch_ << "\n";
    file << "fov " << fov_ << "\n";
    file << "movement_speed " << movement_speed_ << "\n";
    file << "mouse_sensitivity " << mouse_sensitivity_ << "\n";
    file << "near_plane " << near_plane_ << "\n";
    file << "far_plane " << far_plane_ << "\n";
    
    file.close();
    std::cout << "Camera state saved to: " << filename << std::endl;
}

bool Camera::loadState(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for loading camera state: " << filename << std::endl;
        return false;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        std::string key;
        iss >> key;
        
        if (key == "position") {
            iss >> position_.x >> position_.y >> position_.z;
        } else if (key == "yaw") {
            iss >> yaw_;
        } else if (key == "pitch") {
            iss >> pitch_;
        } else if (key == "fov") {
            iss >> fov_;
        } else if (key == "movement_speed") {
            iss >> movement_speed_;
        } else if (key == "mouse_sensitivity") {
            iss >> mouse_sensitivity_;
        } else if (key == "near_plane") {
            iss >> near_plane_;
        } else if (key == "far_plane") {
            iss >> far_plane_;
        }
    }
    
    file.close();
    updateCameraVectors();
    std::cout << "Camera state loaded from: " << filename << std::endl;
    return true;
}

} // namespace SplatRender