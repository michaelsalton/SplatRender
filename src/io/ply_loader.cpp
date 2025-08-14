#include "io/ply_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cstring>

namespace SplatRender {

PLYLoader::PLYLoader() : progress_callback_(nullptr) {
}

PLYLoader::~PLYLoader() {
}

bool PLYLoader::load(const std::string& filename, std::vector<Gaussian3D>& gaussians) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        last_error_ = "Failed to open file: " + filename;
        return false;
    }
    
    PLYHeader header;
    if (!parseHeader(file, header)) {
        file.close();
        return false;
    }
    
    // Reserve space for Gaussians
    gaussians.clear();
    gaussians.reserve(header.vertex_count);
    
    bool success = false;
    if (header.format == PLYHeader::Format::ASCII) {
        success = readASCIIData(file, header, gaussians);
    } else {
        success = readBinaryData(file, header, gaussians);
    }
    
    file.close();
    
    if (success) {
        std::cout << "Loaded " << gaussians.size() << " Gaussians from " << filename << std::endl;
    }
    
    return success;
}

bool PLYLoader::parseHeader(std::ifstream& file, PLYHeader& header) {
    std::string line;
    
    // Check for PLY magic number
    std::getline(file, line);
    if (line != "ply") {
        last_error_ = "Invalid PLY file: missing 'ply' header";
        return false;
    }
    
    // Parse format
    std::getline(file, line);
    if (line.find("format ascii") == 0) {
        header.format = PLYHeader::Format::ASCII;
    } else if (line.find("format binary_little_endian") == 0) {
        header.format = PLYHeader::Format::BINARY_LITTLE_ENDIAN;
    } else if (line.find("format binary_big_endian") == 0) {
        header.format = PLYHeader::Format::BINARY_BIG_ENDIAN;
    } else {
        last_error_ = "Unsupported PLY format: " + line;
        return false;
    }
    
    int property_index = 0;
    bool in_vertex_element = false;
    
    while (std::getline(file, line)) {
        if (line == "end_header") {
            header.header_size = file.tellg();
            break;
        }
        
        // Parse element vertex count
        if (line.find("element vertex") == 0) {
            std::istringstream iss(line);
            std::string element, vertex;
            iss >> element >> vertex >> header.vertex_count;
            in_vertex_element = true;
            property_index = 0;
        } else if (line.find("element") == 0) {
            in_vertex_element = false;
        }
        
        // Parse properties
        if (in_vertex_element && line.find("property") == 0) {
            if (!parseProperties(line, header, property_index)) {
                return false;
            }
            property_index++;
        }
    }
    
    // Calculate bytes per vertex for binary format
    if (header.format != PLYHeader::Format::ASCII) {
        header.bytes_per_vertex = property_index * sizeof(float);
    }
    
    // Initialize f_rest_indices if needed
    if (header.f_rest_indices.empty()) {
        // Pre-allocate for 45 SH coefficients (15 per channel, 3 channels)
        header.f_rest_indices.resize(45, -1);
    }
    
    return true;
}

bool PLYLoader::parseProperties(const std::string& line, PLYHeader& header, int& property_index) {
    std::istringstream iss(line);
    std::string property, type, name;
    iss >> property >> type >> name;
    
    if (type != "float") {
        last_error_ = "Unsupported property type: " + type;
        return false;
    }
    
    // Map property names to indices
    if (name == "x") {
        header.position_x_idx = property_index;
    } else if (name == "y") {
        header.position_y_idx = property_index;
    } else if (name == "z") {
        header.position_z_idx = property_index;
    } else if (name == "scale_0") {
        header.scale_0_idx = property_index;
    } else if (name == "scale_1") {
        header.scale_1_idx = property_index;
    } else if (name == "scale_2") {
        header.scale_2_idx = property_index;
    } else if (name == "rot_0") {
        header.rot_0_idx = property_index;
    } else if (name == "rot_1") {
        header.rot_1_idx = property_index;
    } else if (name == "rot_2") {
        header.rot_2_idx = property_index;
    } else if (name == "rot_3") {
        header.rot_3_idx = property_index;
    } else if (name == "f_dc_0") {
        header.f_dc_0_idx = property_index;
    } else if (name == "f_dc_1") {
        header.f_dc_1_idx = property_index;
    } else if (name == "f_dc_2") {
        header.f_dc_2_idx = property_index;
    } else if (name == "opacity") {
        header.opacity_idx = property_index;
    } else if (name.find("f_rest_") == 0) {
        // Parse SH coefficient index
        int sh_idx = std::stoi(name.substr(7));
        if (sh_idx >= 0 && sh_idx < 45) {
            header.f_rest_indices[sh_idx] = property_index;
        }
    }
    
    return true;
}

bool PLYLoader::readBinaryData(std::ifstream& file, const PLYHeader& header, std::vector<Gaussian3D>& gaussians) {
    // Seek to start of binary data
    file.seekg(header.header_size);
    
    // Buffer for one vertex
    std::vector<float> vertex_buffer(header.bytes_per_vertex / sizeof(float));
    
    for (size_t i = 0; i < header.vertex_count; ++i) {
        // Read vertex data
        file.read(reinterpret_cast<char*>(vertex_buffer.data()), header.bytes_per_vertex);
        
        if (!file.good()) {
            last_error_ = "Failed to read binary data at vertex " + std::to_string(i);
            return false;
        }
        
        // Handle endianness if needed
        if (header.format == PLYHeader::Format::BINARY_BIG_ENDIAN) {
            // Swap bytes for big-endian
            for (float& value : vertex_buffer) {
                char* bytes = reinterpret_cast<char*>(&value);
                std::swap(bytes[0], bytes[3]);
                std::swap(bytes[1], bytes[2]);
            }
        }
        
        // Parse vertex into Gaussian
        Gaussian3D gaussian = parseVertex(vertex_buffer, header);
        gaussians.push_back(gaussian);
        
        // Report progress
        if (progress_callback_ && i % 1000 == 0) {
            float progress = static_cast<float>(i) / static_cast<float>(header.vertex_count);
            progress_callback_(progress);
        }
    }
    
    // Final progress callback
    if (progress_callback_) {
        progress_callback_(1.0f);
    }
    
    return true;
}

bool PLYLoader::readASCIIData(std::ifstream& file, const PLYHeader& header, std::vector<Gaussian3D>& gaussians) {
    std::string line;
    std::vector<float> vertex_buffer;
    
    for (size_t i = 0; i < header.vertex_count; ++i) {
        if (!std::getline(file, line)) {
            last_error_ = "Failed to read ASCII data at vertex " + std::to_string(i);
            return false;
        }
        
        // Parse line into floats
        vertex_buffer.clear();
        std::istringstream iss(line);
        float value;
        while (iss >> value) {
            vertex_buffer.push_back(value);
        }
        
        // Parse vertex into Gaussian
        Gaussian3D gaussian = parseVertex(vertex_buffer, header);
        gaussians.push_back(gaussian);
        
        // Report progress
        if (progress_callback_ && i % 1000 == 0) {
            float progress = static_cast<float>(i) / static_cast<float>(header.vertex_count);
            progress_callback_(progress);
        }
    }
    
    // Final progress callback
    if (progress_callback_) {
        progress_callback_(1.0f);
    }
    
    return true;
}

Gaussian3D PLYLoader::parseVertex(const std::vector<float>& vertex_data, const PLYHeader& header) {
    Gaussian3D gaussian;
    
    // Extract position
    if (header.position_x_idx >= 0 && static_cast<size_t>(header.position_x_idx) < vertex_data.size()) {
        gaussian.position.x = vertex_data[header.position_x_idx];
    }
    if (header.position_y_idx >= 0 && static_cast<size_t>(header.position_y_idx) < vertex_data.size()) {
        gaussian.position.y = vertex_data[header.position_y_idx];
    }
    if (header.position_z_idx >= 0 && static_cast<size_t>(header.position_z_idx) < vertex_data.size()) {
        gaussian.position.z = vertex_data[header.position_z_idx];
    }
    
    // Extract scale
    if (header.scale_0_idx >= 0 && static_cast<size_t>(header.scale_0_idx) < vertex_data.size()) {
        gaussian.scale.x = std::exp(vertex_data[header.scale_0_idx]); // Scale is stored as log
    }
    if (header.scale_1_idx >= 0 && static_cast<size_t>(header.scale_1_idx) < vertex_data.size()) {
        gaussian.scale.y = std::exp(vertex_data[header.scale_1_idx]);
    }
    if (header.scale_2_idx >= 0 && static_cast<size_t>(header.scale_2_idx) < vertex_data.size()) {
        gaussian.scale.z = std::exp(vertex_data[header.scale_2_idx]);
    }
    
    // Extract rotation (quaternion)
    if (header.rot_0_idx >= 0 && static_cast<size_t>(header.rot_0_idx) < vertex_data.size()) {
        gaussian.rotation.w = vertex_data[header.rot_0_idx];
    }
    if (header.rot_1_idx >= 0 && static_cast<size_t>(header.rot_1_idx) < vertex_data.size()) {
        gaussian.rotation.x = vertex_data[header.rot_1_idx];
    }
    if (header.rot_2_idx >= 0 && static_cast<size_t>(header.rot_2_idx) < vertex_data.size()) {
        gaussian.rotation.y = vertex_data[header.rot_2_idx];
    }
    if (header.rot_3_idx >= 0 && static_cast<size_t>(header.rot_3_idx) < vertex_data.size()) {
        gaussian.rotation.z = vertex_data[header.rot_3_idx];
    }
    
    // Normalize quaternion
    float quat_length = glm::length(gaussian.rotation);
    if (quat_length > 0.0f) {
        gaussian.rotation /= quat_length;
    } else {
        gaussian.rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f); // Identity
    }
    
    // Extract SH DC coefficients (0th degree)
    if (header.f_dc_0_idx >= 0 && static_cast<size_t>(header.f_dc_0_idx) < vertex_data.size()) {
        gaussian.sh_coeffs[0] = vertex_data[header.f_dc_0_idx];
    }
    if (header.f_dc_1_idx >= 0 && static_cast<size_t>(header.f_dc_1_idx) < vertex_data.size()) {
        gaussian.sh_coeffs[1] = vertex_data[header.f_dc_1_idx];
    }
    if (header.f_dc_2_idx >= 0 && static_cast<size_t>(header.f_dc_2_idx) < vertex_data.size()) {
        gaussian.sh_coeffs[2] = vertex_data[header.f_dc_2_idx];
    }
    
    // Extract remaining SH coefficients
    for (size_t i = 0; i < header.f_rest_indices.size(); ++i) {
        int idx = header.f_rest_indices[i];
        if (idx >= 0 && static_cast<size_t>(idx) < vertex_data.size()) {
            // f_rest starts at SH index 3 (after DC terms)
            // Make sure we don't exceed the sh_coeffs array bounds
            size_t sh_idx = i + 3;
            if (sh_idx < gaussian.sh_coeffs.size()) {
                gaussian.sh_coeffs[sh_idx] = vertex_data[idx];
            }
        }
    }
    
    // Extract opacity and apply sigmoid
    if (header.opacity_idx >= 0 && static_cast<size_t>(header.opacity_idx) < vertex_data.size()) {
        float opacity_logit = vertex_data[header.opacity_idx];
        gaussian.opacity = 1.0f / (1.0f + std::exp(-opacity_logit)); // Sigmoid
    } else {
        gaussian.opacity = 1.0f;
    }
    
    // Clamp opacity to valid range
    gaussian.opacity = std::max(0.0f, std::min(1.0f, gaussian.opacity));
    
    return gaussian;
}

} // namespace SplatRender