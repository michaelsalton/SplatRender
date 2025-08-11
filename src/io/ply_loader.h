#pragma once

#include <string>
#include <vector>
#include <memory>
#include "math/gaussian.h"

namespace SplatRender {

class PLYLoader {
public:
    struct PLYHeader {
        enum class Format {
            ASCII,
            BINARY_LITTLE_ENDIAN,
            BINARY_BIG_ENDIAN
        };
        
        Format format;
        size_t vertex_count;
        
        // Property indices in the vertex data
        int position_x_idx = -1;
        int position_y_idx = -1;
        int position_z_idx = -1;
        
        int scale_0_idx = -1;
        int scale_1_idx = -1;
        int scale_2_idx = -1;
        
        int rot_0_idx = -1;
        int rot_1_idx = -1;
        int rot_2_idx = -1;
        int rot_3_idx = -1;
        
        int f_dc_0_idx = -1;
        int f_dc_1_idx = -1;
        int f_dc_2_idx = -1;
        
        int opacity_idx = -1;
        
        // Spherical harmonic coefficients (45 total for RGB)
        std::vector<int> f_rest_indices;
        
        size_t bytes_per_vertex = 0;
        size_t header_size = 0;
    };
    
    PLYLoader();
    ~PLYLoader();
    
    // Load Gaussians from PLY file
    bool load(const std::string& filename, std::vector<Gaussian3D>& gaussians);
    
    // Get last error message
    const std::string& getLastError() const { return last_error_; }
    
    // Progress callback (optional)
    using ProgressCallback = std::function<void(float progress)>;
    void setProgressCallback(ProgressCallback callback) { progress_callback_ = callback; }

private:
    bool parseHeader(std::ifstream& file, PLYHeader& header);
    bool parseProperties(const std::string& line, PLYHeader& header, int& property_index);
    bool readBinaryData(std::ifstream& file, const PLYHeader& header, std::vector<Gaussian3D>& gaussians);
    bool readASCIIData(std::ifstream& file, const PLYHeader& header, std::vector<Gaussian3D>& gaussians);
    
    Gaussian3D parseVertex(const std::vector<float>& vertex_data, const PLYHeader& header);
    
    std::string last_error_;
    ProgressCallback progress_callback_;
};

} // namespace SplatRender