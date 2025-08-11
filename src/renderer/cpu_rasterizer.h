#pragma once

#include <vector>
#include <memory>
#include "math/gaussian.h"

namespace SplatRender {

class Camera;

struct RenderSettings {
    int width = 1920;
    int height = 1080;
    float alpha_threshold = 0.01f;  // Minimum alpha to render
    int tile_size = 16;             // Tile size for optimization
    bool enable_depth_test = true;
    bool enable_culling = true;
};

class CPURasterizer {
public:
    CPURasterizer();
    ~CPURasterizer();
    
    // Initialize rasterizer with given settings
    void initialize(const RenderSettings& settings);
    
    // Render Gaussians to output buffer
    void render(const std::vector<Gaussian3D>& gaussians,
                const Camera& camera,
                std::vector<float>& output_buffer);
    
    // Get/set render settings
    const RenderSettings& getSettings() const { return settings_; }
    void setSettings(const RenderSettings& settings) { settings_ = settings; }
    
    // Performance statistics
    struct Stats {
        float projection_time_ms = 0.0f;
        float sorting_time_ms = 0.0f;
        float rasterization_time_ms = 0.0f;
        float total_time_ms = 0.0f;
        int visible_gaussians = 0;
        int culled_gaussians = 0;
    };
    
    const Stats& getStats() const { return stats_; }

private:
    // Rendering pipeline stages
    void projectGaussians(const std::vector<Gaussian3D>& gaussians_3d,
                         const Camera& camera,
                         std::vector<Gaussian2D>& gaussians_2d);
    
    void cullGaussians(std::vector<Gaussian2D>& gaussians);
    
    void rasterizeGaussians(const std::vector<Gaussian2D>& gaussians,
                           std::vector<float>& output_buffer);
    
    // Helper functions
    void clearBuffer(std::vector<float>& buffer);
    bool isGaussianVisible(const Gaussian2D& gaussian) const;
    
    RenderSettings settings_;
    Stats stats_;
    
    // Temporary buffers
    std::vector<Gaussian2D> projected_gaussians_;
    std::vector<float> depth_buffer_;
};

} // namespace SplatRender