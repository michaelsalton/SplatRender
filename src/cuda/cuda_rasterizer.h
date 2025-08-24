#pragma once

#include "cuda_memory.h"
#include "cuda_gl_interop.h"
#include "../renderer/cpu_rasterizer.h"
#include "../math/gaussian.h"
#include <memory>
#include <vector>

namespace SplatRender {
namespace CUDA {

// CUDA version of the rasterizer
class CudaRasterizer {
public:
    CudaRasterizer();
    ~CudaRasterizer();
    
    // Initialize with render settings
    bool initialize(const RenderSettings& settings);
    void shutdown();
    
    // Main render function (matches CPU interface)
    void render(const std::vector<Gaussian3D>& gaussians,
                const Camera& camera,
                std::vector<float>& output_buffer);
    
    // Direct GPU rendering (no CPU buffer copy)
    void renderDirect(const Gaussian3D* d_gaussians, 
                     size_t count,
                     const Camera& camera,
                     cudaSurfaceObject_t surface);
    
    // Render to OpenGL texture directly
    void renderToTexture(const std::vector<Gaussian3D>& gaussians,
                        const Camera& camera,
                        GLuint texture);
    
    // Memory management
    void uploadGaussians(const std::vector<Gaussian3D>& gaussians);
    void allocateBuffers(size_t max_gaussians);
    void freeBuffers();
    
    // Get/set settings
    const RenderSettings& getSettings() const { return settings_; }
    void setSettings(const RenderSettings& settings);
    
    // Performance statistics
    struct Stats {
        float upload_time_ms = 0.0f;
        float projection_time_ms = 0.0f;
        float sorting_time_ms = 0.0f;
        float rasterization_time_ms = 0.0f;
        float download_time_ms = 0.0f;
        float total_time_ms = 0.0f;
        int visible_gaussians = 0;
        int culled_gaussians = 0;
        size_t memory_used_bytes = 0;
    };
    
    const Stats& getStats() const { return stats_; }
    void resetStats() { stats_ = Stats(); }
    
    // Check if CUDA is available and initialized
    bool isAvailable() const { return initialized_; }
    
private:
    // Settings
    RenderSettings settings_;
    bool initialized_;
    size_t max_gaussians_;
    
    // Device memory buffers
    CudaMemory<Gaussian3D> d_gaussians_3d_;     // Input 3D Gaussians
    CudaMemory<Gaussian2D> d_gaussians_2d_;     // Projected 2D Gaussians
    CudaMemory<float> d_output_buffer_;         // Output image buffer
    CudaMemory<float> d_depth_buffer_;          // Depth buffer
    CudaMemory<uint32_t> d_tile_lists_;         // Tile assignment lists
    CudaMemory<uint32_t> d_tile_counts_;        // Gaussians per tile
    CudaMemory<int> d_visible_indices_;         // Indices of visible Gaussians
    
    // Host pinned memory for faster transfers
    std::unique_ptr<PinnedMemory<float>> h_pinned_output_;
    
    // OpenGL interop
    std::unique_ptr<CudaGLInterop> gl_interop_;
    
    // Statistics
    Stats stats_;
    std::unique_ptr<CudaTimer> timer_;
    
    // Helper functions
    void ensureBufferSizes(size_t num_gaussians);
    int getTileCount() const;
};

// Factory function to create appropriate rasterizer based on CUDA availability
std::unique_ptr<CPURasterizer> createRasterizer(bool prefer_cuda = true);

} // namespace CUDA
} // namespace SplatRender