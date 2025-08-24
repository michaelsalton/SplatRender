#include "cuda_rasterizer.h"
#include "cuda_manager.h"
#include "../core/camera.h"
#include <iostream>
#include <algorithm>

namespace SplatRender {
namespace CUDA {

// Placeholder kernels - will be implemented in Phase 8
__global__ void projectGaussiansKernel(const Gaussian3D* gaussians_3d,
                                       Gaussian2D* gaussians_2d,
                                       int* visible_indices,
                                       int count,
                                       const float* view_matrix,
                                       const float* proj_matrix,
                                       int width, int height) {
    // TODO: Implement in Phase 8
}

__global__ void rasterizeKernel(const Gaussian2D* gaussians_2d,
                                float* output_buffer,
                                int count,
                                int width, int height) {
    // TODO: Implement in Phase 8
}

CudaRasterizer::CudaRasterizer() 
    : initialized_(false), max_gaussians_(0) {
    timer_ = std::make_unique<CudaTimer>();
}

CudaRasterizer::~CudaRasterizer() {
    shutdown();
}

bool CudaRasterizer::initialize(const RenderSettings& settings) {
    if (initialized_) {
        return true;
    }
    
    // Check if CUDA is available
    if (!CudaManager::getInstance().isInitialized()) {
        if (!CudaManager::getInstance().initialize()) {
            std::cerr << "Failed to initialize CUDA" << std::endl;
            return false;
        }
    }
    
    settings_ = settings;
    
    // Allocate initial buffers (will grow as needed)
    size_t initial_capacity = 10000;
    allocateBuffers(initial_capacity);
    
    // Create pinned memory for output
    size_t output_size = settings.width * settings.height * 4; // RGBA
    h_pinned_output_ = std::make_unique<PinnedMemory<float>>(output_size);
    
    // Initialize OpenGL interop
    gl_interop_ = std::make_unique<CudaGLInterop>();
    
    initialized_ = true;
    
    std::cout << "CUDA Rasterizer initialized successfully" << std::endl;
    std::cout << "  Resolution: " << settings.width << "x" << settings.height << std::endl;
    std::cout << "  Tile Size: " << settings.tile_size << "x" << settings.tile_size << std::endl;
    std::cout << "  Initial Capacity: " << initial_capacity << " Gaussians" << std::endl;
    
    return true;
}

void CudaRasterizer::shutdown() {
    if (!initialized_) {
        return;
    }
    
    freeBuffers();
    h_pinned_output_.reset();
    gl_interop_.reset();
    
    initialized_ = false;
}

void CudaRasterizer::render(const std::vector<Gaussian3D>& gaussians,
                            const Camera& camera,
                            std::vector<float>& output_buffer) {
    if (!initialized_) {
        std::cerr << "CUDA Rasterizer not initialized" << std::endl;
        return;
    }
    
    if (gaussians.empty()) {
        // Clear output buffer
        size_t buffer_size = settings_.width * settings_.height * 4;
        output_buffer.resize(buffer_size);
        std::fill(output_buffer.begin(), output_buffer.end(), 0.0f);
        return;
    }
    
    timer_->start();
    
    // Upload Gaussians to GPU
    uploadGaussians(gaussians);
    timer_->stop();
    stats_.upload_time_ms = timer_->getElapsedMs();
    
    // Clear output buffer on GPU
    size_t output_size = settings_.width * settings_.height * 4;
    d_output_buffer_.clear();
    
    // TODO: Launch projection kernel (Phase 8)
    // For now, just clear the output
    
    // TODO: Launch rasterization kernel (Phase 8)
    
    // Download result to CPU
    timer_->start();
    output_buffer.resize(output_size);
    d_output_buffer_.copyToHost(output_buffer.data(), output_size);
    timer_->stop();
    stats_.download_time_ms = timer_->getElapsedMs();
    
    // Update total time
    stats_.total_time_ms = stats_.upload_time_ms + stats_.projection_time_ms + 
                           stats_.sorting_time_ms + stats_.rasterization_time_ms + 
                           stats_.download_time_ms;
    
    // Update memory usage
    stats_.memory_used_bytes = d_gaussians_3d_.allocatedBytes() + 
                               d_gaussians_2d_.allocatedBytes() +
                               d_output_buffer_.allocatedBytes() +
                               d_depth_buffer_.allocatedBytes();
}

void CudaRasterizer::renderDirect(const Gaussian3D* d_gaussians, 
                                  size_t count,
                                  const Camera& camera,
                                  cudaSurfaceObject_t surface) {
    if (!initialized_) {
        return;
    }
    
    // This will be implemented with actual kernels in Phase 8
    // For now, it's a placeholder
}

void CudaRasterizer::renderToTexture(const std::vector<Gaussian3D>& gaussians,
                                     const Camera& camera,
                                     GLuint texture) {
    if (!initialized_ || !gl_interop_) {
        return;
    }
    
    // Register texture if not already registered
    if (!gl_interop_->isTextureRegistered()) {
        gl_interop_->registerTexture(texture, settings_.width, settings_.height);
    }
    
    // Upload Gaussians
    uploadGaussians(gaussians);
    
    // Map texture for CUDA
    cudaSurfaceObject_t surface = gl_interop_->mapTextureForCuda();
    
    // Render directly to surface
    renderDirect(d_gaussians_3d_.getDevicePtr(), gaussians.size(), camera, surface);
    
    // Unmap texture
    gl_interop_->unmapTexture();
}

void CudaRasterizer::uploadGaussians(const std::vector<Gaussian3D>& gaussians) {
    ensureBufferSizes(gaussians.size());
    d_gaussians_3d_.copyFromHost(gaussians.data(), gaussians.size());
    stats_.visible_gaussians = static_cast<int>(gaussians.size());
}

void CudaRasterizer::allocateBuffers(size_t max_gaussians) {
    max_gaussians_ = max_gaussians;
    
    // Allocate device buffers
    d_gaussians_3d_.allocate(max_gaussians);
    d_gaussians_2d_.allocate(max_gaussians);
    d_visible_indices_.allocate(max_gaussians);
    
    // Allocate output and depth buffers
    size_t pixel_count = settings_.width * settings_.height;
    d_output_buffer_.allocate(pixel_count * 4); // RGBA
    d_depth_buffer_.allocate(pixel_count);
    
    // Allocate tile buffers
    int tile_count = getTileCount();
    d_tile_lists_.allocate(tile_count * max_gaussians); // Worst case
    d_tile_counts_.allocate(tile_count);
}

void CudaRasterizer::freeBuffers() {
    d_gaussians_3d_.free();
    d_gaussians_2d_.free();
    d_output_buffer_.free();
    d_depth_buffer_.free();
    d_tile_lists_.free();
    d_tile_counts_.free();
    d_visible_indices_.free();
}

void CudaRasterizer::setSettings(const RenderSettings& settings) {
    if (settings.width != settings_.width || settings.height != settings_.height) {
        // Need to reallocate output buffers
        settings_ = settings;
        
        size_t pixel_count = settings_.width * settings_.height;
        d_output_buffer_.allocate(pixel_count * 4);
        d_depth_buffer_.allocate(pixel_count);
        
        int tile_count = getTileCount();
        d_tile_lists_.allocate(tile_count * max_gaussians_);
        d_tile_counts_.allocate(tile_count);
        
        // Reallocate pinned memory
        h_pinned_output_ = std::make_unique<PinnedMemory<float>>(pixel_count * 4);
    } else {
        settings_ = settings;
    }
}

void CudaRasterizer::ensureBufferSizes(size_t num_gaussians) {
    if (num_gaussians > max_gaussians_) {
        // Grow buffers by 50% to avoid frequent reallocations
        size_t new_capacity = static_cast<size_t>(num_gaussians * 1.5);
        allocateBuffers(new_capacity);
    }
}

int CudaRasterizer::getTileCount() const {
    int tiles_x = (settings_.width + settings_.tile_size - 1) / settings_.tile_size;
    int tiles_y = (settings_.height + settings_.tile_size - 1) / settings_.tile_size;
    return tiles_x * tiles_y;
}

// Factory function
std::unique_ptr<CPURasterizer> createRasterizer(bool prefer_cuda) {
    if (prefer_cuda && isCudaAvailable()) {
        // For now, return CPU rasterizer until CUDA kernels are implemented
        // In Phase 8, this will return a wrapper that uses CudaRasterizer
        std::cout << "CUDA available but kernels not yet implemented, using CPU rasterizer" << std::endl;
    }
    return std::make_unique<CPURasterizer>();
}

} // namespace CUDA
} // namespace SplatRender