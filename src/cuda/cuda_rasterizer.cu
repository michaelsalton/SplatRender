#include "cuda_rasterizer.h"
#include "cuda_rasterizer_wrapper.h"
#include "cuda_manager.h"
#include "cuda_constants.h"
#include "kernels/kernels.h"
#include "../core/camera.h"
#include <iostream>
#include <algorithm>
#include <glm/gtc/type_ptr.hpp>

namespace SplatRender {
namespace CUDA {

// Helper function to convert Gaussian3D to GaussianData3D
void convertGaussians(const std::vector<Gaussian3D>& gaussians, GaussianData3D* output) {
    for (size_t i = 0; i < gaussians.size(); i++) {
        const Gaussian3D& src = gaussians[i];
        GaussianData3D& dst = output[i];
        
        dst.position = make_float3(src.position.x, src.position.y, src.position.z);
        dst.opacity = src.opacity;
        dst.scale = make_float3(src.scale.x, src.scale.y, src.scale.z);
        dst.rotation = make_float4(src.rotation.x, src.rotation.y, src.rotation.z, src.rotation.w);
        
        // Copy spherical harmonics coefficients
        for (int j = 0; j < 45; j++) {
            dst.sh_coeffs[j] = src.sh_coeffs[j];
        }
        // Zero out unused coefficients
        for (int j = 45; j < 48; j++) {
            dst.sh_coeffs[j] = 0.0f;
        }
    }
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
    
    // Ensure buffers are large enough
    ensureBufferSizes(gaussians.size());
    
    // Get CUDA stream
    cudaStream_t stream = CudaManager::getInstance().getDefaultStream();
    
    // ========================================================================
    // Step 1: Upload Gaussians to GPU
    // ========================================================================
    timer_->start();
    
    // Allocate pinned memory for conversion
    PinnedMemory<GaussianData3D> h_gaussians_3d(gaussians.size());
    convertGaussians(gaussians, h_gaussians_3d.getHostPtr());
    
    // Upload to device
    CudaMemory<GaussianData3D> d_gaussians_3d_data(gaussians.size());
    d_gaussians_3d_data.copyFromHostAsync(h_gaussians_3d.getHostPtr(), gaussians.size(), stream);
    
    timer_->stop();
    stats_.upload_time_ms = timer_->getElapsedMs();
    
    // ========================================================================
    // Step 2: Projection Kernel
    // ========================================================================
    timer_->start();
    
    // Create camera parameters
    float aspect_ratio = static_cast<float>(settings_.width) / static_cast<float>(settings_.height);
    glm::mat4 view_matrix = camera.getViewMatrix();
    glm::mat4 proj_matrix = camera.getProjectionMatrix(aspect_ratio);
    glm::vec3 cam_pos = camera.getPosition();
    float cam_pos_array[3] = {cam_pos.x, cam_pos.y, cam_pos.z};
    
    CameraParams camera_params = createCameraParams(
        glm::value_ptr(view_matrix),
        glm::value_ptr(proj_matrix),
        cam_pos_array,
        camera.getFOV() * M_PI / 180.0f,  // Convert to radians
        camera.getFOV() * settings_.height / settings_.width * M_PI / 180.0f,
        settings_.width,
        settings_.height
    );
    
    RenderParams render_params = createRenderParams(
        settings_.width,
        settings_.height,
        gaussians.size()
    );
    
    // Allocate buffers for 2D Gaussians
    CudaMemory<GaussianData2D> d_gaussians_2d_data(gaussians.size());
    CudaMemory<int> d_visible_count(1);
    
    // Launch projection kernel
    launchProjectionKernel(
        d_gaussians_3d_data.getDevicePtr(),
        d_gaussians_2d_data.getDevicePtr(),
        d_visible_count.getDevicePtr(),
        camera_params,
        render_params,
        gaussians.size(),
        stream
    );
    
    // Get visible count
    int h_visible_count;
    d_visible_count.copyToHost(&h_visible_count, 1);
    stats_.visible_gaussians = h_visible_count;
    
    timer_->stop();
    stats_.projection_time_ms = timer_->getElapsedMs();
    
    // ========================================================================
    // Step 3: Tiling Kernel
    // ========================================================================
    timer_->start();
    
    int total_tiles = render_params.total_tiles;
    
    // Allocate tile buffers
    CudaMemory<int> d_tile_lists(total_tiles * MAX_GAUSSIANS_PER_TILE);
    CudaMemory<int> d_tile_counts(total_tiles);
    CudaMemory<float> d_tile_depths(total_tiles * MAX_GAUSSIANS_PER_TILE);
    CudaMemory<int> d_tile_offsets(total_tiles);
    
    // Launch tiling kernel
    launchTilingKernel(
        d_gaussians_2d_data.getDevicePtr(),
        d_tile_lists.getDevicePtr(),
        d_tile_counts.getDevicePtr(),
        d_tile_depths.getDevicePtr(),
        h_visible_count,
        render_params,
        stream
    );
    
    // Compact tile lists
    CudaMemory<int> d_tile_lists_compact(h_visible_count * 4);  // Worst case
    CudaMemory<float> d_tile_depths_compact(h_visible_count * 4);
    
    launchCompactionKernel(
        d_tile_lists.getDevicePtr(),
        d_tile_depths.getDevicePtr(),
        d_tile_lists_compact.getDevicePtr(),
        d_tile_depths_compact.getDevicePtr(),
        d_tile_offsets.getDevicePtr(),
        d_tile_counts.getDevicePtr(),
        total_tiles,
        stream
    );
    
    timer_->stop();
    float tiling_time = timer_->getElapsedMs();
    
    // ========================================================================
    // Step 4: Sorting Kernel
    // ========================================================================
    timer_->start();
    
    launchSortingKernel(
        d_tile_lists_compact.getDevicePtr(),
        d_tile_depths_compact.getDevicePtr(),
        d_tile_counts.getDevicePtr(),
        d_tile_offsets.getDevicePtr(),
        total_tiles,
        stream
    );
    
    timer_->stop();
    stats_.sorting_time_ms = timer_->getElapsedMs() + tiling_time;
    
    // ========================================================================
    // Step 5: Rasterization Kernel
    // ========================================================================
    timer_->start();
    
    // Allocate output image
    CudaMemory<float4> d_output_image(settings_.width * settings_.height);
    
    // Clear image
    launchClearImageKernel(
        d_output_image.getDevicePtr(),
        settings_.width,
        settings_.height,
        make_float4(0.0f, 0.0f, 0.0f, 0.0f),
        stream
    );
    
    // Launch rasterization kernel
    launchRasterizationKernel(
        d_gaussians_2d_data.getDevicePtr(),
        d_tile_lists_compact.getDevicePtr(),
        d_tile_counts.getDevicePtr(),
        d_tile_offsets.getDevicePtr(),
        d_output_image.getDevicePtr(),
        render_params,
        stream,
        false  // Use tile-based version
    );
    
    timer_->stop();
    stats_.rasterization_time_ms = timer_->getElapsedMs();
    
    // ========================================================================
    // Step 6: Download result to CPU
    // ========================================================================
    timer_->start();
    
    // Synchronize to ensure kernels are complete
    cudaStreamSynchronize(stream);
    
    // Resize output buffer
    size_t output_size = settings_.width * settings_.height * 4;
    output_buffer.resize(output_size);
    
    // Copy from device (float4) to host (float array)
    d_output_image.copyToHost(reinterpret_cast<float4*>(output_buffer.data()), 
                              settings_.width * settings_.height);
    
    timer_->stop();
    stats_.download_time_ms = timer_->getElapsedMs();
    
    // Update total time
    stats_.total_time_ms = stats_.upload_time_ms + stats_.projection_time_ms + 
                           stats_.sorting_time_ms + stats_.rasterization_time_ms + 
                           stats_.download_time_ms;
    
    // Update memory usage
    stats_.memory_used_bytes = d_gaussians_3d_data.allocatedBytes() + 
                               d_gaussians_2d_data.allocatedBytes() +
                               d_output_image.allocatedBytes() +
                               d_tile_lists.allocatedBytes() +
                               d_tile_counts.allocatedBytes();
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
        try {
            auto wrapper = std::make_unique<CudaRasterizerWrapper>();
            std::cout << "Using CUDA-accelerated rasterizer" << std::endl;
            return wrapper;
        } catch (const std::exception& e) {
            std::cerr << "Failed to create CUDA rasterizer: " << e.what() << std::endl;
            std::cerr << "Falling back to CPU rasterizer" << std::endl;
        }
    }
    return std::make_unique<CPURasterizer>();
}

} // namespace CUDA
} // namespace SplatRender