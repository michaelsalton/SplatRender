#include "cuda_rasterizer_wrapper.h"
#include <iostream>

namespace SplatRender {
namespace CUDA {

CudaRasterizerWrapper::CudaRasterizerWrapper() 
    : cuda_rasterizer_(std::make_unique<CudaRasterizer>()), use_cuda_(false) {
}

CudaRasterizerWrapper::~CudaRasterizerWrapper() = default;

void CudaRasterizerWrapper::initialize(const RenderSettings& settings) {
    // Initialize CPU rasterizer base (for fallback if needed)
    CPURasterizer::initialize(settings);
    
    // Try to initialize CUDA rasterizer
    if (cuda_rasterizer_->initialize(settings)) {
        use_cuda_ = true;
        std::cout << "CUDA Rasterizer initialized for GPU acceleration" << std::endl;
    } else {
        use_cuda_ = false;
        std::cerr << "Failed to initialize CUDA rasterizer, falling back to CPU" << std::endl;
    }
}

void CudaRasterizerWrapper::render(const std::vector<Gaussian3D>& gaussians,
                                   const Camera& camera,
                                   std::vector<float>& output_buffer) {
    if (use_cuda_ && cuda_rasterizer_->isAvailable()) {
        // Use CUDA rasterizer
        cuda_rasterizer_->render(gaussians, camera, output_buffer);
        updateStats();
    } else {
        // Fall back to CPU rasterizer
        CPURasterizer::render(gaussians, camera, output_buffer);
    }
}

void CudaRasterizerWrapper::setSettings(const RenderSettings& settings) {
    CPURasterizer::setSettings(settings);
    if (cuda_rasterizer_) {
        cuda_rasterizer_->setSettings(settings);
    }
}

void CudaRasterizerWrapper::updateStats() {
    // Stats updating would require modifying the base class to make stats_ protected
    // For now, stats are only accessible through the CUDA rasterizer directly
}

} // namespace CUDA
} // namespace SplatRender