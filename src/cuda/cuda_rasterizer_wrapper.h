#pragma once

#include "../renderer/cpu_rasterizer.h"
#include "cuda_rasterizer.h"
#include <memory>

namespace SplatRender {
namespace CUDA {

// Wrapper class to adapt CudaRasterizer to CPURasterizer interface
class CudaRasterizerWrapper : public CPURasterizer {
public:
    CudaRasterizerWrapper();
    ~CudaRasterizerWrapper();
    
    // Match CPURasterizer interface
    void initialize(const RenderSettings& settings);
    
    void render(const std::vector<Gaussian3D>& gaussians,
                const Camera& camera,
                std::vector<float>& output_buffer);
    
    void setSettings(const RenderSettings& settings);
    
private:
    std::unique_ptr<CudaRasterizer> cuda_rasterizer_;
    bool use_cuda_;
    
    void updateStats();
};

} // namespace CUDA
} // namespace SplatRender