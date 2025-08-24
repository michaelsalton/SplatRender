#include "cuda_constants_opt.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>

namespace SplatRender {
namespace CUDA {

// ============================================================================
// Constant Memory Definitions
// ============================================================================

// Define the constant memory variables (64KB total available)
__constant__ float c_view_matrix[16];
__constant__ float c_proj_matrix[16];
__constant__ float c_viewproj_matrix[16];
__constant__ CameraParams c_camera_params;
__constant__ RenderParams c_render_params;

// ============================================================================
// Host Functions to Update Constant Memory
// ============================================================================

void updateConstantViewMatrix(const float* view_matrix) {
    CUDA_CHECK(cudaMemcpyToSymbol(c_view_matrix, view_matrix, 16 * sizeof(float)));
}

void updateConstantProjMatrix(const float* proj_matrix) {
    CUDA_CHECK(cudaMemcpyToSymbol(c_proj_matrix, proj_matrix, 16 * sizeof(float)));
}

void updateConstantViewProjMatrix(const float* viewproj_matrix) {
    CUDA_CHECK(cudaMemcpyToSymbol(c_viewproj_matrix, viewproj_matrix, 16 * sizeof(float)));
}

void updateConstantCameraParams(const CameraParams& params) {
    CUDA_CHECK(cudaMemcpyToSymbol(c_camera_params, &params, sizeof(CameraParams)));
}

void updateConstantRenderParams(const RenderParams& params) {
    CUDA_CHECK(cudaMemcpyToSymbol(c_render_params, &params, sizeof(RenderParams)));
}

// Convenience function to update all matrices at once
void updateConstantMatrices(const float* view, const float* proj) {
    // Update individual matrices
    updateConstantViewMatrix(view);
    updateConstantProjMatrix(proj);
    
    // Compute and update view-projection matrix
    float viewproj[16];
    
    // Matrix multiplication: viewproj = proj * view
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 4; k++) {
                sum += proj[i * 4 + k] * view[k * 4 + j];
            }
            viewproj[i * 4 + j] = sum;
        }
    }
    
    updateConstantViewProjMatrix(viewproj);
}

// ============================================================================
// Memory Usage Reporting
// ============================================================================

size_t getConstantMemoryUsage() {
    // Calculate total constant memory usage
    size_t total = 0;
    total += 16 * sizeof(float);     // view_matrix
    total += 16 * sizeof(float);     // proj_matrix
    total += 16 * sizeof(float);     // viewproj_matrix
    total += sizeof(CameraParams);   // camera_params
    total += sizeof(RenderParams);   // render_params
    
    return total;
}

void printConstantMemoryUsage() {
    size_t used = getConstantMemoryUsage();
    size_t available = 65536;  // 64KB constant memory
    
    printf("Constant Memory Usage:\n");
    printf("  Used: %zu bytes (%.1f%%)\n", used, (float)used / available * 100.0f);
    printf("  Available: %zu bytes\n", available - used);
    printf("  Details:\n");
    printf("    View Matrix:     %zu bytes\n", 16 * sizeof(float));
    printf("    Proj Matrix:     %zu bytes\n", 16 * sizeof(float));
    printf("    ViewProj Matrix: %zu bytes\n", 16 * sizeof(float));
    printf("    Camera Params:   %zu bytes\n", sizeof(CameraParams));
    printf("    Render Params:   %zu bytes\n", sizeof(RenderParams));
}

}  // namespace CUDA
}  // namespace SplatRender