#pragma once

#include "../cuda_constants.h"
#include <cuda_runtime.h>

namespace SplatRender {
namespace CUDA {

// ============================================================================
// Kernel Launch Functions
// ============================================================================

// Projection kernel
void launchProjectionKernel(
    const GaussianData3D* d_gaussians_3d,
    GaussianData2D* d_gaussians_2d,
    int* d_visible_count,
    const CameraParams& camera,
    const RenderParams& render,
    int num_gaussians,
    cudaStream_t stream = 0
);

// Tiling kernel
void launchTilingKernel(
    const GaussianData2D* d_gaussians_2d,
    int* d_tile_lists,
    int* d_tile_counts,
    float* d_tile_depths,
    int num_visible,
    const RenderParams& render,
    cudaStream_t stream = 0
);

// Compaction kernel for tile lists
void launchCompactionKernel(
    const int* d_tile_lists_in,
    const float* d_tile_depths_in,
    int* d_tile_lists_out,
    float* d_tile_depths_out,
    int* d_tile_offsets,
    const int* d_tile_counts,
    int total_tiles,
    cudaStream_t stream = 0
);

// Sorting kernel
void launchSortingKernel(
    int* d_tile_lists,
    float* d_tile_depths,
    const int* d_tile_counts,
    const int* d_tile_offsets,
    int total_tiles,
    cudaStream_t stream = 0
);

// Rasterization kernel
void launchRasterizationKernel(
    const GaussianData2D* d_gaussians_2d,
    const int* d_tile_lists,
    const int* d_tile_counts,
    const int* d_tile_offsets,
    float4* d_output_image,
    const RenderParams& render,
    cudaStream_t stream = 0,
    bool use_fast_version = false
);

// Clear image kernel
void launchClearImageKernel(
    float4* d_output_image,
    int width,
    int height,
    float4 clear_color = make_float4(0.0f, 0.0f, 0.0f, 0.0f),
    cudaStream_t stream = 0
);

// ============================================================================
// Helper Functions
// ============================================================================

// Convert camera matrices to CameraParams structure
inline CameraParams createCameraParams(
    const float* view_matrix,
    const float* proj_matrix,
    const float* camera_pos,
    float fov_x,
    float fov_y,
    int width,
    int height
) {
    CameraParams params;
    
    // Copy matrices
    memcpy(&params.view_matrix, view_matrix, sizeof(float) * 16);
    memcpy(&params.proj_matrix, proj_matrix, sizeof(float) * 16);
    
    // Compute view-projection matrix
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            params.view_proj_matrix.m[i][j] = 0.0f;
            for (int k = 0; k < 4; k++) {
                params.view_proj_matrix.m[i][j] += 
                    params.proj_matrix.m[i][k] * params.view_matrix.m[k][j];
            }
        }
    }
    
    // Set camera position
    params.camera_pos = make_float3(camera_pos[0], camera_pos[1], camera_pos[2]);
    
    // Compute FOV tangents
    params.tan_fovx = tanf(fov_x * 0.5f);
    params.tan_fovy = tanf(fov_y * 0.5f);
    
    // Compute focal lengths
    params.focal_x = width / (2.0f * params.tan_fovx);
    params.focal_y = height / (2.0f * params.tan_fovy);
    
    // Principal point (center of image)
    params.cx = width * 0.5f;
    params.cy = height * 0.5f;
    
    return params;
}

// Create RenderParams structure
inline RenderParams createRenderParams(
    int width,
    int height,
    int num_gaussians,
    float near_plane = 0.1f,
    float far_plane = 100.0f
) {
    RenderParams params;
    
    params.screen_width = width;
    params.screen_height = height;
    params.tiles_x = (width + TILE_SIZE - 1) / TILE_SIZE;
    params.tiles_y = (height + TILE_SIZE - 1) / TILE_SIZE;
    params.total_tiles = params.tiles_x * params.tiles_y;
    params.num_gaussians = num_gaussians;
    params.near_plane = near_plane;
    params.far_plane = far_plane;
    
    return params;
}

} // namespace CUDA
} // namespace SplatRender