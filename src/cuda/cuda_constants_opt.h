#pragma once

#include "cuda_constants.h"

namespace SplatRender {
namespace CUDA {

// ============================================================================
// Constant Memory Declarations
// ============================================================================

// Constant memory for frequently accessed read-only data
// These are declared extern here and defined in cuda_constants_opt.cu

// Camera matrices in constant memory (64KB total available)
extern __constant__ float c_view_matrix[16];      // 4x4 view matrix
extern __constant__ float c_proj_matrix[16];      // 4x4 projection matrix  
extern __constant__ float c_viewproj_matrix[16];  // Precomputed view * proj
extern __constant__ CameraParams c_camera_params; // Full camera parameters
extern __constant__ RenderParams c_render_params; // Render parameters

// ============================================================================
// Optimized Device Functions
// ============================================================================

// Transform point using constant memory matrices
__device__ __forceinline__ float3 transformPointConstMem(const float3& p) {
    float4 p4 = make_float4(p.x, p.y, p.z, 1.0f);
    float4 result;
    
    // Use constant memory view-projection matrix
    result.x = c_viewproj_matrix[0] * p4.x + c_viewproj_matrix[1] * p4.y + 
               c_viewproj_matrix[2] * p4.z + c_viewproj_matrix[3] * p4.w;
    result.y = c_viewproj_matrix[4] * p4.x + c_viewproj_matrix[5] * p4.y + 
               c_viewproj_matrix[6] * p4.z + c_viewproj_matrix[7] * p4.w;
    result.z = c_viewproj_matrix[8] * p4.x + c_viewproj_matrix[9] * p4.y + 
               c_viewproj_matrix[10] * p4.z + c_viewproj_matrix[11] * p4.w;
    result.w = c_viewproj_matrix[12] * p4.x + c_viewproj_matrix[13] * p4.y + 
               c_viewproj_matrix[14] * p4.z + c_viewproj_matrix[15] * p4.w;
    
    if (result.w != 0.0f) {
        float inv_w = 1.0f / result.w;
        return make_float3(result.x * inv_w, result.y * inv_w, result.z * inv_w);
    }
    return make_float3(result.x, result.y, result.z);
}

// Transform to view space using constant memory
__device__ __forceinline__ float3 transformToViewConstMem(const float3& p) {
    float4 p4 = make_float4(p.x, p.y, p.z, 1.0f);
    float3 result;
    
    result.x = c_view_matrix[0] * p4.x + c_view_matrix[1] * p4.y + 
               c_view_matrix[2] * p4.z + c_view_matrix[3] * p4.w;
    result.y = c_view_matrix[4] * p4.x + c_view_matrix[5] * p4.y + 
               c_view_matrix[6] * p4.z + c_view_matrix[7] * p4.w;
    result.z = c_view_matrix[8] * p4.x + c_view_matrix[9] * p4.y + 
               c_view_matrix[10] * p4.z + c_view_matrix[11] * p4.w;
    
    return result;
}

// Project point using constant memory projection matrix
__device__ __forceinline__ float4 projectPointConstMem(const float3& view_pos) {
    float4 p4 = make_float4(view_pos.x, view_pos.y, view_pos.z, 1.0f);
    float4 result;
    
    result.x = c_proj_matrix[0] * p4.x + c_proj_matrix[1] * p4.y + 
               c_proj_matrix[2] * p4.z + c_proj_matrix[3] * p4.w;
    result.y = c_proj_matrix[4] * p4.x + c_proj_matrix[5] * p4.y + 
               c_proj_matrix[6] * p4.z + c_proj_matrix[7] * p4.w;
    result.z = c_proj_matrix[8] * p4.x + c_proj_matrix[9] * p4.y + 
               c_proj_matrix[10] * p4.z + c_proj_matrix[11] * p4.w;
    result.w = c_proj_matrix[12] * p4.x + c_proj_matrix[13] * p4.y + 
               c_proj_matrix[14] * p4.z + c_proj_matrix[15] * p4.w;
    
    return result;
}

// ============================================================================
// Shared Memory Optimization Helpers
// ============================================================================

// Structure for shared memory Gaussian data (optimized layout)
struct SharedGaussianData {
    float2 center;
    float depth;
    float radius;
    float cov_a, cov_b, cov_c;
    float alpha;
    float3 color;
};

// Load Gaussians to shared memory with coalesced access
template<int BLOCK_SIZE>
__device__ void loadGaussiansToShared(
    const GaussianData2D* __restrict__ global_gaussians,
    SharedGaussianData* __restrict__ shared_gaussians,
    int start_idx,
    int count,
    int tid
) {
    // Each thread loads one or more Gaussians
    for (int i = tid; i < count; i += BLOCK_SIZE) {
        if (start_idx + i < c_render_params.num_gaussians) {
            const GaussianData2D& g = global_gaussians[start_idx + i];
            SharedGaussianData& s = shared_gaussians[i];
            
            s.center = g.center;
            s.depth = g.depth;
            s.radius = g.radius;
            s.cov_a = g.cov_a;
            s.cov_b = g.cov_b;
            s.cov_c = g.cov_c;
            s.alpha = g.alpha;
            s.color = g.color;
        }
    }
    __syncthreads();
}

// ============================================================================
// Warp-Level Optimizations
// ============================================================================

// Warp-level reduction for finding minimum depth
__device__ __forceinline__ float warpReduceMin(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp-level reduction for finding maximum depth
__device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp-level reduction for summing values
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Check if any thread in warp meets condition
__device__ __forceinline__ bool warpAny(bool predicate) {
    return __ballot_sync(0xffffffff, predicate) != 0;
}

// Check if all threads in warp meet condition
__device__ __forceinline__ bool warpAll(bool predicate) {
    return __ballot_sync(0xffffffff, predicate) == 0xffffffff;
}

// ============================================================================
// Memory Access Optimizations
// ============================================================================

// Vectorized load for float4 data
__device__ __forceinline__ float4 loadFloat4(const float4* addr) {
    return __ldg(addr);  // Use read-only cache
}

// Vectorized load for float2 data
__device__ __forceinline__ float2 loadFloat2(const float2* addr) {
    return __ldg(addr);  // Use read-only cache
}

// ============================================================================
// Early Culling Optimizations
// ============================================================================

// Fast frustum culling check
__device__ __forceinline__ bool frustumCullConstMem(const float3& pos) {
    // Transform to view space
    float3 view_pos = transformToViewConstMem(pos);
    
    // Check if behind camera
    if (view_pos.z > -c_camera_params.focal_y * 0.1f) return true;
    
    // Check against frustum planes using tan(fov)
    float abs_x = fabsf(view_pos.x);
    float abs_y = fabsf(view_pos.y);
    float neg_z = -view_pos.z;
    
    if (abs_x > c_camera_params.tan_fovx * neg_z) return true;
    if (abs_y > c_camera_params.tan_fovy * neg_z) return true;
    
    return false;
}

// Fast opacity culling
__device__ __forceinline__ bool opacityCull(float opacity) {
    return opacity < MIN_OPACITY;
}

// Fast size culling
__device__ __forceinline__ bool sizeCull(float radius, float depth) {
    float pixel_size = radius * 2.0f / depth;
    return pixel_size < MIN_RADIUS;
}

// ============================================================================
// Tile Assignment Optimizations
// ============================================================================

// Compute tile ID from pixel coordinates
__device__ __forceinline__ int getTileId(int x, int y) {
    int tile_x = x / TILE_SIZE;
    int tile_y = y / TILE_SIZE;
    return tile_y * c_render_params.tiles_x + tile_x;
}

// Check if Gaussian overlaps tile (conservative test)
__device__ __forceinline__ bool overlapsTile(
    const float2& center,
    float radius,
    int tile_x,
    int tile_y
) {
    float tile_min_x = tile_x * TILE_SIZE;
    float tile_max_x = (tile_x + 1) * TILE_SIZE;
    float tile_min_y = tile_y * TILE_SIZE;
    float tile_max_y = (tile_y + 1) * TILE_SIZE;
    
    return (center.x + radius >= tile_min_x) &&
           (center.x - radius <= tile_max_x) &&
           (center.y + radius >= tile_min_y) &&
           (center.y - radius <= tile_max_y);
}

}  // namespace CUDA
}  // namespace SplatRender