#include "../cuda_constants_opt.h"
#include "../cuda_utils.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace SplatRender {
namespace CUDA {

// ============================================================================
// Optimized Projection Kernel using Constant Memory
// ============================================================================

__global__ void projectGaussiansKernelOpt(
    const GaussianData3D* __restrict__ gaussians_3d,
    GaussianData2D* __restrict__ gaussians_2d,
    int* __restrict__ visible_count,
    const int num_gaussians
) {
    // Get global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_gaussians) return;
    
    // Load 3D Gaussian
    GaussianData3D g3d = gaussians_3d[idx];
    
    // Early culling: check opacity
    if (opacityCull(g3d.opacity)) return;
    
    // ========================================================================
    // Step 1: Fast frustum culling using constant memory
    // ========================================================================
    
    if (frustumCullConstMem(g3d.position)) return;
    
    // ========================================================================
    // Step 2: Transform using constant memory matrices (single pass)
    // ========================================================================
    
    // Use precomputed view-projection matrix from constant memory
    float3 pos_clip = transformPointConstMem(g3d.position);
    
    // Get view space position for depth
    float3 pos_view = transformToViewConstMem(g3d.position);
    float depth = -pos_view.z;  // Positive depth
    
    // ========================================================================
    // Step 3: Convert to screen space
    // ========================================================================
    
    float2 screen_pos;
    screen_pos.x = (pos_clip.x + 1.0f) * 0.5f * c_render_params.screen_width;
    screen_pos.y = (pos_clip.y + 1.0f) * 0.5f * c_render_params.screen_height;
    
    // ========================================================================
    // Step 4: Compute 2D covariance matrix (optimized)
    // ========================================================================
    
    // Compute 3D covariance
    float cov3d[6];
    compute3DCovariance(g3d.scale, g3d.rotation, cov3d);
    
    // Use constant memory camera parameters for Jacobian
    float J[4];  // 2x2 Jacobian matrix
    J[0] = c_camera_params.focal_x / depth;
    J[1] = 0.0f;
    J[2] = 0.0f;
    J[3] = c_camera_params.focal_y / depth;
    
    // Transform covariance to 2D (simplified for upper 2x2 of 3x3)
    // Extract view-space covariance relevant components
    float cov_view[3];  // Only need upper triangle of 2x2
    
    // Apply view rotation to covariance (simplified)
    // This is an approximation - full implementation would transform entire 3x3
    cov_view[0] = cov3d[0];  // xx
    cov_view[1] = cov3d[1];  // xy
    cov_view[2] = cov3d[3];  // yy
    
    // Apply Jacobian: Σ_2D = J * Σ_view * J^T
    float cov_2d[3];
    cov_2d[0] = J[0] * J[0] * cov_view[0];  // a = J[0,0]^2 * Σ[0,0]
    cov_2d[1] = J[0] * J[3] * cov_view[1];  // b = J[0,0] * J[1,1] * Σ[0,1]
    cov_2d[2] = J[3] * J[3] * cov_view[2];  // c = J[1,1]^2 * Σ[1,1]
    
    // Add minimum covariance for numerical stability
    cov_2d[0] += 0.3f;
    cov_2d[2] += 0.3f;
    
    // ========================================================================
    // Step 5: Compute bounding radius
    // ========================================================================
    
    // Compute eigenvalues for bounding radius (3-sigma extent)
    float tr = cov_2d[0] + cov_2d[2];
    float det = cov_2d[0] * cov_2d[2] - cov_2d[1] * cov_2d[1];
    
    if (det <= 0.0f) return;  // Degenerate Gaussian
    
    float discriminant = tr * tr - 4.0f * det;
    if (discriminant < 0.0f) return;
    
    float sqrt_disc = sqrtf(discriminant);
    float lambda1 = 0.5f * (tr + sqrt_disc);
    float lambda2 = 0.5f * (tr - sqrt_disc);
    
    float radius = CULLING_THRESHOLD * sqrtf(fmaxf(lambda1, lambda2));
    
    // Size culling
    if (sizeCull(radius, depth)) return;
    
    // ========================================================================
    // Step 6: Evaluate spherical harmonics for view-dependent color
    // ========================================================================
    
    // Compute view direction from camera to Gaussian
    float3 view_dir = normalize(g3d.position - c_camera_params.camera_pos);
    float3 color = evaluateSH(g3d.sh_coeffs, view_dir);
    
    // ========================================================================
    // Step 7: Write output (with atomic counter)
    // ========================================================================
    
    // Get output index with atomic increment
    int output_idx = atomicAdd(visible_count, 1);
    
    if (output_idx < c_render_params.num_gaussians) {
        GaussianData2D& g2d = gaussians_2d[output_idx];
        
        g2d.center = screen_pos;
        g2d.depth = depth;
        g2d.radius = radius;
        g2d.cov_a = cov_2d[0];
        g2d.cov_b = cov_2d[1];
        g2d.cov_c = cov_2d[2];
        g2d.alpha = g3d.opacity;
        g2d.color = color;
        g2d.pad = 0.0f;
    }
}

// ============================================================================
// Optimized Projection Kernel with Shared Memory for Batch Processing
// ============================================================================

__global__ void projectGaussiansKernelOptShared(
    const GaussianData3D* __restrict__ gaussians_3d,
    GaussianData2D* __restrict__ gaussians_2d,
    int* __restrict__ visible_count,
    const int num_gaussians
) {
    // Shared memory for batch processing
    // Note: GaussianData3D is 240 bytes, so we need to limit block size
    const int SHARED_BLOCK_SIZE = 64;  // Reduced from PROJECTION_BLOCK_SIZE
    __shared__ GaussianData3D shared_gaussians[SHARED_BLOCK_SIZE];
    __shared__ int shared_visible[SHARED_BLOCK_SIZE];
    __shared__ int block_visible_count;
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Initialize shared memory
    if (tid == 0) {
        block_visible_count = 0;
    }
    __syncthreads();
    
    // Load batch to shared memory (coalesced)
    if (idx < num_gaussians) {
        shared_gaussians[tid] = gaussians_3d[idx];
    }
    __syncthreads();
    
    // Process Gaussian
    bool is_visible = false;
    GaussianData2D g2d_local;
    
    if (idx < num_gaussians) {
        GaussianData3D& g3d = shared_gaussians[tid];
        
        // Early culling
        if (!opacityCull(g3d.opacity) && !frustumCullConstMem(g3d.position)) {
            // Transform using constant memory
            float3 pos_clip = transformPointConstMem(g3d.position);
            float3 pos_view = transformToViewConstMem(g3d.position);
            float depth = -pos_view.z;
            
            // Screen space conversion
            g2d_local.center.x = (pos_clip.x + 1.0f) * 0.5f * c_render_params.screen_width;
            g2d_local.center.y = (pos_clip.y + 1.0f) * 0.5f * c_render_params.screen_height;
            g2d_local.depth = depth;
            
            // Compute 2D covariance (simplified)
            float cov3d[6];
            compute3DCovariance(g3d.scale, g3d.rotation, cov3d);
            
            float J_scale_x = c_camera_params.focal_x / depth;
            float J_scale_y = c_camera_params.focal_y / depth;
            
            g2d_local.cov_a = J_scale_x * J_scale_x * cov3d[0] + 0.3f;
            g2d_local.cov_b = J_scale_x * J_scale_y * cov3d[1];
            g2d_local.cov_c = J_scale_y * J_scale_y * cov3d[3] + 0.3f;
            
            // Compute radius
            float tr = g2d_local.cov_a + g2d_local.cov_c;
            float det = g2d_local.cov_a * g2d_local.cov_c - g2d_local.cov_b * g2d_local.cov_b;
            
            if (det > 0.0f) {
                float discriminant = tr * tr - 4.0f * det;
                if (discriminant >= 0.0f) {
                    float sqrt_disc = sqrtf(discriminant);
                    float lambda_max = 0.5f * (tr + sqrt_disc);
                    g2d_local.radius = CULLING_THRESHOLD * sqrtf(lambda_max);
                    
                    if (!sizeCull(g2d_local.radius, depth)) {
                        // Evaluate SH
                        float3 view_dir = normalize(g3d.position - c_camera_params.camera_pos);
                        g2d_local.color = evaluateSH(g3d.sh_coeffs, view_dir);
                        g2d_local.alpha = g3d.opacity;
                        g2d_local.pad = 0.0f;
                        
                        is_visible = true;
                    }
                }
            }
        }
    }
    
    // Mark visible Gaussians in shared memory
    shared_visible[tid] = is_visible ? 1 : 0;
    __syncthreads();
    
    // Compact visible Gaussians within block using warp-level primitives
    if (is_visible) {
        int local_idx = atomicAdd(&block_visible_count, 1);
        shared_visible[tid] = local_idx;
    }
    __syncthreads();
    
    // Block leader reserves space in global output
    __shared__ int block_output_offset;
    if (tid == 0 && block_visible_count > 0) {
        block_output_offset = atomicAdd(visible_count, block_visible_count);
    }
    __syncthreads();
    
    // Write compacted output
    if (is_visible && block_output_offset + shared_visible[tid] < c_render_params.num_gaussians) {
        gaussians_2d[block_output_offset + shared_visible[tid]] = g2d_local;
    }
}

// ============================================================================
// Host-side function declarations (to be called from cuda_constants_opt.cu)
// ============================================================================

extern void updateConstantCameraParams(const CameraParams& params);
extern void updateConstantRenderParams(const RenderParams& params);
extern void updateConstantMatrices(const float* view, const float* proj);

void launchProjectionKernelOpt(
    const GaussianData3D* d_gaussians_3d,
    GaussianData2D* d_gaussians_2d,
    int* d_visible_count,
    const CameraParams& camera_params,
    const RenderParams& render_params,
    int num_gaussians,
    cudaStream_t stream,
    bool use_shared_memory
) {
    // Update constant memory
    updateConstantCameraParams(camera_params);
    updateConstantRenderParams(render_params);
    
    // Compute view-projection matrix and upload
    float viewproj[16];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 4; k++) {
                sum += camera_params.proj_matrix.m[i][k] * camera_params.view_matrix.m[k][j];
            }
            viewproj[i * 4 + j] = sum;
        }
    }
    
    // Update all matrices
    updateConstantMatrices(
        reinterpret_cast<const float*>(&camera_params.view_matrix),
        reinterpret_cast<const float*>(&camera_params.proj_matrix)
    );
    
    // Reset visible count
    cudaMemsetAsync(d_visible_count, 0, sizeof(int), stream);
    
    // Launch configuration
    int block_size = use_shared_memory ? 64 : PROJECTION_BLOCK_SIZE;  // Reduced for shared memory version
    int grid_size = (num_gaussians + block_size - 1) / block_size;
    
    // Launch appropriate kernel
    if (use_shared_memory) {
        projectGaussiansKernelOptShared<<<grid_size, block_size, 0, stream>>>(
            d_gaussians_3d, d_gaussians_2d, d_visible_count, num_gaussians
        );
    } else {
        projectGaussiansKernelOpt<<<grid_size, block_size, 0, stream>>>(
            d_gaussians_3d, d_gaussians_2d, d_visible_count, num_gaussians
        );
    }
    
    CUDA_CHECK_KERNEL();
}

}  // namespace CUDA
}  // namespace SplatRender