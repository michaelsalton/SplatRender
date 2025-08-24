#include "../cuda_constants.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace SplatRender {
namespace CUDA {

// ============================================================================
// Projection Kernel - Transform 3D Gaussians to 2D screen space
// ============================================================================

__global__ void projectGaussiansKernel(
    const GaussianData3D* __restrict__ gaussians_3d,
    GaussianData2D* __restrict__ gaussians_2d,
    int* __restrict__ visible_count,
    const CameraParams camera,
    const RenderParams render,
    const int num_gaussians
) {
    // Get global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_gaussians) return;
    
    // Load 3D Gaussian
    GaussianData3D g3d = gaussians_3d[idx];
    
    // Early culling: check opacity
    if (g3d.opacity < MIN_OPACITY) return;
    
    // ========================================================================
    // Step 1: Transform to view space
    // ========================================================================
    
    float4 pos_world = make_float4(g3d.position.x, g3d.position.y, g3d.position.z, 1.0f);
    
    // Apply view matrix
    float4 pos_view;
    pos_view.x = camera.view_matrix.m[0][0] * pos_world.x + camera.view_matrix.m[0][1] * pos_world.y + 
                 camera.view_matrix.m[0][2] * pos_world.z + camera.view_matrix.m[0][3];
    pos_view.y = camera.view_matrix.m[1][0] * pos_world.x + camera.view_matrix.m[1][1] * pos_world.y + 
                 camera.view_matrix.m[1][2] * pos_world.z + camera.view_matrix.m[1][3];
    pos_view.z = camera.view_matrix.m[2][0] * pos_world.x + camera.view_matrix.m[2][1] * pos_world.y + 
                 camera.view_matrix.m[2][2] * pos_world.z + camera.view_matrix.m[2][3];
    pos_view.w = 1.0f;
    
    // Frustum culling: check if behind camera (in view space, z is negative for visible objects)
    if (pos_view.z > -render.near_plane) return;
    
    // ========================================================================
    // Step 2: Project to clip space
    // ========================================================================
    
    float4 pos_clip;
    pos_clip.x = camera.proj_matrix.m[0][0] * pos_view.x + camera.proj_matrix.m[0][1] * pos_view.y + 
                 camera.proj_matrix.m[0][2] * pos_view.z + camera.proj_matrix.m[0][3];
    pos_clip.y = camera.proj_matrix.m[1][0] * pos_view.x + camera.proj_matrix.m[1][1] * pos_view.y + 
                 camera.proj_matrix.m[1][2] * pos_view.z + camera.proj_matrix.m[1][3];
    pos_clip.z = camera.proj_matrix.m[2][0] * pos_view.x + camera.proj_matrix.m[2][1] * pos_view.y + 
                 camera.proj_matrix.m[2][2] * pos_view.z + camera.proj_matrix.m[2][3];
    pos_clip.w = camera.proj_matrix.m[3][0] * pos_view.x + camera.proj_matrix.m[3][1] * pos_view.y + 
                 camera.proj_matrix.m[3][2] * pos_view.z + camera.proj_matrix.m[3][3];
    
    // Perspective divide to get NDC coordinates
    if (pos_clip.w <= 0.0f) return;
    float3 pos_ndc = make_float3(pos_clip.x / pos_clip.w, pos_clip.y / pos_clip.w, pos_clip.z / pos_clip.w);
    
    // Frustum culling in NDC space (with margin for Gaussians near edges)
    if (fabsf(pos_ndc.x) > 3.0f || fabsf(pos_ndc.y) > 3.0f) return;
    
    // ========================================================================
    // Step 3: Convert to screen space
    // ========================================================================
    
    float2 screen_pos;
    screen_pos.x = (pos_ndc.x + 1.0f) * 0.5f * render.screen_width;
    screen_pos.y = (pos_ndc.y + 1.0f) * 0.5f * render.screen_height;
    
    // ========================================================================
    // Step 4: Compute 2D covariance matrix
    // ========================================================================
    
    // First compute 3D covariance
    float cov3d[6];
    compute3DCovariance(g3d.scale, g3d.rotation, cov3d);
    
    // Compute Jacobian of projection at this point
    float depth = -pos_view.z;  // Positive depth
    float J[4];  // 2x2 Jacobian matrix
    
    // Simplified Jacobian for perspective projection
    J[0] = camera.focal_x / depth;  // dx/dX
    J[1] = 0.0f;                     // dx/dY
    J[2] = 0.0f;                     // dy/dX
    J[3] = camera.focal_y / depth;  // dy/dY
    
    // Transform 3D covariance to 2D
    // We need to extract the relevant 2x2 submatrix from the 3x3 covariance
    // and transform it: Σ_2D = J * Σ_3D * J^T
    
    // View space transformation of covariance (simplified)
    float T[4];  // 2x2 submatrix of transformed covariance
    T[0] = cov3d[0];  // xx
    T[1] = cov3d[1];  // xy
    T[2] = cov3d[1];  // yx (symmetric)
    T[3] = cov3d[3];  // yy
    
    // Apply Jacobian: Σ_2D = J * T * J^T
    float cov2d[3];  // Store as symmetric matrix (a, b, c)
    cov2d[0] = J[0] * J[0] * T[0] + J[0] * J[1] * T[2] + J[1] * J[0] * T[1] + J[1] * J[1] * T[3];
    cov2d[1] = J[2] * J[0] * T[0] + J[2] * J[1] * T[2] + J[3] * J[0] * T[1] + J[3] * J[1] * T[3];
    cov2d[2] = J[2] * J[2] * T[0] + J[2] * J[3] * T[2] + J[3] * J[2] * T[1] + J[3] * J[3] * T[3];
    
    // ========================================================================
    // Step 5: Compute screen space radius
    // ========================================================================
    
    // Eigenvalues of 2x2 covariance matrix for bounding radius
    float a = cov2d[0];
    float b = cov2d[1];
    float c = cov2d[2];
    
    float trace = a + c;
    float det = a * c - b * b;
    
    if (det <= 0.0f) return;  // Degenerate Gaussian
    
    float discriminant = trace * trace - 4.0f * det;
    if (discriminant < 0.0f) return;
    
    float sqrt_disc = sqrtf(discriminant);
    float lambda1 = 0.5f * (trace + sqrt_disc);
    float lambda2 = 0.5f * (trace - sqrt_disc);
    
    float radius = CULLING_THRESHOLD * sqrtf(fmaxf(lambda1, lambda2));
    
    // Skip if too small
    if (radius < MIN_RADIUS) return;
    
    // ========================================================================
    // Step 6: Evaluate spherical harmonics for view-dependent color
    // ========================================================================
    
    float3 view_dir = normalize(camera.camera_pos - g3d.position);
    float3 color = evaluateSH(g3d.sh_coeffs, view_dir);
    
    // ========================================================================
    // Step 7: Write visible Gaussian to output
    // ========================================================================
    
    // Atomically get output index
    int output_idx = atomicAdd(visible_count, 1);
    
    // Bounds check (should not happen with proper allocation)
    if (output_idx >= render.num_gaussians) return;
    
    // Create and write 2D Gaussian
    GaussianData2D g2d;
    g2d.center = screen_pos;
    g2d.depth = depth;
    g2d.radius = radius;
    g2d.cov_a = cov2d[0];
    g2d.cov_b = cov2d[1];
    g2d.cov_c = cov2d[2];
    g2d.alpha = g3d.opacity;
    g2d.color = color;
    g2d.pad = 0.0f;
    
    gaussians_2d[output_idx] = g2d;
}

// ============================================================================
// Host-side kernel launcher
// ============================================================================

void launchProjectionKernel(
    const GaussianData3D* d_gaussians_3d,
    GaussianData2D* d_gaussians_2d,
    int* d_visible_count,
    const CameraParams& camera,
    const RenderParams& render,
    int num_gaussians,
    cudaStream_t stream
) {
    // Reset visible count
    cudaMemsetAsync(d_visible_count, 0, sizeof(int), stream);
    
    // Calculate grid dimensions
    dim3 blocks((num_gaussians + PROJECTION_BLOCK_SIZE - 1) / PROJECTION_BLOCK_SIZE);
    dim3 threads(PROJECTION_BLOCK_SIZE);
    
    // Launch kernel
    projectGaussiansKernel<<<blocks, threads, 0, stream>>>(
        d_gaussians_3d,
        d_gaussians_2d,
        d_visible_count,
        camera,
        render,
        num_gaussians
    );
}

} // namespace CUDA
} // namespace SplatRender