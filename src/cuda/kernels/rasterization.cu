#include "../cuda_constants.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace SplatRender {
namespace CUDA {

// ============================================================================
// Rasterization Kernel - Render sorted Gaussians to pixels
// ============================================================================

__global__ void rasterizeKernel(
    const GaussianData2D* __restrict__ gaussians_2d,
    const int* __restrict__ tile_lists,
    const int* __restrict__ tile_counts,
    const int* __restrict__ tile_offsets,
    float4* __restrict__ output_image,
    const RenderParams render
) {
    // Get tile coordinates
    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;
    
    if (tile_x >= render.tiles_x || tile_y >= render.tiles_y) return;
    
    // Get pixel coordinates within tile
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;
    
    // Calculate global pixel coordinates
    int pixel_x = tile_x * TILE_SIZE + local_x;
    int pixel_y = tile_y * TILE_SIZE + local_y;
    
    // Check bounds
    if (pixel_x >= render.screen_width || pixel_y >= render.screen_height) return;
    
    // Get tile index
    int tile_idx = tile_y * render.tiles_x + tile_x;
    int gaussian_count = tile_counts[tile_idx];
    
    // Early exit if no Gaussians in this tile
    if (gaussian_count == 0) {
        int pixel_idx = pixel_y * render.screen_width + pixel_x;
        output_image[pixel_idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }
    
    // ========================================================================
    // Step 1: Initialize pixel state
    // ========================================================================
    
    float2 pixel_pos = make_float2(pixel_x + 0.5f, pixel_y + 0.5f);  // Pixel center
    float3 accumulated_color = make_float3(0.0f, 0.0f, 0.0f);
    float accumulated_alpha = 0.0f;
    float transmittance = 1.0f;
    
    // Get tile's Gaussian list offset
    int list_offset = tile_offsets[tile_idx];
    
    // ========================================================================
    // Step 2: Process Gaussians in front-to-back order
    // ========================================================================
    
    // Shared memory for batch loading Gaussians
    __shared__ struct {
        float2 center[BATCH_SIZE];
        float cov_a[BATCH_SIZE];
        float cov_b[BATCH_SIZE];
        float cov_c[BATCH_SIZE];
        float3 color[BATCH_SIZE];
        float alpha[BATCH_SIZE];
    } s_batch;
    
    // Process Gaussians in batches
    for (int batch_start = 0; batch_start < gaussian_count; batch_start += BATCH_SIZE) {
        int batch_end = min(batch_start + BATCH_SIZE, gaussian_count);
        int batch_size = batch_end - batch_start;
        
        // Cooperatively load batch into shared memory
        int tid = threadIdx.y * blockDim.x + threadIdx.x;
        if (tid < batch_size) {
            int gaussian_idx = tile_lists[list_offset + batch_start + tid];
            GaussianData2D g = gaussians_2d[gaussian_idx];
            
            s_batch.center[tid] = g.center;
            s_batch.cov_a[tid] = g.cov_a;
            s_batch.cov_b[tid] = g.cov_b;
            s_batch.cov_c[tid] = g.cov_c;
            s_batch.color[tid] = g.color;
            s_batch.alpha[tid] = g.alpha;
        }
        __syncthreads();
        
        // Process Gaussians in this batch
        for (int i = 0; i < batch_size; i++) {
            // Early termination if transmittance is too low
            if (transmittance < TRANSMITTANCE_THRESHOLD) break;
            
            // Evaluate Gaussian at this pixel
            float gaussian_val = evaluateGaussian2D(
                pixel_pos,
                s_batch.center[i],
                s_batch.cov_a[i],
                s_batch.cov_b[i],
                s_batch.cov_c[i]
            );
            
            // Skip if contribution is negligible
            if (gaussian_val < 0.01f) continue;
            
            // Compute alpha for this Gaussian
            float alpha = s_batch.alpha[i] * gaussian_val;
            alpha = fminf(alpha, 0.99f);  // Clamp to avoid numerical issues
            
            // Front-to-back alpha blending
            float weight = alpha * transmittance;
            accumulated_color.x += weight * s_batch.color[i].x;
            accumulated_color.y += weight * s_batch.color[i].y;
            accumulated_color.z += weight * s_batch.color[i].z;
            accumulated_alpha += weight;
            
            // Update transmittance
            transmittance *= (1.0f - alpha);
        }
        
        __syncthreads();  // Ensure all threads are done with shared memory
    }
    
    // ========================================================================
    // Step 3: Write final pixel color
    // ========================================================================
    
    int pixel_idx = pixel_y * render.screen_width + pixel_x;
    output_image[pixel_idx] = make_float4(
        accumulated_color.x,
        accumulated_color.y,
        accumulated_color.z,
        accumulated_alpha
    );
}

// ============================================================================
// Alternative Fast Rasterization Kernel (simplified, no shared memory)
// ============================================================================

__global__ void rasterizeFastKernel(
    const GaussianData2D* __restrict__ gaussians_2d,
    const int* __restrict__ tile_lists,
    const int* __restrict__ tile_counts,
    const int* __restrict__ tile_offsets,
    float4* __restrict__ output_image,
    const RenderParams render
) {
    // Calculate pixel coordinates
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (pixel_x >= render.screen_width || pixel_y >= render.screen_height) return;
    
    // Determine which tile this pixel belongs to
    int tile_x = pixel_x / TILE_SIZE;
    int tile_y = pixel_y / TILE_SIZE;
    int tile_idx = tile_y * render.tiles_x + tile_x;
    
    // Get Gaussian count for this tile
    int gaussian_count = tile_counts[tile_idx];
    
    // Initialize pixel state
    float2 pixel_pos = make_float2(pixel_x + 0.5f, pixel_y + 0.5f);
    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    float alpha = 0.0f;
    float T = 1.0f;  // Transmittance
    
    if (gaussian_count > 0) {
        int list_offset = tile_offsets[tile_idx];
        
        // Process each Gaussian
        for (int i = 0; i < gaussian_count && T > TRANSMITTANCE_THRESHOLD; i++) {
            int gaussian_idx = tile_lists[list_offset + i];
            GaussianData2D g = gaussians_2d[gaussian_idx];
            
            // Evaluate Gaussian
            float val = evaluateGaussian2D(pixel_pos, g.center, g.cov_a, g.cov_b, g.cov_c);
            if (val < 0.01f) continue;
            
            // Alpha blending
            float a = fminf(g.alpha * val, 0.99f);
            float weight = a * T;
            
            color.x += weight * g.color.x;
            color.y += weight * g.color.y;
            color.z += weight * g.color.z;
            alpha += weight;
            
            T *= (1.0f - a);
        }
    }
    
    // Write output
    int pixel_idx = pixel_y * render.screen_width + pixel_x;
    output_image[pixel_idx] = make_float4(color.x, color.y, color.z, alpha);
}

// ============================================================================
// Host-side kernel launchers
// ============================================================================

void launchRasterizationKernel(
    const GaussianData2D* d_gaussians_2d,
    const int* d_tile_lists,
    const int* d_tile_counts,
    const int* d_tile_offsets,
    float4* d_output_image,
    const RenderParams& render,
    cudaStream_t stream,
    bool use_fast_version
) {
    if (use_fast_version) {
        // Fast version: one thread per pixel
        dim3 blocks((render.screen_width + 15) / 16, (render.screen_height + 15) / 16);
        dim3 threads(16, 16);
        
        rasterizeFastKernel<<<blocks, threads, 0, stream>>>(
            d_gaussians_2d,
            d_tile_lists,
            d_tile_counts,
            d_tile_offsets,
            d_output_image,
            render
        );
    } else {
        // Tile-based version: one block per tile
        dim3 blocks(render.tiles_x, render.tiles_y);
        dim3 threads(TILE_SIZE, TILE_SIZE);
        
        rasterizeKernel<<<blocks, threads, 0, stream>>>(
            d_gaussians_2d,
            d_tile_lists,
            d_tile_counts,
            d_tile_offsets,
            d_output_image,
            render
        );
    }
}

// Clear output image
__global__ void clearImageKernel(
    float4* __restrict__ output_image,
    int width,
    int height,
    float4 clear_color
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;
    
    if (idx < total_pixels) {
        output_image[idx] = clear_color;
    }
}

void launchClearImageKernel(
    float4* d_output_image,
    int width,
    int height,
    float4 clear_color,
    cudaStream_t stream
) {
    int total_pixels = width * height;
    dim3 blocks((total_pixels + 255) / 256);
    dim3 threads(256);
    
    clearImageKernel<<<blocks, threads, 0, stream>>>(
        d_output_image,
        width,
        height,
        clear_color
    );
}

} // namespace CUDA
} // namespace SplatRender