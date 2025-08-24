#include "../cuda_constants.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace SplatRender {
namespace CUDA {

// ============================================================================
// Tiling Kernel - Assign Gaussians to screen tiles
// ============================================================================

__global__ void tilingKernel(
    const GaussianData2D* __restrict__ gaussians_2d,
    int* __restrict__ tile_lists,         // Flattened array of tile lists
    int* __restrict__ tile_counts,        // Number of Gaussians per tile
    float* __restrict__ tile_depths,      // Depth values for sorting
    const int num_visible,
    const RenderParams render
) {
    // Get global thread index (one thread per visible Gaussian)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_visible) return;
    
    // Load 2D Gaussian
    GaussianData2D g2d = gaussians_2d[idx];
    
    // ========================================================================
    // Step 1: Compute bounding box in tile space
    // ========================================================================
    
    int min_tile_x, max_tile_x, min_tile_y, max_tile_y;
    computeBoundingBox(
        g2d.center, g2d.radius,
        render.screen_width, render.screen_height,
        min_tile_x, max_tile_x, min_tile_y, max_tile_y
    );
    
    // ========================================================================
    // Step 2: Assign Gaussian to all affected tiles
    // ========================================================================
    
    for (int tile_y = min_tile_y; tile_y <= max_tile_y; tile_y++) {
        for (int tile_x = min_tile_x; tile_x <= max_tile_x; tile_x++) {
            // Compute tile index
            int tile_idx = tile_y * render.tiles_x + tile_x;
            
            // Bounds check
            if (tile_idx < 0 || tile_idx >= render.total_tiles) continue;
            
            // Atomically increment count and get insertion position
            int position = atomicAdd(&tile_counts[tile_idx], 1);
            
            // Check for overflow
            if (position >= MAX_GAUSSIANS_PER_TILE) {
                // Rollback the counter increment
                atomicSub(&tile_counts[tile_idx], 1);
                continue;
            }
            
            // Calculate global index in tile lists array
            int list_idx = tile_idx * MAX_GAUSSIANS_PER_TILE + position;
            
            // Store Gaussian index and depth
            tile_lists[list_idx] = idx;
            tile_depths[list_idx] = g2d.depth;
        }
    }
}

// ============================================================================
// Compaction Kernel - Compact tile lists to remove gaps
// ============================================================================

__global__ void compactTileListsKernel(
    const int* __restrict__ tile_lists_in,
    const float* __restrict__ tile_depths_in,
    int* __restrict__ tile_lists_out,
    float* __restrict__ tile_depths_out,
    int* __restrict__ tile_offsets,
    const int* __restrict__ tile_counts,
    const int total_tiles
) {
    // One thread per tile
    int tile_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile_idx >= total_tiles) return;
    
    int count = tile_counts[tile_idx];
    if (count == 0) {
        tile_offsets[tile_idx] = 0;
        return;
    }
    
    // Calculate offset for this tile (exclusive scan would be better)
    int offset = 0;
    for (int i = 0; i < tile_idx; i++) {
        offset += tile_counts[i];
    }
    tile_offsets[tile_idx] = offset;
    
    // Copy Gaussians for this tile to compacted array
    int in_base = tile_idx * MAX_GAUSSIANS_PER_TILE;
    for (int i = 0; i < count; i++) {
        tile_lists_out[offset + i] = tile_lists_in[in_base + i];
        tile_depths_out[offset + i] = tile_depths_in[in_base + i];
    }
}

// ============================================================================
// Prefix Sum Kernel - Calculate tile offsets efficiently
// ============================================================================

__global__ void prefixSumKernel(
    const int* __restrict__ tile_counts,
    int* __restrict__ tile_offsets,
    int total_tiles
) {
    extern __shared__ int s_data[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Load data into shared memory
    s_data[tid] = (idx < total_tiles) ? tile_counts[idx] : 0;
    __syncthreads();
    
    // Up-sweep (reduce) phase
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            s_data[index] += s_data[index - stride];
        }
        __syncthreads();
    }
    
    // Down-sweep phase
    if (tid == blockDim.x - 1) {
        s_data[tid] = 0;
    }
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            int temp = s_data[index - stride];
            s_data[index - stride] = s_data[index];
            s_data[index] += temp;
        }
        __syncthreads();
    }
    
    // Write result
    if (idx < total_tiles) {
        tile_offsets[idx] = s_data[tid];
    }
}

// ============================================================================
// Host-side kernel launchers
// ============================================================================

void launchTilingKernel(
    const GaussianData2D* d_gaussians_2d,
    int* d_tile_lists,
    int* d_tile_counts,
    float* d_tile_depths,
    int num_visible,
    const RenderParams& render,
    cudaStream_t stream
) {
    // Reset tile counts
    cudaMemsetAsync(d_tile_counts, 0, render.total_tiles * sizeof(int), stream);
    
    // Calculate grid dimensions
    dim3 blocks((num_visible + TILING_BLOCK_SIZE - 1) / TILING_BLOCK_SIZE);
    dim3 threads(TILING_BLOCK_SIZE);
    
    // Launch tiling kernel
    tilingKernel<<<blocks, threads, 0, stream>>>(
        d_gaussians_2d,
        d_tile_lists,
        d_tile_counts,
        d_tile_depths,
        num_visible,
        render
    );
}

void launchCompactionKernel(
    const int* d_tile_lists_in,
    const float* d_tile_depths_in,
    int* d_tile_lists_out,
    float* d_tile_depths_out,
    int* d_tile_offsets,
    const int* d_tile_counts,
    int total_tiles,
    cudaStream_t stream
) {
    dim3 blocks((total_tiles + 255) / 256);
    dim3 threads(256);
    
    compactTileListsKernel<<<blocks, threads, 0, stream>>>(
        d_tile_lists_in,
        d_tile_depths_in,
        d_tile_lists_out,
        d_tile_depths_out,
        d_tile_offsets,
        d_tile_counts,
        total_tiles
    );
}

} // namespace CUDA
} // namespace SplatRender