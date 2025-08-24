#include "../cuda_constants.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

namespace SplatRender {
namespace CUDA {

namespace cg = cooperative_groups;

// ============================================================================
// Bitonic Sort for Small Arrays (within a warp)
// ============================================================================

__device__ void bitonicSortWarp(float* keys, int* values, int count) {
    int tid = threadIdx.x & 31;  // Thread index within warp
    
    // Bitonic sort within warp (up to 32 elements)
    for (int size = 2; size <= 32 && size <= count; size *= 2) {
        for (int stride = size / 2; stride > 0; stride /= 2) {
            int idx1 = tid;
            int idx2 = tid ^ stride;
            
            if (idx2 > idx1 && idx2 < count) {
                if ((tid & size) == 0) {
                    // Sort ascending
                    if (keys[idx1] > keys[idx2]) {
                        // Swap
                        float temp_key = keys[idx1];
                        keys[idx1] = keys[idx2];
                        keys[idx2] = temp_key;
                        
                        int temp_val = values[idx1];
                        values[idx1] = values[idx2];
                        values[idx2] = temp_val;
                    }
                } else {
                    // Sort descending (for bitonic sequence)
                    if (keys[idx1] < keys[idx2]) {
                        // Swap
                        float temp_key = keys[idx1];
                        keys[idx1] = keys[idx2];
                        keys[idx2] = temp_key;
                        
                        int temp_val = values[idx1];
                        values[idx1] = values[idx2];
                        values[idx2] = temp_val;
                    }
                }
            }
            __syncwarp();
        }
    }
}

// ============================================================================
// Bitonic Sort for Medium Arrays (within a block)
// ============================================================================

__device__ void bitonicSortBlock(float* s_keys, int* s_values, int count) {
    int tid = threadIdx.x;
    
    // Bitonic sort for larger arrays (up to block size)
    for (int size = 2; size <= count; size *= 2) {
        // Bitonic merge
        for (int stride = size / 2; stride > 0; stride /= 2) {
            __syncthreads();
            
            int idx = tid;
            if (idx < count) {
                int partner = idx ^ stride;
                
                if (partner < count && partner > idx) {
                    bool ascending = ((idx & size) == 0);
                    
                    if ((ascending && s_keys[idx] > s_keys[partner]) ||
                        (!ascending && s_keys[idx] < s_keys[partner])) {
                        // Swap
                        float temp_key = s_keys[idx];
                        s_keys[idx] = s_keys[partner];
                        s_keys[partner] = temp_key;
                        
                        int temp_val = s_values[idx];
                        s_values[idx] = s_values[partner];
                        s_values[partner] = temp_val;
                    }
                }
            }
        }
    }
    __syncthreads();
}

// ============================================================================
// Sorting Kernel - Sort Gaussians within each tile by depth
// ============================================================================

__global__ void sortTileGaussiansKernel(
    int* __restrict__ tile_lists,
    float* __restrict__ tile_depths,
    const int* __restrict__ tile_counts,
    const int* __restrict__ tile_offsets,
    const int total_tiles
) {
    // One block per tile
    int tile_idx = blockIdx.x;
    if (tile_idx >= total_tiles) return;
    
    int count = tile_counts[tile_idx];
    if (count <= 1) return;  // No need to sort
    
    int offset = tile_offsets[tile_idx];
    int tid = threadIdx.x;
    
    // Allocate shared memory for sorting
    extern __shared__ char s_mem[];
    float* s_keys = (float*)s_mem;
    int* s_values = (int*)&s_keys[MAX_SHARED_GAUSSIANS];
    
    // Load data into shared memory (coalesced)
    for (int i = tid; i < count; i += blockDim.x) {
        s_keys[i] = tile_depths[offset + i];
        s_values[i] = tile_lists[offset + i];
    }
    __syncthreads();
    
    // Choose sorting algorithm based on count
    if (count <= 32) {
        // Small sort - use single warp
        if (tid < 32) {
            bitonicSortWarp(s_keys, s_values, count);
        }
    } else if (count <= blockDim.x) {
        // Medium sort - use block-wide bitonic sort
        bitonicSortBlock(s_keys, s_values, count);
    } else {
        // Large sort - use radix sort or iterative bitonic
        // For now, we'll do a simple bubble sort (not optimal)
        for (int i = 0; i < count - 1; i++) {
            __syncthreads();
            
            // Parallel odd-even sort
            int idx = tid * 2 + (i & 1);
            if (idx < count - 1) {
                if (s_keys[idx] > s_keys[idx + 1]) {
                    // Swap
                    float temp_key = s_keys[idx];
                    s_keys[idx] = s_keys[idx + 1];
                    s_keys[idx + 1] = temp_key;
                    
                    int temp_val = s_values[idx];
                    s_values[idx] = s_values[idx + 1];
                    s_values[idx + 1] = temp_val;
                }
            }
        }
    }
    __syncthreads();
    
    // Write sorted data back to global memory
    for (int i = tid; i < count; i += blockDim.x) {
        tile_depths[offset + i] = s_keys[i];
        tile_lists[offset + i] = s_values[i];
    }
}

// ============================================================================
// Radix Sort Kernel for Large Arrays
// ============================================================================

__global__ void radixSortKernel(
    float* __restrict__ keys,
    int* __restrict__ values,
    float* __restrict__ keys_out,
    int* __restrict__ values_out,
    int count,
    int bit
) {
    extern __shared__ int s_histogram[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Initialize histogram
    if (tid < 2) {
        s_histogram[tid] = 0;
    }
    __syncthreads();
    
    // Build histogram
    if (idx < count) {
        // Convert float to sortable int
        int key_int = __float_as_int(keys[idx]);
        // Handle negative floats
        key_int = (key_int >= 0) ? (key_int | 0x80000000) : (~key_int);
        
        int bucket = (key_int >> bit) & 1;
        atomicAdd(&s_histogram[bucket], 1);
    }
    __syncthreads();
    
    // Compute offsets (prefix sum)
    if (tid == 0) {
        int temp = s_histogram[0];
        s_histogram[0] = 0;
        s_histogram[1] = temp;
    }
    __syncthreads();
    
    // Scatter elements
    if (idx < count) {
        int key_int = __float_as_int(keys[idx]);
        key_int = (key_int >= 0) ? (key_int | 0x80000000) : (~key_int);
        
        int bucket = (key_int >> bit) & 1;
        int offset = atomicAdd(&s_histogram[bucket], 1);
        
        keys_out[blockIdx.x * blockDim.x + offset] = keys[idx];
        values_out[blockIdx.x * blockDim.x + offset] = values[idx];
    }
}

// ============================================================================
// Host-side kernel launcher
// ============================================================================

void launchSortingKernel(
    int* d_tile_lists,
    float* d_tile_depths,
    const int* d_tile_counts,
    const int* d_tile_offsets,
    int total_tiles,
    cudaStream_t stream
) {
    // Launch one block per tile
    dim3 blocks(total_tiles);
    dim3 threads(SORTING_BLOCK_SIZE);
    
    // Calculate shared memory size
    size_t shared_mem_size = sizeof(float) * MAX_SHARED_GAUSSIANS +  // Keys
                            sizeof(int) * MAX_SHARED_GAUSSIANS;      // Values
    
    sortTileGaussiansKernel<<<blocks, threads, shared_mem_size, stream>>>(
        d_tile_lists,
        d_tile_depths,
        d_tile_counts,
        d_tile_offsets,
        total_tiles
    );
}

} // namespace CUDA
} // namespace SplatRender