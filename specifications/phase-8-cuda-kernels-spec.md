# Phase 8: CUDA Kernel Implementation Specification

## Overview
This document specifies the implementation details for Phase 8 of the SplatRender project, focusing on CUDA kernel development for GPU-accelerated 3D Gaussian Splatting on the NVIDIA RTX 2070.

## Goals
- Implement high-performance CUDA kernels for Gaussian splatting
- Achieve 60+ FPS at 1920x1080 with 100K+ Gaussians
- Optimize memory access patterns and GPU occupancy
- Maintain rendering quality matching CPU implementation

## System Context
- **GPU**: NVIDIA RTX 2070 (8GB VRAM, Compute Capability 7.5)
- **CUDA**: Version 12.0
- **SMs**: 36 Streaming Multiprocessors
- **Max Threads/Block**: 1024
- **Warp Size**: 32
- **Shared Memory**: 48KB per SM

## Architecture Overview

### Rendering Pipeline
```
1. Projection Stage (projectGaussiansKernel)
   ├── Transform 3D Gaussians to view space
   ├── Compute 2D covariance using Jacobian
   ├── View frustum culling
   └── Spherical harmonics evaluation

2. Tiling Stage (tilingKernel)
   ├── Compute Gaussian bounding boxes
   ├── Assign to 16x16 pixel tiles
   ├── Build per-tile lists
   └── Count Gaussians per tile

3. Sorting Stage (sortingKernel)
   ├── Per-tile depth sorting
   ├── Key-value pairs (depth, index)
   └── Radix sort in shared memory

4. Rasterization Stage (rasterizeKernel)
   ├── One thread block per tile
   ├── Load tile Gaussians to shared memory
   ├── Evaluate Gaussians per pixel
   └── Alpha blending to output
```

## Detailed Kernel Specifications

### 1. Projection Kernel (`projectGaussiansKernel`)

#### Purpose
Transform 3D Gaussians to 2D screen space with view-dependent color evaluation.

#### Kernel Signature
```cuda
__global__ void projectGaussiansKernel(
    const Gaussian3D* __restrict__ gaussians_3d,  // Input 3D Gaussians
    Gaussian2D* __restrict__ gaussians_2d,         // Output 2D Gaussians
    int* __restrict__ visible_count,               // Atomic counter for visible
    const float* __restrict__ view_matrix,         // 4x4 view matrix
    const float* __restrict__ proj_matrix,         // 4x4 projection matrix
    const float3 camera_pos,                       // Camera world position
    const int num_gaussians,                       // Total input count
    const int screen_width,
    const int screen_height,
    const float tan_fovx,                          // For culling optimization
    const float tan_fovy
);
```

#### Thread Configuration
- **Blocks**: `(num_gaussians + 255) / 256`
- **Threads/Block**: 256
- **One thread per Gaussian**

#### Algorithm
```cuda
1. Thread loads one Gaussian3D
2. Transform position to view space
3. Check frustum culling:
   - Behind camera (z > 0)
   - Outside FOV cone
4. Project to clip space
5. Perspective divide to NDC
6. Convert to screen coordinates
7. Compute 2D covariance:
   - Get 3D covariance from scale/rotation
   - Compute Jacobian of projection
   - Apply: Σ_2D = J * W * Σ_3D * W^T * J^T
8. Evaluate spherical harmonics for color
9. Compute screen radius (3-sigma)
10. If visible, atomically increment counter and write
```

#### Memory Access Pattern
- **Coalesced reads**: Sequential Gaussian3D access
- **Coalesced writes**: Sequential Gaussian2D output
- **Shared memory**: View/proj matrices (48 bytes each)

#### Optimizations
- Early culling reduces memory writes
- Warp-level primitives for atomic operations
- Shared memory for frequently accessed matrices
- Fast math intrinsics for trigonometric functions

### 2. Tiling Kernel (`tilingKernel`)

#### Purpose
Assign visible Gaussians to screen tiles for localized processing.

#### Kernel Signature
```cuda
__global__ void tilingKernel(
    const Gaussian2D* __restrict__ gaussians_2d,
    int* __restrict__ tile_lists,        // Flattened tile lists
    int* __restrict__ tile_counts,       // Gaussians per tile
    int* __restrict__ tile_offsets,      // Start index per tile
    const int num_gaussians,
    const int tiles_x,
    const int tiles_y,
    const int max_per_tile
);
```

#### Thread Configuration
- **Blocks**: `(num_gaussians + 255) / 256`
- **Threads/Block**: 256
- **One thread per Gaussian**

#### Algorithm
```cuda
1. Thread loads one Gaussian2D
2. Compute bounding box in tiles:
   - min_tile_x = (center.x - radius) / TILE_SIZE
   - max_tile_x = (center.x + radius) / TILE_SIZE
   - Similar for y
3. For each affected tile:
   - Atomically increment tile count
   - Get insertion index
   - Write Gaussian index to tile list
4. Handle overlapping tiles efficiently
```

#### Data Structure
```cuda
// Tile list organization (Structure of Arrays)
struct TileData {
    int gaussian_indices[MAX_GAUSSIANS_PER_TILE];
    float depths[MAX_GAUSSIANS_PER_TILE];
    int count;
};
```

#### Memory Considerations
- **Worst case**: One Gaussian affects multiple tiles
- **Average case**: 2-4 tiles per Gaussian
- **Memory bound**: Atomic operations are bottleneck

### 3. Sorting Kernel (`sortingKernel`)

#### Purpose
Sort Gaussians within each tile by depth for correct alpha blending.

#### Kernel Signature
```cuda
__global__ void sortingKernel(
    int* __restrict__ tile_lists,
    float* __restrict__ tile_depths,
    const Gaussian2D* __restrict__ gaussians_2d,
    const int* __restrict__ tile_counts,
    const int tiles_x,
    const int tiles_y,
    const int max_per_tile
);
```

#### Thread Configuration
- **Blocks**: One per tile (`tiles_x * tiles_y`)
- **Threads/Block**: 256 (or adaptive based on count)
- **Cooperative groups for dynamic parallelism**

#### Algorithm
```cuda
1. Each block handles one tile
2. Load Gaussian indices and depths to shared memory
3. Perform bitonic sort in shared memory:
   - Key: depth (float)
   - Value: Gaussian index (int)
4. Write sorted indices back to global memory
5. Handle variable counts with early exit
```

#### Sorting Strategy
- **Small counts** (< 32): Single warp sort
- **Medium counts** (32-256): Block-wide bitonic sort
- **Large counts** (> 256): Radix sort with multiple passes

#### Shared Memory Layout
```cuda
__shared__ float s_depths[MAX_SHARED_GAUSSIANS];
__shared__ int s_indices[MAX_SHARED_GAUSSIANS];
```

### 4. Rasterization Kernel (`rasterizeKernel`)

#### Purpose
Render sorted Gaussians to output image with alpha blending.

#### Kernel Signature
```cuda
__global__ void rasterizeKernel(
    const Gaussian2D* __restrict__ gaussians_2d,
    const int* __restrict__ tile_lists,
    const int* __restrict__ tile_counts,
    float4* __restrict__ output_image,    // RGBA output
    const int tiles_x,
    const int tiles_y,
    const int screen_width,
    const int screen_height
);
```

#### Thread Configuration
- **Blocks**: One per tile
- **Threads/Block**: 16x16 = 256 (one per pixel)
- **2D thread indexing within tile**

#### Algorithm
```cuda
1. Each thread handles one pixel
2. Compute pixel position (tile_base + thread_idx)
3. Initialize accumulated color = (0,0,0,0)
4. Initialize transmittance T = 1.0
5. For each Gaussian in tile (front-to-back):
   a. Load Gaussian to shared memory (coalesced)
   b. Sync threads
   c. Evaluate Gaussian at pixel:
      - Compute distance from center
      - Evaluate 2D Gaussian: exp(-0.5 * d^T * Σ^-1 * d)
   d. Compute alpha: α = opacity * gaussian_value
   e. Update color: C += α * T * gaussian_color
   f. Update transmittance: T *= (1 - α)
   g. Early termination if T < 0.001
6. Write final color to output
```

#### Shared Memory Optimization
```cuda
// Load Gaussians in batches to shared memory
__shared__ struct {
    float2 center[BATCH_SIZE];
    float4 cov_2d[BATCH_SIZE];  // Packed covariance
    float3 color[BATCH_SIZE];
    float alpha[BATCH_SIZE];
} s_gaussians;
```

#### Optimizations
- **Warp divergence**: Minimize with sorted depth order
- **Memory coalescing**: Batch loads to shared memory
- **Register pressure**: Pack data structures
- **Early termination**: Skip pixels with low transmittance

## Memory Layout and Management

### Global Memory Organization
```
Device Memory Layout:
├── Input Gaussians3D    [N * 176 bytes]
├── Projected Gaussians2D [N * 64 bytes]
├── Tile Lists           [TILES * MAX_PER_TILE * 4 bytes]
├── Tile Counts          [TILES * 4 bytes]
├── Tile Offsets         [TILES * 4 bytes]
└── Output Image         [WIDTH * HEIGHT * 16 bytes]
```

### Memory Bandwidth Optimization
1. **Coalesced Access**: Align structures to 128-byte boundaries
2. **Texture Memory**: Consider for Gaussian data (spatial locality)
3. **Constant Memory**: View/projection matrices
4. **L2 Cache**: Optimize for 4MB L2 on RTX 2070

### Memory Pool Strategy
```cuda
// Pre-allocate maximum expected memory
struct GpuMemoryPool {
    Gaussian3D* gaussians_3d;     // Max: 200K
    Gaussian2D* gaussians_2d;     // Max: 200K
    int* tile_lists;              // Max: TILES * 1000
    float* tile_depths;           // Max: TILES * 1000
    int* tile_counts;             // TILES
    int* tile_offsets;            // TILES
};
```

## Performance Optimization Strategies

### 1. Occupancy Optimization
- **Target**: 50%+ occupancy
- **Register usage**: < 64 per thread
- **Shared memory**: < 48KB per block
- **Block size**: 256 threads (8 warps)

### 2. Memory Access Patterns
```cuda
// Good: Coalesced access
float4 data = gaussians[blockIdx.x * blockDim.x + threadIdx.x];

// Bad: Strided access
float4 data = gaussians[threadIdx.x * stride];
```

### 3. Divergence Minimization
```cuda
// Use warp-level primitives
unsigned mask = __ballot_sync(0xffffffff, condition);
if (mask != 0) {
    // All threads in warp take same path
}
```

### 4. Atomic Operation Optimization
```cuda
// Use atomicAdd for counters
int index = atomicAdd(&tile_counts[tile_id], 1);

// Consider atomicCAS for complex updates
int old = tile_data[idx];
int assumed;
do {
    assumed = old;
    int new_val = compute_update(assumed);
    old = atomicCAS(&tile_data[idx], assumed, new_val);
} while (assumed != old);
```

## Launch Configuration Guidelines

### Projection Kernel
```cuda
dim3 blocks((num_gaussians + 255) / 256);
dim3 threads(256);
projectGaussiansKernel<<<blocks, threads, shared_size>>>(...)
```

### Tiling Kernel
```cuda
dim3 blocks((num_gaussians + 255) / 256);
dim3 threads(256);
tilingKernel<<<blocks, threads>>>(...)
```

### Sorting Kernel
```cuda
dim3 blocks(tiles_x * tiles_y);
dim3 threads(256);  // Or adaptive
sortingKernel<<<blocks, threads, shared_memory_size>>>(...)
```

### Rasterization Kernel
```cuda
dim3 blocks(tiles_x, tiles_y);
dim3 threads(16, 16);  // One thread per pixel in tile
rasterizeKernel<<<blocks, threads, shared_memory_size>>>(...)
```

## Error Handling

### Kernel Error Checking
```cuda
// After each kernel launch
CUDA_CHECK_KERNEL();  // Macro from cuda_utils.h

// For debugging
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("Kernel failed: %s\n", cudaGetErrorString(err));
}
```

### Bounds Checking
```cuda
// Always check array bounds
if (gaussian_id >= num_gaussians) return;
if (tile_x >= tiles_x || tile_y >= tiles_y) return;
```

### Overflow Protection
```cuda
// Protect against tile list overflow
int count = atomicAdd(&tile_counts[tile_id], 1);
if (count >= MAX_PER_TILE) {
    atomicSub(&tile_counts[tile_id], 1);
    return;
}
```

## Testing Strategy

### Unit Tests
1. **Projection Test**: Compare with CPU implementation
2. **Tiling Test**: Verify tile assignments
3. **Sorting Test**: Check depth ordering
4. **Rasterization Test**: Pixel-perfect comparison

### Performance Benchmarks
```cuda
// Use CUDA events for timing
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
kernel<<<blocks, threads>>>(...);
cudaEventRecord(stop);

cudaEventSynchronize(stop);
float ms;
cudaEventElapsedTime(&ms, start, stop);
```

### Validation Methods
1. **Checksum validation**: Sum of all pixel values
2. **Visual comparison**: Side-by-side CPU/GPU
3. **Statistical analysis**: PSNR, SSIM metrics

## Implementation Phases

### Phase 8.1: Basic Kernels (Week 1)
1. Implement projection kernel
2. Implement simple tiling (no sorting)
3. Basic rasterization (no optimization)
4. Verify correctness

### Phase 8.2: Sorting & Optimization (Week 2)
1. Implement sorting kernel
2. Add shared memory optimizations
3. Implement early termination
4. Profile and optimize

### Phase 8.3: Advanced Features (Week 3)
1. Dynamic tile sizing
2. Adaptive algorithms
3. Multi-resolution rendering
4. Performance tuning

## Performance Targets

### Kernel Timings (100K Gaussians)
- **Projection**: < 2ms
- **Tiling**: < 1ms
- **Sorting**: < 3ms
- **Rasterization**: < 10ms
- **Total**: < 16ms (60+ FPS)

### Memory Usage
- **Peak VRAM**: < 2GB
- **Bandwidth**: < 200 GB/s
- **L2 Cache Hit Rate**: > 80%

### Quality Metrics
- **PSNR vs CPU**: > 40dB
- **No visible artifacts**
- **Correct alpha blending**

## Common Pitfalls and Solutions

### 1. Race Conditions
**Problem**: Multiple threads writing to same tile list
**Solution**: Use atomic operations with proper ordering

### 2. Memory Overflow
**Problem**: Too many Gaussians per tile
**Solution**: Dynamic allocation or fallback to CPU

### 3. Warp Divergence
**Problem**: Different threads taking different paths
**Solution**: Sort by depth, use warp-level primitives

### 4. Register Spilling
**Problem**: Too many local variables
**Solution**: Reduce register usage, use shared memory

### 5. Uncoalesced Access
**Problem**: Strided or random memory access
**Solution**: Restructure data layout (SoA vs AoS)

## Integration with Existing System

### CudaRasterizer Updates
```cpp
class CudaRasterizer {
    // Add kernel launch methods
    void launchProjectionKernel(...);
    void launchTilingKernel(...);
    void launchSortingKernel(...);
    void launchRasterizationKernel(...);
    
    // Performance metrics
    struct KernelTimings {
        float projection_ms;
        float tiling_ms;
        float sorting_ms;
        float rasterization_ms;
    };
};
```

### Engine Integration
```cpp
// In Engine::render()
if (use_cuda) {
    cuda_rasterizer->render(gaussians, camera, output);
} else {
    cpu_rasterizer->render(gaussians, camera, output);
}
```

## Debugging Tools

### CUDA-GDB
```bash
cuda-gdb ./SplatRender
(cuda-gdb) break projectGaussiansKernel
(cuda-gdb) run
(cuda-gdb) info cuda threads
```

### Nsight Compute
```bash
ncu --target-processes all ./SplatRender
```

### Printf Debugging
```cuda
if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("Gaussian[0]: pos=(%.2f,%.2f,%.2f)\n", 
           g.position.x, g.position.y, g.position.z);
}
```

## Future Enhancements

### Advanced Features
1. **Level of Detail**: Adaptive Gaussian density
2. **Temporal Coherence**: Frame-to-frame optimization
3. **Multi-GPU**: Split workload across GPUs
4. **Ray Tracing**: RTX core integration

### Optimization Opportunities
1. **Tensor Cores**: Mixed precision computation
2. **Graph API**: Reduce kernel launch overhead
3. **Persistent Kernels**: Keep data in GPU memory
4. **Compression**: Reduce memory bandwidth

## References

### Academic Papers
- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [EWA Splatting](https://www.cs.umd.edu/~zwicker/publications/EWASplatting-TVCG02.pdf)

### CUDA Documentation
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)

### Implementation References
- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)

## Deliverables

### Code Files
1. `cuda_constants.h` - Shared constants and structures
2. `kernels/projection.cu` - Projection kernel implementation
3. `kernels/tiling.cu` - Tiling kernel implementation
4. `kernels/sorting.cu` - Sorting kernel implementation
5. `kernels/rasterization.cu` - Rasterization kernel implementation
6. `cuda_rasterizer.cu` - Updated integration
7. `test_kernels.cu` - Kernel unit tests

### Documentation
1. This specification document
2. Kernel API documentation
3. Performance analysis report
4. Integration guide

### Tests
1. Correctness tests against CPU
2. Performance benchmarks
3. Memory leak tests
4. Stress tests with large datasets

## Success Criteria

### Functional
- [ ] All kernels produce correct output
- [ ] Matches CPU implementation quality
- [ ] No memory leaks or crashes
- [ ] Handles edge cases gracefully

### Performance
- [ ] 60+ FPS with 100K Gaussians at 1080p
- [ ] < 2GB VRAM usage
- [ ] < 16ms total frame time
- [ ] Linear scaling with Gaussian count

### Quality
- [ ] Clean, documented code
- [ ] Comprehensive error handling
- [ ] Efficient memory usage
- [ ] Maintainable architecture

---

*This specification will be updated as implementation progresses and new optimizations are discovered.*