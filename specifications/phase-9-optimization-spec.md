# Phase 9: CUDA Optimization Specification

## Overview
This document specifies the optimization strategies for Phase 9 of the SplatRender project, focusing on maximizing GPU performance for the CUDA kernels implemented in Phase 8.

## Goals
- Achieve consistent 60+ FPS at 1920x1080 with 100K+ Gaussians
- Reduce memory bandwidth usage by 30-50%
- Maximize GPU occupancy (target: 70%+)
- Minimize kernel launch overhead
- Implement performance monitoring and profiling infrastructure

## Current Performance Baseline
*To be measured after initial profiling*
- Current FPS with 100K Gaussians: TBD
- Memory bandwidth utilization: TBD
- GPU occupancy: TBD
- Kernel execution times: TBD

## Optimization Categories

### 1. Memory Access Optimization

#### 1.1 Shared Memory Utilization
**Goal**: Maximize L1 cache and shared memory usage to reduce global memory access

**Strategy**:
```cuda
// Current approach (global memory access)
Gaussian2D g = gaussians_2d[gaussian_idx];

// Optimized approach (shared memory staging)
__shared__ Gaussian2D shared_gaussians[BLOCK_SIZE];
if (threadIdx.x < num_to_load) {
    shared_gaussians[threadIdx.x] = gaussians_2d[block_offset + threadIdx.x];
}
__syncthreads();
```

**Implementation Tasks**:
- Analyze current shared memory usage per kernel
- Implement tile-based loading for projection kernel
- Use shared memory for frequently accessed matrices
- Optimize shared memory bank conflicts

#### 1.2 Memory Coalescing
**Goal**: Ensure all global memory accesses are coalesced

**Analysis Points**:
- Structure of Arrays (SoA) vs Array of Structures (AoS)
- Alignment requirements (128-byte boundaries)
- Access patterns in each kernel

**Optimizations**:
```cuda
// Convert Gaussian2D from AoS to SoA
struct Gaussians2D_SoA {
    float* center_x;
    float* center_y;
    float* cov_a;
    float* cov_b;
    float* cov_c;
    float* color_r;
    float* color_g;
    float* color_b;
    float* alpha;
    float* depth;
};
```

#### 1.3 Constant Memory Usage
**Goal**: Move read-only data to constant memory for broadcast efficiency

**Candidates**:
- View matrix (16 floats)
- Projection matrix (16 floats)
- Camera parameters
- Tile dimensions

**Implementation**:
```cuda
__constant__ float c_view_matrix[16];
__constant__ float c_proj_matrix[16];
__constant__ CameraParams c_camera;

// Host code
cudaMemcpyToSymbol(c_view_matrix, view_matrix, 16 * sizeof(float));
```

### 2. Register Pressure Reduction

#### 2.1 Register Usage Analysis
**Goal**: Keep register usage below 64 per thread for maximum occupancy

**Current Usage** (to be measured):
- projectGaussiansKernel: TBD registers
- tilingKernel: TBD registers
- sortingKernel: TBD registers
- rasterizeKernel: TBD registers

**Optimization Strategies**:
```cuda
// Use float2/float4 for better packing
float4 packed_cov = make_float4(cov.xx, cov.xy, cov.yy, alpha);

// Recompute instead of storing when beneficial
float det = packed_cov.x * packed_cov.z - packed_cov.y * packed_cov.y;
```

#### 2.2 Loop Unrolling Control
```cuda
#pragma unroll 4  // Controlled unrolling
for (int i = 0; i < num_gaussians; i += 4) {
    // Process 4 Gaussians per iteration
}
```

### 3. Algorithm Optimization

#### 3.1 Early Culling Improvements
**Current Culling**:
- View frustum culling
- Behind camera check

**Enhanced Culling**:
```cuda
// Size-based culling
float screen_size = radius * 2.0f / depth;
if (screen_size < MIN_PIXEL_SIZE) return;

// Opacity-based culling
if (opacity < MIN_OPACITY_THRESHOLD) return;

// Tile boundary culling (conservative)
if (!intersectsTileBounds(gaussian, tile)) return;
```

#### 3.2 Tile Size Optimization
**Current**: Fixed 16x16 tiles

**Dynamic Tile Sizing**:
```cuda
// Adaptive tile size based on Gaussian density
int tile_size = (gaussian_density > HIGH_DENSITY) ? 8 : 
                (gaussian_density < LOW_DENSITY) ? 32 : 16;
```

**Experimental Sizes**:
- 8x8: Better for high density areas
- 16x16: Current baseline
- 32x32: Better for sparse areas

#### 3.3 Sorting Algorithm Selection
**Current**: Bitonic sort for all tile sizes

**Adaptive Sorting**:
```cuda
if (count <= 32) {
    // Warp-level sort (no shared memory sync)
    warpSort(keys, values, count);
} else if (count <= 256) {
    // Bitonic sort in shared memory
    bitonicSort(keys, values, count);
} else {
    // Radix sort for large counts
    radixSort(keys, values, count);
}
```

### 4. Occupancy Optimization

#### 4.1 Block Size Tuning
**Goal**: Find optimal thread block sizes for each kernel

**Test Matrix**:
| Kernel | Current | Test Sizes | Optimal |
|--------|---------|------------|---------|
| Projection | 256 | 128, 256, 512 | TBD |
| Tiling | 256 | 128, 256, 512 | TBD |
| Sorting | 256 | 64, 128, 256 | TBD |
| Rasterization | 16x16 | 8x8, 16x16, 32x32 | TBD |

#### 4.2 Launch Configuration Optimization
```cuda
// Dynamic launch configuration
int block_size;
int min_grid_size;
cudaOccupancyMaxPotentialBlockSize(
    &min_grid_size, &block_size, 
    kernelFunction, 0, 0
);
```

### 5. Warp-Level Optimizations

#### 5.1 Warp Primitives
```cuda
// Use warp vote functions
unsigned mask = __ballot_sync(0xffffffff, pixel_covered);
if (__popc(mask) < MIN_PIXELS_PER_WARP) return;

// Warp-level reductions
float sum = __shfl_down_sync(0xffffffff, value, delta);
```

#### 5.2 Divergence Minimization
```cuda
// Group similar work in warps
int warp_id = threadIdx.x / 32;
int lane_id = threadIdx.x % 32;

// Process Gaussians in depth-sorted order within warp
int gaussian_idx = warp_id * 32 + lane_id;
```

### 6. Mixed Precision Investigation

#### 6.1 FP16 for Color Components
```cuda
// Use half precision for colors
__half3 color_half = __float2half3_rn(color_float);

// Compute in FP16 when possible
__half alpha = __hmul(opacity_half, gaussian_value_half);
```

#### 6.2 Tensor Core Utilization (RTX 2070)
```cuda
// Investigate wmma API for matrix operations
#include <mma.h>
using namespace nvcuda;

wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
```

## Performance Monitoring Infrastructure

### 1. Kernel Timing System
```cpp
class KernelProfiler {
public:
    struct KernelStats {
        float min_ms = FLT_MAX;
        float max_ms = 0.0f;
        float avg_ms = 0.0f;
        float total_ms = 0.0f;
        int call_count = 0;
    };
    
    void startTimer(const std::string& kernel_name);
    void endTimer(const std::string& kernel_name);
    KernelStats getStats(const std::string& kernel_name);
    void printReport();
    
private:
    std::map<std::string, KernelStats> kernel_stats;
    std::map<std::string, cudaEvent_t> start_events;
    std::map<std::string, cudaEvent_t> stop_events;
};
```

### 2. Memory Profiling
```cpp
class MemoryProfiler {
    struct MemoryStats {
        size_t peak_usage;
        size_t current_usage;
        size_t allocation_count;
        float bandwidth_gb_s;
    };
    
    void recordAllocation(size_t bytes);
    void recordDeallocation(size_t bytes);
    void recordTransfer(size_t bytes, float time_ms);
    MemoryStats getStats();
};
```

### 3. Performance Metrics Dashboard
```cpp
class PerformanceMonitor {
    void update(float frame_time_ms);
    void displayOverlay();
    
    struct Metrics {
        float fps;
        float frame_time_ms;
        float gpu_utilization;
        float memory_bandwidth;
        int rendered_gaussians;
        int culled_gaussians;
    };
};
```

## Testing and Validation

### 1. Performance Regression Tests
```cpp
class PerformanceTest {
    bool runTest(const std::string& test_name);
    bool validatePerformance(float baseline_ms, float current_ms);
    
    // Test scenarios
    void test_10k_gaussians();
    void test_100k_gaussians();
    void test_1m_gaussians();
    void test_dense_scene();
    void test_sparse_scene();
};
```

### 2. Correctness Validation
- Pixel-perfect comparison with CPU implementation
- Statistical metrics (PSNR, SSIM)
- Visual quality assessment

### 3. Stress Testing
- Maximum Gaussian count before performance degradation
- Memory pressure scenarios
- Thermal throttling behavior

## Implementation Plan

### Week 1: Profiling and Analysis
1. Set up Nsight Compute profiling
2. Baseline performance measurements
3. Identify primary bottlenecks
4. Create performance monitoring infrastructure

### Week 2: Memory Optimizations
1. Implement shared memory optimizations
2. Add constant memory for matrices
3. Optimize memory access patterns
4. Test SoA vs AoS layouts

### Week 3: Algorithm and Occupancy
1. Enhance culling algorithms
2. Experiment with tile sizes
3. Optimize block configurations
4. Implement adaptive algorithms

### Week 4: Advanced Optimizations
1. Add warp-level primitives
2. Investigate mixed precision
3. Final performance tuning
4. Documentation and benchmarks

## Success Metrics

### Performance Targets
- [ ] 60+ FPS with 100K Gaussians at 1920x1080
- [ ] 30+ FPS with 500K Gaussians at 1920x1080
- [ ] < 10ms kernel execution time for typical scenes
- [ ] 70%+ GPU occupancy average

### Quality Targets
- [ ] No visual quality degradation
- [ ] Numerically stable (no NaN/Inf)
- [ ] Consistent performance (< 10% variance)

### Resource Targets
- [ ] < 2GB VRAM for 100K Gaussians
- [ ] < 200 GB/s memory bandwidth
- [ ] < 150W power consumption

## Optimization Checklist

### Memory Optimizations
- [ ] Shared memory for Gaussian data
- [ ] Constant memory for matrices
- [ ] Coalesced memory access patterns
- [ ] Aligned data structures
- [ ] Texture memory evaluation

### Algorithm Optimizations
- [ ] Enhanced frustum culling
- [ ] Size-based culling
- [ ] Opacity threshold culling
- [ ] Adaptive tile sizing
- [ ] Optimized sorting algorithms

### GPU Utilization
- [ ] Optimal block sizes determined
- [ ] Maximum occupancy achieved
- [ ] Warp efficiency > 90%
- [ ] Minimal divergence

### Advanced Features
- [ ] Mixed precision evaluation
- [ ] Tensor core investigation
- [ ] Multi-stream execution
- [ ] Graph API consideration

## Tools and Commands

### Profiling with Nsight Compute
```bash
# Basic profiling
ncu --target-processes all ./build/SplatRender

# Detailed kernel analysis
ncu --kernel-name projectGaussiansKernel --launch-skip 10 --launch-count 1 ./build/SplatRender

# Full metrics collection
ncu --set full --export profile_report ./build/SplatRender
```

### Profiling with Nsight Systems
```bash
# System-wide profiling
nsys profile --stats=true ./build/SplatRender

# GPU trace
nsys profile --trace=cuda,opengl ./build/SplatRender
```

### CUDA Memcheck
```bash
# Memory leak detection
cuda-memcheck --leak-check full ./build/SplatRender

# Race condition detection
cuda-memcheck --tool racecheck ./build/SplatRender
```

## References

### Documentation
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Compute User Manual](https://docs.nvidia.com/nsight-compute/)
- [CUDA Optimization Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#performance-guidelines)

### Papers
- "Parallel Prefix Sum (Scan) with CUDA" - GPU Gems 3
- "Optimizing Parallel Reduction in CUDA" - Mark Harris
- "Fast Fixed-Radius Nearest Neighbors" - GPU Gems 3

---

*This specification will guide the optimization phase. Updates will be made as profiling reveals specific bottlenecks and optimization opportunities.*