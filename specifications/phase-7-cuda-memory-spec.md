# Phase 7: CUDA Memory Management Specification

## Overview
This document specifies the implementation details for Phase 7 of the SplatRender project, focusing on CUDA memory management, device initialization, and OpenGL interoperability on Linux systems with NVIDIA GPUs.

## Goals
- Establish CUDA context and device management
- Implement efficient memory allocation and transfer mechanisms
- Set up CUDA-OpenGL interoperability for zero-copy rendering
- Create foundation for GPU-accelerated Gaussian splatting

## System Requirements
- **Platform**: Linux (Ubuntu 20.04+ recommended)
- **GPU**: NVIDIA GPU with Compute Capability 7.0+ (RTX 2070 or newer)
- **CUDA**: Version 11.8+ (tested with 12.0)
- **Driver**: NVIDIA driver 520+ 
- **VRAM**: Minimum 4GB, 8GB recommended

## Architecture Overview

### Module Structure
```
src/cuda/
├── cuda_utils.h           # Error checking, helpers, profiling
├── cuda_manager.h/.cu      # Device and context management
├── cuda_memory.h/.cu       # Memory allocation and transfers
├── cuda_gl_interop.h/.cu   # OpenGL-CUDA interoperability
└── cuda_rasterizer.h/.cu   # GPU rasterizer interface
```

### Class Hierarchy
```
CudaManager (Singleton)
    ├── Device Management
    ├── Context Handling
    └── Resource Lifecycle

CudaMemory<T>
    ├── Device Allocation
    ├── Host-Device Transfers
    └── Memory Pools

CudaGLInterop
    ├── Texture Registration
    ├── Surface Mapping
    └── Synchronization

CudaRasterizer : public Rasterizer
    ├── GPU Buffers
    ├── Kernel Launches
    └── Performance Metrics
```

## Detailed Implementation

### 1. CUDA Utilities (`cuda_utils.h`)
**Purpose**: Common utilities, error checking, and helper functions

**Key Components**:
- `CUDA_CHECK(call)`: Error checking macro with detailed error messages
- `CUDA_CHECK_KERNEL()`: Post-kernel launch error checking
- `CudaTimer`: High-precision GPU timing for profiling
- `getOptimalBlockSize()`: Calculate optimal kernel launch configuration
- `selectBestDevice()`: Choose GPU with most memory

**Error Handling Strategy**:
- All CUDA calls wrapped in CUDA_CHECK
- Exceptions thrown with detailed context
- Graceful fallback to CPU renderer on failure

### 2. CUDA Manager (`cuda_manager.h/cu`)
**Purpose**: Centralized CUDA context and device management

**Class Definition**:
```cpp
class CudaManager {
public:
    static CudaManager& getInstance();
    
    bool initialize(int deviceId = -1);
    void shutdown();
    
    // Device properties
    const cudaDeviceProp& getDeviceProperties() const;
    size_t getAvailableMemory() const;
    int getComputeCapability() const;
    
    // Stream management
    cudaStream_t getDefaultStream();
    cudaStream_t createStream();
    void destroyStream(cudaStream_t stream);
    
private:
    CudaManager() = default;
    int device_id_;
    cudaDeviceProp device_props_;
    std::vector<cudaStream_t> streams_;
    bool initialized_;
};
```

**Initialization Process**:
1. Query available devices
2. Select device (user-specified or best available)
3. Set device and create primary context
4. Query and store device properties
5. Create default stream for async operations

### 3. CUDA Memory Management (`cuda_memory.h/cu`)
**Purpose**: Efficient GPU memory allocation and data transfers

**Template Class**:
```cpp
template<typename T>
class CudaMemory {
public:
    CudaMemory(size_t count);
    ~CudaMemory();
    
    // Memory operations
    void allocate(size_t count);
    void free();
    void copyToDevice(const T* host_data, size_t count);
    void copyToHost(T* host_data, size_t count);
    void copyAsync(const T* host_data, size_t count, cudaStream_t stream);
    
    // Accessors
    T* getDevicePtr() { return d_ptr_; }
    size_t size() const { return count_; }
    size_t bytes() const { return count_ * sizeof(T); }
    
private:
    T* d_ptr_;
    size_t count_;
};
```

**Memory Pool Implementation**:
```cpp
class CudaMemoryPool {
public:
    void* allocate(size_t bytes);
    void deallocate(void* ptr);
    void reset();
    
private:
    struct Block {
        void* ptr;
        size_t size;
        bool in_use;
    };
    std::vector<Block> blocks_;
    size_t total_allocated_;
};
```

**Pinned Memory Support**:
- Use `cudaHostAlloc()` for pinned memory
- Enables faster transfers (up to 2x speedup)
- Automatic fallback to regular memory if allocation fails

### 4. CUDA-OpenGL Interoperability (`cuda_gl_interop.h/cu`)
**Purpose**: Zero-copy rendering through direct GPU memory access

**Class Definition**:
```cpp
class CudaGLInterop {
public:
    CudaGLInterop();
    ~CudaGLInterop();
    
    // OpenGL resource registration
    void registerTexture(GLuint texture, int width, int height);
    void unregisterTexture();
    
    // Resource mapping
    cudaSurfaceObject_t mapTextureForCuda();
    void unmapTexture();
    
    // Direct rendering
    void renderToTexture(const void* cuda_buffer, int width, int height);
    
private:
    cudaGraphicsResource_t cuda_resource_;
    cudaSurfaceObject_t surface_;
    cudaArray_t array_;
    bool is_mapped_;
};
```

**Interop Workflow**:
1. Create OpenGL texture
2. Register with CUDA: `cudaGraphicsGLRegisterImage()`
3. Map for CUDA access: `cudaGraphicsMapResources()`
4. Get CUDA array: `cudaGraphicsSubResourceGetMappedArray()`
5. Create surface object for kernel writes
6. Unmap after kernel execution
7. OpenGL can now display the texture

### 5. CUDA Rasterizer Interface (`cuda_rasterizer.h/cu`)
**Purpose**: GPU-accelerated rasterization matching CPU interface

**Class Definition**:
```cpp
class CudaRasterizer : public Rasterizer {
public:
    CudaRasterizer();
    ~CudaRasterizer();
    
    // Rasterizer interface
    bool initialize(const RenderSettings& settings) override;
    void render(const std::vector<Gaussian3D>& gaussians,
                const Camera& camera,
                std::vector<float>& output_buffer) override;
    
    // CUDA-specific methods
    void renderDirect(const Gaussian3D* d_gaussians, 
                     size_t count,
                     const Camera& camera,
                     cudaSurfaceObject_t surface);
    
    // Memory management
    void uploadGaussians(const std::vector<Gaussian3D>& gaussians);
    void allocateBuffers(size_t max_gaussians);
    
private:
    // Device buffers
    CudaMemory<Gaussian3D> d_gaussians_3d_;
    CudaMemory<Gaussian2D> d_gaussians_2d_;
    CudaMemory<uint32_t> d_tile_lists_;
    CudaMemory<float> d_depth_buffer_;
    
    // Interop
    std::unique_ptr<CudaGLInterop> gl_interop_;
    
    // Settings
    RenderSettings settings_;
    size_t max_gaussians_;
};
```

## Memory Layout and Requirements

### Per-Gaussian Memory:
- **Gaussian3D**: 176 bytes
  - Position (3 floats): 12 bytes
  - Scale (3 floats): 12 bytes
  - Rotation (4 floats): 16 bytes
  - SH coefficients (48 floats): 192 bytes
  - Opacity (1 float): 4 bytes
  - Padding: 4 bytes

- **Gaussian2D**: 64 bytes
  - Center (2 floats): 8 bytes
  - Covariance (4 floats): 16 bytes
  - Color (3 floats): 12 bytes
  - Alpha (1 float): 4 bytes
  - Depth (1 float): 4 bytes
  - Tile ID (1 int): 4 bytes
  - Radius (1 float): 4 bytes
  - Padding: 8 bytes

### Total Memory Estimation:
For 100K Gaussians at 1920x1080:
- Gaussian3D array: 17.6 MB
- Gaussian2D array: 6.4 MB
- Tile lists (worst case): ~50 MB
- Depth buffer: 8 MB
- **Total**: ~82 MB VRAM

### Memory Allocation Strategy:
1. **Pre-allocation**: Allocate for maximum expected Gaussians
2. **Dynamic growth**: Reallocate if exceeding capacity
3. **Memory pools**: Reuse allocations across frames
4. **Pinned memory**: For frequent CPU-GPU transfers

## Performance Considerations

### Optimization Strategies:
1. **Coalesced Memory Access**
   - Align data structures to 128-byte boundaries
   - Use Structure of Arrays (SoA) where beneficial

2. **Stream Usage**
   - Overlap compute and transfer
   - Multiple streams for independent operations

3. **Texture Memory**
   - Consider texture cache for read-only data
   - Spatial locality benefits for image access

4. **Unified Memory** (Optional)
   - Simplifies memory management
   - Automatic migration between host and device
   - May have performance overhead

### Profiling Metrics:
- Memory allocation time
- Transfer bandwidth (GB/s)
- Kernel execution time
- OpenGL interop overhead
- Total frame time

## Integration with Existing System

### CMake Configuration:
```cmake
if(USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES 75)  # For RTX 2070
    
    set(CUDA_SOURCES
        src/cuda/cuda_manager.cu
        src/cuda/cuda_memory.cu
        src/cuda/cuda_gl_interop.cu
        src/cuda/cuda_rasterizer.cu
    )
    
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --use_fast_math
            --relocatable-device-code=true
            -lineinfo
        >
    )
endif()
```

### Engine Integration:
```cpp
// In Engine::initialize()
if (use_cuda && CudaManager::getInstance().initialize()) {
    rasterizer_ = std::make_unique<CudaRasterizer>();
    std::cout << "Using CUDA rasterizer" << std::endl;
} else {
    rasterizer_ = std::make_unique<CPURasterizer>();
    std::cout << "Using CPU rasterizer" << std::endl;
}
```

## Testing Strategy

### Unit Tests:
1. **Memory Allocation Tests**
   - Allocate/free various sizes
   - Verify alignment
   - Check for memory leaks

2. **Transfer Tests**
   - Host to device transfer correctness
   - Async transfer completion
   - Bandwidth measurement

3. **Interop Tests**
   - Texture registration/unregistration
   - Map/unmap cycles
   - Rendering to texture

### Integration Tests:
1. **Fallback Testing**
   - Graceful fallback when CUDA unavailable
   - Performance comparison CPU vs GPU

2. **Stress Testing**
   - Maximum Gaussian count
   - Memory pressure scenarios
   - Long-running stability

### Performance Benchmarks:
- Memory allocation: < 1ms for 100K Gaussians
- Transfer rate: > 10 GB/s for pinned memory
- Interop overhead: < 0.1ms per frame

## Error Handling

### Error Categories:
1. **Initialization Errors**
   - No CUDA device available
   - Insufficient compute capability
   - Driver version mismatch

2. **Memory Errors**
   - Out of memory
   - Invalid memory access
   - Alignment violations

3. **Interop Errors**
   - OpenGL context issues
   - Resource registration failures
   - Synchronization problems

### Recovery Strategies:
- Automatic fallback to CPU renderer
- Graceful degradation (reduce Gaussian count)
- Clear error messages for debugging
- Memory cleanup on failure

## Future Enhancements

### Phase 8 Preparation:
- Kernel launch infrastructure ready
- Memory layout optimized for kernels
- Profiling framework in place

### Potential Optimizations:
- Multi-GPU support
- Dynamic memory management
- Compression for memory savings
- Graph-based kernel execution

## Deliverables

### Code Files:
1. ✅ `cuda_utils.h` - Utilities and helpers
2. ⏳ `cuda_manager.h/cu` - Device management
3. ⏳ `cuda_memory.h/cu` - Memory operations
4. ⏳ `cuda_gl_interop.h/cu` - OpenGL interop
5. ⏳ `cuda_rasterizer.h/cu` - Rasterizer interface

### Documentation:
1. ✅ This specification document
2. ⏳ API documentation in headers
3. ⏳ Integration guide
4. ⏳ Performance tuning guide

### Tests:
1. ⏳ CUDA initialization test
2. ⏳ Memory management test
3. ⏳ Interop functionality test
4. ⏳ Performance benchmark suite

## Success Criteria

### Functional:
- [x] CUDA device detection and initialization
- [ ] Memory allocation and transfers working
- [ ] OpenGL interop established
- [ ] Fallback to CPU renderer functional

### Performance:
- [ ] Memory allocation < 5ms for 100K Gaussians
- [ ] Transfer bandwidth > 10 GB/s
- [ ] Zero-copy rendering via interop
- [ ] No memory leaks

### Quality:
- [ ] Clean error handling
- [ ] Comprehensive logging
- [ ] Well-documented code
- [ ] Unit tests passing

---

*This specification will be updated as implementation progresses and new requirements emerge.*