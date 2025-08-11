# SplatRender Project Overview
*A comprehensive specification for building a high-performance 3D Gaussian Splatting renderer*

## Executive Summary

SplatRender is a from-scratch implementation of 3D Gaussian Splatting, designed to achieve real-time rendering performance through custom CUDA kernels while maintaining educational clarity. The project emphasizes understanding the mathematical foundations and engineering optimizations required for neural rendering techniques.

### Key Objectives
- **Performance**: Achieve 60+ FPS at 1080p resolution with 100K+ Gaussians
- **Educational**: Clear, modular codebase suitable for learning and research
- **Independence**: Built from mathematical foundations without dependency on existing implementations
- **Cross-platform Development**: Prototype on macOS, deploy on Linux with CUDA

## System Architecture

### High-Level Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                        Application Layer                      │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │   Engine    │  │    Camera    │  │   Input Handler  │   │
│  └─────────────┘  └──────────────┘  └──────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                        Rendering Layer                        │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ CPU Raster  │  │ CUDA Raster  │  │ OpenGL Display  │   │
│  └─────────────┘  └──────────────┘  └──────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                      Mathematics Layer                        │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  Gaussian   │  │ Matrix Ops   │  │ Spherical Harm. │   │
│  └─────────────┘  └──────────────┘  └──────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                          I/O Layer                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    PLY Loader                        │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Module Breakdown

#### Core Components (Cross-platform)
- **Engine**: Main application loop, initialization, resource management
- **Camera**: View matrix generation, FPS-style controls, projection matrices
- **Input**: Keyboard/mouse handling via GLFW, camera movement

#### Mathematics (Pure C++)
- **Gaussian**: 3D/2D Gaussian data structures, covariance computation
- **Matrix Operations**: Projection math, Jacobian computation, transformations
- **Spherical Harmonics**: View-dependent color evaluation (up to degree 3)

#### I/O Operations
- **PLY Loader**: Parse Gaussian parameters from PLY files
  - Position, scale, rotation quaternions
  - Spherical harmonic coefficients
  - Color and opacity

#### Rendering Pipeline
- **CPU Rasterizer**: Reference implementation for validation
- **OpenGL Display**: Texture display, shader management
- **CUDA Rasterizer** (Linux only): High-performance GPU implementation

## Data Structures

### 3D Gaussian Representation
```cpp
struct Gaussian3D {
    // Spatial properties
    glm::vec3 position;      // World space position
    glm::vec3 scale;         // Scale factors along axes
    glm::quat rotation;      // Orientation quaternion
    
    // Appearance properties
    glm::vec3 color;         // Base RGB color
    float opacity;           // Alpha value
    
    // View-dependent appearance
    std::array<float, 15> sh_coeffs;  // SH coefficients (RGB, degree 3)
};
```

### 2D Gaussian (Screen Space)
```cpp
struct Gaussian2D {
    // Screen space properties
    glm::vec2 center;        // Pixel coordinates
    glm::mat2 cov_2d;        // 2D covariance matrix
    
    // Rendering properties
    glm::vec3 color;         // View-dependent RGB
    float alpha;             // Opacity after projection
    float depth;             // Z-depth for sorting
    
    // Optimization data
    uint32_t tile_id;        // Which tile(s) this affects
    float radius;            // Bounding radius in pixels
};
```

### Tile Data Structure
```cpp
struct TileData {
    static constexpr int TILE_SIZE = 16;
    
    uint32_t gaussian_count;
    uint32_t* gaussian_indices;  // Indices into Gaussian array
    float* depths;               // For depth sorting
};
```

## Mathematical Foundations

### 3D Gaussian Covariance
The 3D covariance matrix is constructed from scale and rotation:
```
Σ = R · S · S^T · R^T

Where:
- R = rotation matrix from quaternion
- S = diagonal scale matrix
```

### 3D to 2D Projection
Project 3D Gaussian to screen space using view-projection Jacobian:
```
Σ_2D = J · W · Σ_3D · W^T · J^T

Where:
- J = Jacobian of perspective projection
- W = World-to-view transformation matrix
```

### Alpha Blending
Front-to-back alpha compositing:
```
C = Σ(α_i · c_i · Π(1 - α_j)) for all j < i

Where:
- α_i = opacity of Gaussian i
- c_i = color of Gaussian i
- Product term ensures occlusion
```

### Spherical Harmonics
View-dependent color evaluation:
```
c(v) = c_0 + Σ(c_l,m · Y_l,m(v))

Where:
- c_0 = base color (DC component)
- c_l,m = SH coefficients
- Y_l,m = SH basis functions
- v = view direction
```

## Development Roadmap

### Phase 1: Foundation
- [x] Project structure setup
- [ ] CMake configuration with platform detection
- [ ] Basic OpenGL window creation
- [ ] Core data structures

### Phase 2: Mathematics
- [ ] 3D to 2D projection implementation
- [ ] Covariance matrix computation
- [ ] Spherical harmonics evaluation
- [ ] CPU reference rasterizer

### Phase 3: Display System
- [ ] OpenGL texture display
- [ ] Shader implementation
- [ ] Camera controls
- [ ] Basic rendering loop

### Phase 4: CUDA Foundation
- [ ] CUDA memory management
- [ ] Host-device transfers
- [ ] CUDA-OpenGL interop
- [ ] Basic projection kernel

### Phase 5: CUDA Rasterization
- [ ] Tile assignment algorithm
- [ ] Per-tile depth sorting
- [ ] Parallel rasterization kernel
- [ ] Alpha blending implementation

### Phase 6: Optimization 
- [ ] Shared memory optimization
- [ ] Memory coalescing
- [ ] Warp-level primitives
- [ ] Occupancy tuning

### Phase 7: Polish
- [ ] Multi-resolution support
- [ ] Performance monitoring
- [ ] Error handling
- [ ] Documentation

## Platform Strategy

### macOS Development (Current Phase)
- **Focus**: Core algorithms, mathematics, CPU implementation
- **Tools**: Xcode/CLion, OpenGL 4.1, GLFW
- **Components**: Everything except CUDA kernels

### Linux Deployment (Production Phase)
- **Focus**: CUDA optimization, performance tuning
- **Tools**: CUDA Toolkit 11.8+, NVIDIA Nsight
- **Requirements**: NVIDIA GPU with CC 7.0+

### Build Configuration
```cmake
if(APPLE)
    # macOS: CPU implementation + OpenGL
    set(USE_CUDA OFF)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
elseif(UNIX)
    # Linux: Full CUDA implementation
    find_package(CUDAToolkit REQUIRED)
    set(USE_CUDA ON)
    enable_language(CUDA)
endif()
```

## Performance Specifications

### Target Metrics
- **Frame Rate**: 60+ FPS at 1920x1080
- **Gaussian Count**: 100,000+ simultaneous Gaussians
- **Memory Usage**: <2GB VRAM for typical scenes
- **Load Time**: <100ms for average PLY files

### Optimization Strategies
1. **Tile-based Processing**: 16x16 pixel tiles for cache efficiency
2. **Early Termination**: Skip Gaussians with negligible contribution
3. **Memory Patterns**: Coalesced access, shared memory usage
4. **Parallel Sorting**: GPU-accelerated per-tile depth sorting

## Technical Requirements

### Hardware Requirements
- **GPU**: NVIDIA RTX 20-series or newer (Linux)
- **VRAM**: 8GB recommended
- **RAM**: 16GB system memory
- **CPU**: Modern multi-core processor

### Software Dependencies
- **C++ Compiler**: GCC 9+ or Clang 10+
- **CUDA**: Toolkit 11.8+ (Linux only)
- **CMake**: 3.18 or newer
- **OpenGL**: 4.1+ core profile
- **Libraries**:
  - GLFW 3.3+ (window management)
  - GLEW 2.1+ (OpenGL extensions)
  - GLM 0.9.9+ (mathematics)
  - Eigen 3.4+ (linear algebra)

## File Format Specifications

### Input PLY Format
```
ply
format binary_little_endian 1.0
element vertex N
property float x
property float y
property float z
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float f_rest_0
...
property float f_rest_44
end_header
[binary data]
```

## Testing Strategy

### Unit Tests
- Mathematical operations (projection, covariance)
- Data structure serialization
- Memory management

### Integration Tests
- PLY loading correctness
- CPU vs GPU result comparison
- Rendering pipeline validation

### Performance Tests
- Frame time measurements
- Memory usage profiling
- Kernel execution timing

## Success Criteria

### Functional Requirements
- [x] Load and parse PLY files
- [ ] Correct 3D to 2D projection
- [ ] Proper alpha blending
- [ ] Real-time camera controls
- [ ] Stable 60+ FPS performance

### Quality Requirements
- [ ] Clean, modular architecture
- [ ] Comprehensive error handling
- [ ] Memory leak free
- [ ] Well-documented code

### Performance Requirements
- [ ] <16.67ms frame time (60 FPS)
- [ ] <2GB VRAM usage
- [ ] <100ms load time
- [ ] Linear scaling with Gaussian count

## Risk Mitigation

### Technical Risks
1. **CUDA Complexity**: Start with CPU implementation for validation
2. **Memory Management**: Use RAII, smart pointers, careful profiling
3. **Performance Targets**: Profile early, optimize based on data
4. **Platform Differences**: Abstract platform-specific code

### Mitigation Strategies
- Incremental development with working milestones
- Extensive testing at each phase
- Performance profiling from early stages
- Clean abstraction layers

## Future Extensions

### Potential Enhancements
- Dynamic Gaussian adaptation
- Temporal stability improvements
- Multi-GPU support
- WebGPU implementation
- VR/AR rendering support

### Research Opportunities
- Alternative sorting algorithms
- Hierarchical culling
- Compression techniques
- Neural network integration

---

*This specification serves as the foundational blueprint for SplatRender development. It will be updated as the project evolves and new insights are gained.*