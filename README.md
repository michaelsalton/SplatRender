# SplatRender

A high-performance 3D Gaussian Splatting renderer implemented from scratch in C++/CUDA. This project uses modern neural rendering techniques and implements the complete rasterization pipeline without dependencies on existing implementations.

## Project Goals

- **Educational**: Understand 3D Gaussian Splatting at the implementation level
- **Performance**: Achieve real-time rendering (60+ FPS) through custom CUDA kernels
- **Independence**: Built from mathematical foundations, not derived from existing code
- **Research-Ready**: Clean codebase suitable for algorithmic extensions

## Features

- Custom CUDA rasterization kernels
- Tile-based parallel processing 
- Real-time camera controls
- PLY model loading
- Spherical harmonics evaluation
- Alpha blending with depth sorting

## Algorithm Details

### Core Implementation

This renderer implements 3D Gaussian Splatting from the mathematical foundations:

1. **3D to 2D Projection**: Transform oriented 3D Gaussians to screen space
2. **Covariance Computation**: Project 3D covariance matrices using view transformation Jacobian
3. **Tile-based Rasterization**: Divide screen into 16×16 tiles for parallel processing
4. **Depth Sorting**: Order Gaussians front-to-back within each tile
5. **Alpha Blending**: Accumulate colors using differentiable alpha composition
6. **Spherical Harmonics**: Evaluate view-dependent appearance up to degree 3

### CUDA Optimization Strategies

- **Memory Coalescing**: Optimize global memory access patterns
- **Shared Memory**: Cache frequently accessed data within thread blocks
- **Occupancy Optimization**: Balance threads per block vs register usage
- **Warp-level Primitives**: Use cooperative thread operations

## Mathematical Foundations

### 3D Gaussian covariance matrix
Σ = R · S · S^T · R^T

### 2D projection via Jacobian  
Σ_2D = J · Σ_3D · J^T

### Alpha blending equation
C = Σ(α_i · c_i · ∏(1 - α_j)) for j < i

## Requirements

### Hardware
- NVIDIA GPU with Compute Capability 7.0+ (RTX 20-series or newer)
- 8GB+ VRAM recommended
- 16GB+ system RAM

### Software
- Linux
- CUDA Toolkit 11.8+
- GCC 9+ or Clang 10+
- CMake 3.18+

## References

- 3D Gaussian Splatting for Real-Time Radiance Field Rendering (Kerbl et al., SIGGRAPH 2023)
- CUDA Programming Guide (NVIDIA)
- Real-Time Rendering (Akenine-Möller et al.)

## Acknowledgments

- Original 3DGS research team at Inria for the foundational paper
- NVIDIA for CUDA toolkit and excellent documentation
- Graphics programming community for invaluable resources

**Note**: This is an independent implementation created for educational purposes. It is not affiliated with or endorsed by the original 3D Gaussian Splatting research team.

Built with ❤️ for learning and advancing real-time neural rendering
