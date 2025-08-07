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

**3D to 2D Projection**: Transform oriented 3D Gaussians to screen space
**Covariance Computation**: Project 3D covariance matrices using view transformation Jacobian
**Tile-based Rasterization**: Divide screen into 16×16 tiles for parallel processing
**Depth Sorting**: Order Gaussians front-to-back within each tile
**Alpha Blending**: Accumulate colors using differentiable alpha composition
**Spherical Harmonics**: Evaluate view-dependent appearance up to degree 3

### CUDA Optimization Strategies

**Memory Coalescing**: Optimize global memory access patterns
**Shared Memory**: Cache frequently accessed data within thread blocks
**Occupancy Optimization**: Balance threads per block vs register usage
**Warp-level Primitives**: Use cooperative thread operations

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
