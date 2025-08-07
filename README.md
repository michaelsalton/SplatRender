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

