# SplatRender Requirements

## System Requirements

### Operating System
- **Development**: macOS 11.0+ (Big Sur or later)
- **Production**: Ubuntu 20.04+ or other Linux distributions with NVIDIA GPU support

### Hardware Requirements
- **Minimum**:
  - CPU: Modern multi-core processor (Intel Core i5 or equivalent)
  - RAM: 8GB system memory
  - GPU: Any GPU with OpenGL 4.1+ support (for development)
  
- **Recommended**:
  - CPU: Intel Core i7/i9 or AMD Ryzen 7/9
  - RAM: 16GB+ system memory
  - GPU: NVIDIA RTX 2070 or better (for CUDA implementation)
  - VRAM: 8GB+ (for handling large Gaussian splat scenes)

### GPU Requirements (Linux/CUDA)
- NVIDIA GPU with Compute Capability 7.0+ (RTX 20-series or newer)
- CUDA Toolkit 11.8 or later
- NVIDIA Driver 520.61.05 or later

## Software Dependencies

### Core Build Tools
- **CMake**: 3.18 or later
- **C++ Compiler**:
  - macOS: Xcode Command Line Tools with Apple Clang 12.0+
  - Linux: GCC 9+ or Clang 10+

### Required Libraries

#### macOS Installation (via Homebrew)
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required dependencies
brew install cmake
brew install glm
brew install glew
brew install glfw
brew install eigen
```

#### Linux Installation (Ubuntu/Debian)
```bash
# Update package list
sudo apt update

# Install build tools
sudo apt install build-essential cmake git

# Install graphics libraries
sudo apt install libgl1-mesa-dev libglu1-mesa-dev
sudo apt install libglew-dev
sudo apt install libglfw3-dev

# Install math libraries
sudo apt install libglm-dev
sudo apt install libeigen3-dev

# Install CUDA (for GPU acceleration)
# Follow NVIDIA's official installation guide for your distribution
# https://developer.nvidia.com/cuda-downloads
```

### Library Versions
| Library | Minimum Version | Purpose |
|---------|----------------|----------|
| GLM | 0.9.9 | Vector/matrix mathematics |
| GLEW | 2.1.0 | OpenGL extension loading |
| GLFW | 3.3 | Window management and input |
| Eigen | 3.4 | Linear algebra operations |
| OpenGL | 4.1 | Graphics rendering |
| CUDA | 11.8 | GPU acceleration (Linux only) |

## Development Environment Setup

### IDE Requirements
- **Recommended**: Visual Studio Code with extensions:
  - C/C++ (Microsoft)
  - CMake Tools (Microsoft)
  - CUDA (optional, for Linux development)
- **Alternative**: CLion, Xcode (macOS), or any C++-capable IDE

### VS Code Extensions
```bash
# Install via VS Code command palette
code --install-extension ms-vscode.cpptools
code --install-extension ms-vscode.cmake-tools
code --install-extension nvidia.nsight-vscode-edition  # For CUDA development
```

## Python Dependencies (Optional, for data processing)
```bash
# For PLY file manipulation and testing
pip install numpy
pip install plyfile
pip install matplotlib
```

## Build Instructions

### macOS Build
```bash
# Clone the repository
git clone <repository-url>
cd SplatRender

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(sysctl -n hw.ncpu)
```

### Linux Build (with CUDA)
```bash
# Clone the repository
git clone <repository-url>
cd SplatRender

# Create build directory
mkdir build && cd build

# Configure with CMake (CUDA will be auto-detected)
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)
```

## Verification

### Check Dependencies (macOS)
```bash
# Check installed packages
brew list | grep -E "glm|glew|glfw|eigen"

# Check library locations
ls -la /opt/homebrew/include/GL/
ls -la /opt/homebrew/include/glm/
```

### Check Dependencies (Linux)
```bash
# Check OpenGL
glxinfo | grep "OpenGL version"

# Check CUDA (if installed)
nvidia-smi
nvcc --version

# Check libraries
pkg-config --modversion glfw3
pkg-config --modversion glew
```

## Troubleshooting

### Common Issues

1. **GLM headers not found**
   - Ensure GLM is installed: `brew install glm` (macOS) or `sudo apt install libglm-dev` (Linux)
   - Check include path in CMakeLists.txt

2. **GLEW linking errors**
   - Verify GLEW installation: `brew list glew` (macOS)
   - Check library path: `/opt/homebrew/lib/libGLEW.*` (macOS)

3. **CUDA not detected (Linux)**
   - Ensure CUDA toolkit is in PATH: `export PATH=/usr/local/cuda/bin:$PATH`
   - Add to `.bashrc` for persistence

4. **CMake configuration fails**
   - Delete build directory and reconfigure: `rm -rf build && mkdir build`
   - Check CMake version: `cmake --version`

### Performance Considerations
- Debug builds will be significantly slower than Release builds
- Enable compiler optimizations: `-O3` for best performance
- CUDA implementation requires Linux for full performance

## Optional Tools

### Profiling and Debugging
- **NVIDIA Nsight Compute**: CUDA kernel profiling
- **NVIDIA Nsight Graphics**: Graphics debugging
- **Valgrind**: Memory leak detection (Linux)
- **Instruments**: Performance profiling (macOS)

### Data Processing
- **CloudCompare**: View and convert point cloud data
- **MeshLab**: 3D mesh processing
- **Blender**: 3D modeling and conversion

## License Compliance
This project uses the following open-source libraries:
- GLM: MIT License
- GLEW: Modified BSD License
- GLFW: zlib License
- Eigen: MPL2 License

Ensure compliance with all library licenses in your usage.