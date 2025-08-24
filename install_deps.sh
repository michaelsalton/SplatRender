#!/bin/bash

# Install script for SplatRender dependencies on Ubuntu 24.04

echo "Installing dependencies for SplatRender..."
echo "This will install CUDA toolkit and required graphics libraries"
echo ""

# Update package lists
echo "Updating package lists..."
sudo apt update

# Install CUDA toolkit
echo "Installing CUDA toolkit..."
sudo apt install -y nvidia-cuda-toolkit

# Install build essentials and CMake
echo "Installing build tools..."
sudo apt install -y build-essential cmake git

# Install OpenGL, GLFW, and GLEW
echo "Installing graphics libraries..."
sudo apt install -y libgl1-mesa-dev libglu1-mesa-dev
sudo apt install -y libglfw3-dev
sudo apt install -y libglew-dev

# Install Eigen3 and GLM
echo "Installing math libraries..."
sudo apt install -y libeigen3-dev
sudo apt install -y libglm-dev

# Install X11 development libraries
echo "Installing X11 development libraries..."
sudo apt install -y libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev

echo ""
echo "Installation complete!"
echo "You can now build the project with:"
echo "  mkdir build && cd build"
echo "  cmake .."
echo "  make -j$(nproc)"