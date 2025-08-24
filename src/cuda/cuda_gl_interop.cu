#include "cuda_gl_interop.h"
#include <iostream>
#include <cstring>

namespace SplatRender {
namespace CUDA {

// Kernel for copying buffer to surface
__global__ void copyToSurfaceKernel(cudaSurfaceObject_t surface, const float* buffer, 
                                    int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = (y * width + x) * channels;
    
    if (channels == 4) {
        float4 pixel = make_float4(buffer[idx], buffer[idx + 1], buffer[idx + 2], buffer[idx + 3]);
        surf2Dwrite(pixel, surface, x * sizeof(float4), y);
    } else if (channels == 3) {
        float4 pixel = make_float4(buffer[idx], buffer[idx + 1], buffer[idx + 2], 1.0f);
        surf2Dwrite(pixel, surface, x * sizeof(float4), y);
    }
}

CudaGLInterop::CudaGLInterop() 
    : gl_texture_(0), gl_buffer_(0), texture_target_(GL_TEXTURE_2D),
      texture_resource_(nullptr), buffer_resource_(nullptr),
      cuda_array_(nullptr), surface_(0), buffer_ptr_(nullptr),
      texture_mapped_(false), buffer_mapped_(false),
      width_(0), height_(0) {
}

CudaGLInterop::~CudaGLInterop() {
    unregisterAll();
}

bool CudaGLInterop::registerTexture(GLuint texture, int width, int height, GLenum target) {
    if (texture_resource_) {
        unregisterTexture();
    }
    
    gl_texture_ = texture;
    texture_target_ = target;
    width_ = width;
    height_ = height;
    
    // Register OpenGL texture with CUDA
    cudaError_t error = cudaGraphicsGLRegisterImage(&texture_resource_, gl_texture_, 
                                                    texture_target_, 
                                                    cudaGraphicsRegisterFlagsWriteDiscard);
    
    if (error != cudaSuccess) {
        std::cerr << "Failed to register OpenGL texture with CUDA: " 
                  << cudaGetErrorString(error) << std::endl;
        texture_resource_ = nullptr;
        return false;
    }
    
    return true;
}

bool CudaGLInterop::registerBuffer(GLuint buffer, cudaGraphicsRegisterFlags flags) {
    if (buffer_resource_) {
        unregisterBuffer();
    }
    
    gl_buffer_ = buffer;
    
    // Register OpenGL buffer with CUDA
    cudaError_t error = cudaGraphicsGLRegisterBuffer(&buffer_resource_, gl_buffer_, flags);
    
    if (error != cudaSuccess) {
        std::cerr << "Failed to register OpenGL buffer with CUDA: " 
                  << cudaGetErrorString(error) << std::endl;
        buffer_resource_ = nullptr;
        return false;
    }
    
    return true;
}

void CudaGLInterop::unregisterTexture() {
    if (texture_mapped_) {
        unmapTexture();
    }
    
    if (texture_resource_) {
        cudaGraphicsUnregisterResource(texture_resource_);
        texture_resource_ = nullptr;
    }
    
    gl_texture_ = 0;
    width_ = 0;
    height_ = 0;
}

void CudaGLInterop::unregisterBuffer() {
    if (buffer_mapped_) {
        unmapBuffer();
    }
    
    if (buffer_resource_) {
        cudaGraphicsUnregisterResource(buffer_resource_);
        buffer_resource_ = nullptr;
    }
    
    gl_buffer_ = 0;
}

void CudaGLInterop::unregisterAll() {
    unregisterTexture();
    unregisterBuffer();
}

cudaSurfaceObject_t CudaGLInterop::mapTextureForCuda() {
    if (!texture_resource_ || texture_mapped_) {
        return surface_;
    }
    
    // Map the graphics resource
    CUDA_CHECK(cudaGraphicsMapResources(1, &texture_resource_, 0));
    
    // Get CUDA array from the mapped resource
    CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&cuda_array_, texture_resource_, 0, 0));
    
    // Create surface object
    createSurfaceObject();
    
    texture_mapped_ = true;
    return surface_;
}

void* CudaGLInterop::mapBufferForCuda(size_t* size) {
    if (!buffer_resource_ || buffer_mapped_) {
        return buffer_ptr_;
    }
    
    // Map the graphics resource
    CUDA_CHECK(cudaGraphicsMapResources(1, &buffer_resource_, 0));
    
    // Get device pointer
    size_t mapped_size = 0;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(&buffer_ptr_, &mapped_size, buffer_resource_));
    
    if (size) {
        *size = mapped_size;
    }
    
    buffer_mapped_ = true;
    return buffer_ptr_;
}

void CudaGLInterop::unmapTexture() {
    if (!texture_mapped_) {
        return;
    }
    
    // Destroy surface object
    destroySurfaceObject();
    
    // Unmap the resource
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &texture_resource_, 0));
    
    cuda_array_ = nullptr;
    texture_mapped_ = false;
}

void CudaGLInterop::unmapBuffer() {
    if (!buffer_mapped_) {
        return;
    }
    
    // Unmap the resource
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &buffer_resource_, 0));
    
    buffer_ptr_ = nullptr;
    buffer_mapped_ = false;
}

void CudaGLInterop::unmapAll() {
    unmapTexture();
    unmapBuffer();
}

void CudaGLInterop::renderToTexture(const void* cuda_buffer, int width, int height, int channels) {
    if (!texture_resource_) {
        std::cerr << "No texture registered for CUDA interop" << std::endl;
        return;
    }
    
    // Map texture if not already mapped
    bool was_mapped = texture_mapped_;
    if (!was_mapped) {
        mapTextureForCuda();
    }
    
    // Launch kernel to copy buffer to surface
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    copyToSurfaceKernel<<<gridSize, blockSize>>>(surface_, 
                                                 static_cast<const float*>(cuda_buffer),
                                                 width, height, channels);
    CUDA_CHECK_KERNEL();
    
    // Unmap if we mapped it
    if (!was_mapped) {
        unmapTexture();
    }
}

void CudaGLInterop::createSurfaceObject() {
    if (surface_ != 0 || !cuda_array_) {
        return;
    }
    
    // Create surface object from array
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuda_array_;
    
    CUDA_CHECK(cudaCreateSurfaceObject(&surface_, &resDesc));
}

void CudaGLInterop::destroySurfaceObject() {
    if (surface_ != 0) {
        cudaDestroySurfaceObject(surface_);
        surface_ = 0;
    }
}

// CudaGLInteropManager implementation
CudaGLInteropManager::CudaGLInteropManager() : next_id_(0) {
}

CudaGLInteropManager::~CudaGLInteropManager() {
    clear();
}

int CudaGLInteropManager::createTextureInterop(GLuint texture, int width, int height) {
    auto interop = std::make_unique<CudaGLInterop>();
    
    if (!interop->registerTexture(texture, width, height)) {
        return -1;
    }
    
    int id = next_id_++;
    if (id >= static_cast<int>(interops_.size())) {
        interops_.resize(id + 1);
    }
    interops_[id] = std::move(interop);
    
    return id;
}

int CudaGLInteropManager::createBufferInterop(GLuint buffer, cudaGraphicsRegisterFlags flags) {
    auto interop = std::make_unique<CudaGLInterop>();
    
    if (!interop->registerBuffer(buffer, flags)) {
        return -1;
    }
    
    int id = next_id_++;
    if (id >= static_cast<int>(interops_.size())) {
        interops_.resize(id + 1);
    }
    interops_[id] = std::move(interop);
    
    return id;
}

CudaGLInterop* CudaGLInteropManager::getInterop(int id) {
    if (id < 0 || id >= static_cast<int>(interops_.size()) || !interops_[id]) {
        return nullptr;
    }
    return interops_[id].get();
}

void CudaGLInteropManager::removeInterop(int id) {
    if (id >= 0 && id < static_cast<int>(interops_.size())) {
        interops_[id].reset();
    }
}

void CudaGLInteropManager::mapAll() {
    for (auto& interop : interops_) {
        if (interop) {
            if (interop->isTextureRegistered()) {
                interop->mapTextureForCuda();
            }
            if (interop->isBufferRegistered()) {
                interop->mapBufferForCuda();
            }
        }
    }
}

void CudaGLInteropManager::unmapAll() {
    for (auto& interop : interops_) {
        if (interop) {
            interop->unmapAll();
        }
    }
}

void CudaGLInteropManager::clear() {
    interops_.clear();
    next_id_ = 0;
}

// InteropHelpers implementation
namespace InteropHelpers {

void copyToTexture(GLuint texture, const void* cuda_buffer, int width, int height, int channels) {
    CudaGLInterop interop;
    if (interop.registerTexture(texture, width, height)) {
        interop.renderToTexture(cuda_buffer, width, height, channels);
    }
}

void copyFromTexture(void* cuda_buffer, GLuint texture, int width, int height, int channels) {
    // Implementation would require a kernel to read from surface
    // This is a placeholder for now
    std::cerr << "copyFromTexture not yet implemented" << std::endl;
}

void renderToScreen(GLuint texture, const void* cuda_buffer, int width, int height) {
    copyToTexture(texture, cuda_buffer, width, height, 4);
}

bool isGLContextValid() {
    // Check if we have a valid OpenGL context
    GLint current_context = 0;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &current_context);
    return (glGetError() == GL_NO_ERROR);
}

bool initializeInterop() {
    if (!isGLContextValid()) {
        std::cerr << "No valid OpenGL context for CUDA interop" << std::endl;
        return false;
    }
    
    // Set OpenGL device for CUDA
    cudaError_t error = cudaGLSetGLDevice(0);
    if (error != cudaSuccess) {
        std::cerr << "Failed to set OpenGL device for CUDA: " 
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    std::cout << "CUDA-OpenGL interop initialized successfully" << std::endl;
    return true;
}

} // namespace InteropHelpers

} // namespace CUDA
} // namespace SplatRender