#pragma once

#include <GL/glew.h>  // Must be included before any other OpenGL headers
#include "cuda_utils.h"
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <vector>
#include <memory>

namespace SplatRender {
namespace CUDA {

// CUDA-OpenGL interoperability for zero-copy rendering
class CudaGLInterop {
public:
    CudaGLInterop();
    ~CudaGLInterop();
    
    // Register an OpenGL texture for CUDA access
    bool registerTexture(GLuint texture, int width, int height, GLenum target = GL_TEXTURE_2D);
    
    // Register an OpenGL buffer (VBO, PBO) for CUDA access
    bool registerBuffer(GLuint buffer, cudaGraphicsRegisterFlags flags = cudaGraphicsRegisterFlagsNone);
    
    // Unregister resources
    void unregisterTexture();
    void unregisterBuffer();
    void unregisterAll();
    
    // Map resources for CUDA access
    cudaSurfaceObject_t mapTextureForCuda();
    void* mapBufferForCuda(size_t* size = nullptr);
    
    // Unmap resources after CUDA operations
    void unmapTexture();
    void unmapBuffer();
    void unmapAll();
    
    // Direct rendering to texture
    void renderToTexture(const void* cuda_buffer, int width, int height, int channels = 4);
    
    // Get mapped array for custom kernel operations
    cudaArray_t getMappedArray() const { return cuda_array_; }
    
    // Check if resources are registered/mapped
    bool isTextureRegistered() const { return texture_resource_ != nullptr; }
    bool isBufferRegistered() const { return buffer_resource_ != nullptr; }
    bool isTextureMapped() const { return texture_mapped_; }
    bool isBufferMapped() const { return buffer_mapped_; }
    
    // Get texture dimensions
    int getWidth() const { return width_; }
    int getHeight() const { return height_; }
    
private:
    // OpenGL resources
    GLuint gl_texture_;
    GLuint gl_buffer_;
    GLenum texture_target_;
    
    // CUDA resources
    cudaGraphicsResource_t texture_resource_;
    cudaGraphicsResource_t buffer_resource_;
    cudaArray_t cuda_array_;
    cudaSurfaceObject_t surface_;
    void* buffer_ptr_;
    
    // State tracking
    bool texture_mapped_;
    bool buffer_mapped_;
    int width_;
    int height_;
    
    // Helper functions
    void createSurfaceObject();
    void destroySurfaceObject();
};

// Utility class for managing multiple interop resources
class CudaGLInteropManager {
public:
    CudaGLInteropManager();
    ~CudaGLInteropManager();
    
    // Create and register a new interop resource
    int createTextureInterop(GLuint texture, int width, int height);
    int createBufferInterop(GLuint buffer, cudaGraphicsRegisterFlags flags = cudaGraphicsRegisterFlagsNone);
    
    // Get interop resource by ID
    CudaGLInterop* getInterop(int id);
    
    // Remove interop resource
    void removeInterop(int id);
    
    // Map/unmap all resources
    void mapAll();
    void unmapAll();
    
    // Clear all resources
    void clear();
    
private:
    std::vector<std::unique_ptr<CudaGLInterop>> interops_;
    int next_id_;
};

// Helper functions for common interop operations
namespace InteropHelpers {
    // Copy CUDA buffer to OpenGL texture
    void copyToTexture(GLuint texture, const void* cuda_buffer, 
                       int width, int height, int channels = 4);
    
    // Copy OpenGL texture to CUDA buffer
    void copyFromTexture(void* cuda_buffer, GLuint texture, 
                        int width, int height, int channels = 4);
    
    // Render CUDA buffer directly to screen via OpenGL texture
    void renderToScreen(GLuint texture, const void* cuda_buffer, 
                       int width, int height);
    
    // Check if OpenGL context is valid for CUDA interop
    bool isGLContextValid();
    
    // Initialize CUDA-OpenGL interop (call once at startup)
    bool initializeInterop();
}

} // namespace CUDA
} // namespace SplatRender