#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace SplatRender {
namespace CUDA {

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::stringstream ss; \
            ss << "CUDA error at " << __FILE__ << ":" << __LINE__ \
               << " - " << cudaGetErrorString(error); \
            throw std::runtime_error(ss.str()); \
        } \
    } while(0)

// CUDA kernel launch checking
#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t error = cudaGetLastError(); \
        if (error != cudaSuccess) { \
            std::stringstream ss; \
            ss << "CUDA kernel launch failed at " << __FILE__ << ":" << __LINE__ \
               << " - " << cudaGetErrorString(error); \
            throw std::runtime_error(ss.str()); \
        } \
        error = cudaDeviceSynchronize(); \
        if (error != cudaSuccess) { \
            std::stringstream ss; \
            ss << "CUDA kernel execution failed at " << __FILE__ << ":" << __LINE__ \
               << " - " << cudaGetErrorString(error); \
            throw std::runtime_error(ss.str()); \
        } \
    } while(0)

// CUDA safe call for debugging (only in debug mode)
#ifdef DEBUG
    #define CUDA_SAFE_CALL(call) CUDA_CHECK(call)
#else
    #define CUDA_SAFE_CALL(call) call
#endif

// Helper function to get CUDA device properties
inline void printDeviceInfo(int device = 0) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    std::cout << "CUDA Device Information:" << std::endl;
    std::cout << "  Name: " << prop.name << std::endl;
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Total Global Memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  Shared Memory per Block: " << (prop.sharedMemPerBlock / 1024) << " KB" << std::endl;
    std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "  Max Grid Dimensions: [" << prop.maxGridSize[0] << ", " 
              << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << "]" << std::endl;
    std::cout << "  Warp Size: " << prop.warpSize << std::endl;
    std::cout << "  Memory Clock Rate: " << (prop.memoryClockRate / 1000) << " MHz" << std::endl;
    std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
}

// Helper to calculate optimal block size
inline void getOptimalBlockSize(int& blockSize, int& gridSize, int workSize, int maxBlockSize = 256) {
    blockSize = (workSize < maxBlockSize) ? workSize : maxBlockSize;
    gridSize = (workSize + blockSize - 1) / blockSize;
}

// Helper to check if CUDA is available
inline bool isCudaAvailable() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    return (error == cudaSuccess && deviceCount > 0);
}

// Helper to select best CUDA device
inline int selectBestDevice() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        throw std::runtime_error("No CUDA devices found");
    }
    
    int bestDevice = 0;
    size_t maxMemory = 0;
    
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        
        if (prop.totalGlobalMem > maxMemory) {
            maxMemory = prop.totalGlobalMem;
            bestDevice = i;
        }
    }
    
    return bestDevice;
}

// Memory alignment helper
template<typename T>
inline size_t getAlignedSize(size_t count, size_t alignment = 256) {
    size_t bytes = count * sizeof(T);
    return ((bytes + alignment - 1) / alignment) * alignment;
}

// CUDA timer for profiling
class CudaTimer {
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    
    void start() {
        CUDA_CHECK(cudaEventRecord(start_, 0));
    }
    
    void stop() {
        CUDA_CHECK(cudaEventRecord(stop_, 0));
        CUDA_CHECK(cudaEventSynchronize(stop_));
    }
    
    float getElapsedMs() {
        float elapsed = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, start_, stop_));
        return elapsed;
    }
    
private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};

} // namespace CUDA
} // namespace SplatRender