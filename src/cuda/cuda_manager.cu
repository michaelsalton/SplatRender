#include "cuda_manager.h"
#include <iostream>
#include <iomanip>
#include <algorithm>

namespace SplatRender {
namespace CUDA {

bool CudaManager::initialize(int deviceId) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (initialized_) {
        std::cout << "CUDA Manager already initialized" << std::endl;
        return true;
    }
    
    // Check for CUDA devices
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess || deviceCount == 0) {
        std::cerr << "No CUDA devices found or CUDA not available" << std::endl;
        return false;
    }
    
    // Select device
    if (deviceId < 0) {
        // Auto-select best device
        device_id_ = selectBestDevice();
        std::cout << "Auto-selected CUDA device " << device_id_ << std::endl;
    } else if (deviceId >= deviceCount) {
        std::cerr << "Invalid device ID: " << deviceId << " (only " << deviceCount << " devices available)" << std::endl;
        return false;
    } else {
        device_id_ = deviceId;
    }
    
    // Set the device
    CUDA_CHECK(cudaSetDevice(device_id_));
    
    // Get device properties
    CUDA_CHECK(cudaGetDeviceProperties(&device_props_, device_id_));
    
    // Check compute capability (minimum 7.0 for RTX series)
    if (device_props_.major < 7) {
        std::cerr << "Warning: Compute capability " << device_props_.major << "." << device_props_.minor 
                  << " is below recommended 7.0" << std::endl;
    }
    
    // Create default stream
    CUDA_CHECK(cudaStreamCreate(&default_stream_));
    
    // Print device info
    std::cout << "CUDA Initialized Successfully" << std::endl;
    std::cout << "  Device: " << device_props_.name << std::endl;
    std::cout << "  Compute Capability: " << device_props_.major << "." << device_props_.minor << std::endl;
    std::cout << "  Total Memory: " << (device_props_.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  Max Threads per Block: " << device_props_.maxThreadsPerBlock << std::endl;
    std::cout << "  Multiprocessors: " << device_props_.multiProcessorCount << std::endl;
    std::cout << "  Warp Size: " << device_props_.warpSize << std::endl;
    
    initialized_ = true;
    return true;
}

void CudaManager::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!initialized_) {
        return;
    }
    
    // Synchronize device
    cudaDeviceSynchronize();
    
    // Destroy all managed streams
    for (auto stream : managed_streams_) {
        cudaStreamDestroy(stream);
    }
    managed_streams_.clear();
    
    // Destroy default stream
    if (default_stream_) {
        cudaStreamDestroy(default_stream_);
        default_stream_ = nullptr;
    }
    
    // Reset device
    cudaDeviceReset();
    
    initialized_ = false;
    device_id_ = -1;
    
    std::cout << "CUDA Manager shutdown complete" << std::endl;
}

size_t CudaManager::getAvailableMemory() const {
    if (!initialized_) {
        return 0;
    }
    
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    
    cudaError_t error = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (error != cudaSuccess) {
        std::cerr << "Failed to get memory info: " << cudaGetErrorString(error) << std::endl;
        return 0;
    }
    
    return free_bytes;
}

cudaStream_t CudaManager::createStream() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!initialized_) {
        throw std::runtime_error("CUDA Manager not initialized");
    }
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    managed_streams_.push_back(stream);
    
    return stream;
}

void CudaManager::destroyStream(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!initialized_ || stream == default_stream_) {
        return;
    }
    
    // Find and remove from managed streams
    auto it = std::find(managed_streams_.begin(), managed_streams_.end(), stream);
    if (it != managed_streams_.end()) {
        cudaStreamDestroy(stream);
        managed_streams_.erase(it);
    }
}

void CudaManager::synchronizeStream(cudaStream_t stream) {
    if (!initialized_) {
        return;
    }
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void CudaManager::synchronizeDevice() {
    if (!initialized_) {
        return;
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
}

void CudaManager::printMemoryInfo() const {
    if (!initialized_) {
        std::cout << "CUDA Manager not initialized" << std::endl;
        return;
    }
    
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    
    cudaError_t error = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (error != cudaSuccess) {
        std::cerr << "Failed to get memory info: " << cudaGetErrorString(error) << std::endl;
        return;
    }
    
    size_t used_bytes = total_bytes - free_bytes;
    
    std::cout << "CUDA Memory Info:" << std::endl;
    std::cout << "  Total:     " << std::setw(8) << (total_bytes / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  Used:      " << std::setw(8) << (used_bytes / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  Available: " << std::setw(8) << (free_bytes / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  Usage:     " << std::setw(8) << std::fixed << std::setprecision(1) 
              << (100.0 * used_bytes / total_bytes) << " %" << std::endl;
}

} // namespace CUDA
} // namespace SplatRender