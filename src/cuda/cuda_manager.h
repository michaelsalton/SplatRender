#pragma once

#include "cuda_utils.h"
#include <vector>
#include <memory>
#include <mutex>

namespace SplatRender {
namespace CUDA {

class CudaManager {
public:
    // Singleton access
    static CudaManager& getInstance() {
        static CudaManager instance;
        return instance;
    }
    
    // Delete copy/move constructors for singleton
    CudaManager(const CudaManager&) = delete;
    CudaManager& operator=(const CudaManager&) = delete;
    CudaManager(CudaManager&&) = delete;
    CudaManager& operator=(CudaManager&&) = delete;
    
    // Initialize CUDA with optional device selection
    // deviceId = -1 means auto-select best device
    bool initialize(int deviceId = -1);
    
    // Shutdown and cleanup
    void shutdown();
    
    // Check if initialized
    bool isInitialized() const { return initialized_; }
    
    // Device properties
    const cudaDeviceProp& getDeviceProperties() const { return device_props_; }
    int getDeviceId() const { return device_id_; }
    size_t getAvailableMemory() const;
    size_t getTotalMemory() const { return device_props_.totalGlobalMem; }
    int getComputeCapabilityMajor() const { return device_props_.major; }
    int getComputeCapabilityMinor() const { return device_props_.minor; }
    
    // Stream management
    cudaStream_t getDefaultStream() const { return default_stream_; }
    cudaStream_t createStream();
    void destroyStream(cudaStream_t stream);
    void synchronizeStream(cudaStream_t stream);
    void synchronizeDevice();
    
    // Memory info
    void printMemoryInfo() const;
    
    // Performance hints
    size_t getOptimalThreadsPerBlock() const { return 256; }
    size_t getMaxThreadsPerBlock() const { return device_props_.maxThreadsPerBlock; }
    size_t getWarpSize() const { return device_props_.warpSize; }
    
private:
    CudaManager() : initialized_(false), device_id_(-1), default_stream_(nullptr) {}
    ~CudaManager() { shutdown(); }
    
    bool initialized_;
    int device_id_;
    cudaDeviceProp device_props_;
    cudaStream_t default_stream_;
    std::vector<cudaStream_t> managed_streams_;
    mutable std::mutex mutex_;
};

} // namespace CUDA
} // namespace SplatRender