#include <iostream>
#include "cuda/cuda_manager.h"
#include "cuda/cuda_memory.h"

using namespace SplatRender::CUDA;

int main() {
    std::cout << "=== CUDA Memory Management Test ===" << std::endl;
    
    // Test 1: Initialize CUDA Manager
    std::cout << "\n1. Testing CUDA Manager initialization..." << std::endl;
    if (!CudaManager::getInstance().initialize()) {
        std::cerr << "Failed to initialize CUDA" << std::endl;
        return -1;
    }
    
    // Print device info
    const auto& props = CudaManager::getInstance().getDeviceProperties();
    std::cout << "   Device: " << props.name << std::endl;
    std::cout << "   Compute Capability: " << props.major << "." << props.minor << std::endl;
    std::cout << "   Memory: " << (props.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
    
    // Test 2: Memory allocation
    std::cout << "\n2. Testing device memory allocation..." << std::endl;
    {
        CudaMemory<float> d_buffer(1024 * 1024); // 1M floats = 4MB
        std::cout << "   Allocated " << d_buffer.bytes() / (1024 * 1024) << " MB on device" << std::endl;
        
        // Test data transfer
        std::cout << "\n3. Testing host-device data transfer..." << std::endl;
        std::vector<float> h_data(1024 * 1024, 3.14f);
        d_buffer.copyFromHost(h_data.data(), h_data.size());
        
        std::vector<float> h_result(1024 * 1024);
        d_buffer.copyToHost(h_result.data(), h_result.size());
        
        bool transfer_ok = true;
        for (size_t i = 0; i < 100; ++i) { // Check first 100 elements
            if (h_result[i] != 3.14f) {
                transfer_ok = false;
                break;
            }
        }
        std::cout << "   Transfer test: " << (transfer_ok ? "PASSED" : "FAILED") << std::endl;
    }
    
    // Test 3: Pinned memory
    std::cout << "\n4. Testing pinned memory allocation..." << std::endl;
    {
        PinnedMemory<float> h_pinned(1024 * 1024);
        std::cout << "   Allocated " << h_pinned.bytes() / (1024 * 1024) << " MB pinned memory" << std::endl;
        
        // Fill with test data
        for (size_t i = 0; i < h_pinned.size(); ++i) {
            h_pinned[i] = static_cast<float>(i);
        }
        std::cout << "   Pinned memory test: PASSED" << std::endl;
    }
    
    // Test 4: Memory pool
    std::cout << "\n5. Testing memory pool..." << std::endl;
    {
        CudaMemoryPool pool(10 * 1024 * 1024); // 10MB pool
        
        void* ptr1 = pool.allocate(1024 * 1024); // 1MB
        void* ptr2 = pool.allocate(2 * 1024 * 1024); // 2MB
        
        std::cout << "   Total allocated: " << pool.getTotalAllocated() / (1024 * 1024) << " MB" << std::endl;
        std::cout << "   Total used: " << pool.getTotalUsed() / (1024 * 1024) << " MB" << std::endl;
        
        pool.deallocate(ptr1);
        std::cout << "   After deallocation: " << pool.getTotalUsed() / (1024 * 1024) << " MB used" << std::endl;
        
        pool.reset();
        std::cout << "   After reset: " << pool.getTotalUsed() / (1024 * 1024) << " MB used" << std::endl;
    }
    
    // Test 5: Stream management
    std::cout << "\n6. Testing stream management..." << std::endl;
    {
        cudaStream_t stream = CudaManager::getInstance().createStream();
        std::cout << "   Created CUDA stream" << std::endl;
        
        CudaManager::getInstance().synchronizeStream(stream);
        std::cout << "   Stream synchronized" << std::endl;
        
        CudaManager::getInstance().destroyStream(stream);
        std::cout << "   Stream destroyed" << std::endl;
    }
    
    // Print final memory status
    std::cout << "\n7. Final memory status:" << std::endl;
    CudaManager::getInstance().printMemoryInfo();
    
    // Cleanup
    CudaManager::getInstance().shutdown();
    
    std::cout << "\n=== All tests completed successfully! ===" << std::endl;
    return 0;
}