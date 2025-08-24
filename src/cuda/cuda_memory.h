#pragma once

#include "cuda_utils.h"
#include <vector>
#include <memory>
#include <unordered_map>

namespace SplatRender {
namespace CUDA {

// Template class for GPU memory management
template<typename T>
class CudaMemory {
public:
    CudaMemory() : d_ptr_(nullptr), count_(0), allocated_count_(0) {}
    
    explicit CudaMemory(size_t count) : d_ptr_(nullptr), count_(0), allocated_count_(0) {
        allocate(count);
    }
    
    ~CudaMemory() {
        free();
    }
    
    // No copy, only move
    CudaMemory(const CudaMemory&) = delete;
    CudaMemory& operator=(const CudaMemory&) = delete;
    
    CudaMemory(CudaMemory&& other) noexcept 
        : d_ptr_(other.d_ptr_), count_(other.count_), allocated_count_(other.allocated_count_) {
        other.d_ptr_ = nullptr;
        other.count_ = 0;
        other.allocated_count_ = 0;
    }
    
    CudaMemory& operator=(CudaMemory&& other) noexcept {
        if (this != &other) {
            free();
            d_ptr_ = other.d_ptr_;
            count_ = other.count_;
            allocated_count_ = other.allocated_count_;
            other.d_ptr_ = nullptr;
            other.count_ = 0;
            other.allocated_count_ = 0;
        }
        return *this;
    }
    
    // Allocate memory on device
    void allocate(size_t count) {
        if (count == 0) return;
        
        // Only reallocate if we need more space
        if (count > allocated_count_) {
            free();
            size_t bytes = count * sizeof(T);
            CUDA_CHECK(cudaMalloc(&d_ptr_, bytes));
            allocated_count_ = count;
        }
        count_ = count;
    }
    
    // Free device memory
    void free() {
        if (d_ptr_) {
            cudaFree(d_ptr_);
            d_ptr_ = nullptr;
            count_ = 0;
            allocated_count_ = 0;
        }
    }
    
    // Resize (may reallocate)
    void resize(size_t new_count) {
        allocate(new_count);
    }
    
    // Copy data from host to device
    void copyFromHost(const T* host_data, size_t count) {
        if (count > allocated_count_) {
            allocate(count);
        }
        if (count > 0) {
            CUDA_CHECK(cudaMemcpy(d_ptr_, host_data, count * sizeof(T), cudaMemcpyHostToDevice));
            count_ = count;
        }
    }
    
    // Copy data from device to host
    void copyToHost(T* host_data, size_t count) const {
        size_t copy_count = (count < count_) ? count : count_;
        if (copy_count > 0) {
            CUDA_CHECK(cudaMemcpy(host_data, d_ptr_, copy_count * sizeof(T), cudaMemcpyDeviceToHost));
        }
    }
    
    // Async copy from host to device
    void copyFromHostAsync(const T* host_data, size_t count, cudaStream_t stream) {
        if (count > allocated_count_) {
            allocate(count);
        }
        if (count > 0) {
            CUDA_CHECK(cudaMemcpyAsync(d_ptr_, host_data, count * sizeof(T), 
                                       cudaMemcpyHostToDevice, stream));
            count_ = count;
        }
    }
    
    // Async copy from device to host
    void copyToHostAsync(T* host_data, size_t count, cudaStream_t stream) const {
        size_t copy_count = (count < count_) ? count : count_;
        if (copy_count > 0) {
            CUDA_CHECK(cudaMemcpyAsync(host_data, d_ptr_, copy_count * sizeof(T), 
                                       cudaMemcpyDeviceToHost, stream));
        }
    }
    
    // Clear memory (set to zero)
    void clear() {
        if (d_ptr_ && count_ > 0) {
            CUDA_CHECK(cudaMemset(d_ptr_, 0, count_ * sizeof(T)));
        }
    }
    
    // Get device pointer
    T* getDevicePtr() { return d_ptr_; }
    const T* getDevicePtr() const { return d_ptr_; }
    
    // Get size info
    size_t size() const { return count_; }
    size_t capacity() const { return allocated_count_; }
    size_t bytes() const { return count_ * sizeof(T); }
    size_t allocatedBytes() const { return allocated_count_ * sizeof(T); }
    bool empty() const { return count_ == 0; }
    
private:
    T* d_ptr_;
    size_t count_;           // Current number of elements
    size_t allocated_count_; // Allocated capacity
};

// Pinned host memory for faster transfers
template<typename T>
class PinnedMemory {
public:
    PinnedMemory() : h_ptr_(nullptr), count_(0) {}
    
    explicit PinnedMemory(size_t count) : h_ptr_(nullptr), count_(0) {
        allocate(count);
    }
    
    ~PinnedMemory() {
        free();
    }
    
    // No copy, only move
    PinnedMemory(const PinnedMemory&) = delete;
    PinnedMemory& operator=(const PinnedMemory&) = delete;
    
    PinnedMemory(PinnedMemory&& other) noexcept 
        : h_ptr_(other.h_ptr_), count_(other.count_) {
        other.h_ptr_ = nullptr;
        other.count_ = 0;
    }
    
    PinnedMemory& operator=(PinnedMemory&& other) noexcept {
        if (this != &other) {
            free();
            h_ptr_ = other.h_ptr_;
            count_ = other.count_;
            other.h_ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }
    
    void allocate(size_t count) {
        if (count == 0) return;
        
        free();
        size_t bytes = count * sizeof(T);
        CUDA_CHECK(cudaHostAlloc(&h_ptr_, bytes, cudaHostAllocDefault));
        count_ = count;
    }
    
    void free() {
        if (h_ptr_) {
            cudaFreeHost(h_ptr_);
            h_ptr_ = nullptr;
            count_ = 0;
        }
    }
    
    T* getHostPtr() { return h_ptr_; }
    const T* getHostPtr() const { return h_ptr_; }
    
    size_t size() const { return count_; }
    size_t bytes() const { return count_ * sizeof(T); }
    bool empty() const { return count_ == 0; }
    
    // Array access
    T& operator[](size_t idx) { return h_ptr_[idx]; }
    const T& operator[](size_t idx) const { return h_ptr_[idx]; }
    
private:
    T* h_ptr_;
    size_t count_;
};

// Memory pool for efficient allocation
class CudaMemoryPool {
public:
    CudaMemoryPool(size_t initial_size = 1024 * 1024 * 100); // 100MB default
    ~CudaMemoryPool();
    
    void* allocate(size_t bytes, size_t alignment = 256);
    void deallocate(void* ptr);
    void reset();
    
    size_t getTotalAllocated() const { return total_allocated_; }
    size_t getTotalUsed() const { return total_used_; }
    size_t getFragmentation() const;
    
private:
    struct Block {
        void* ptr;
        size_t size;
        size_t offset;
        bool in_use;
    };
    
    void* pool_ptr_;
    size_t pool_size_;
    size_t total_allocated_;
    size_t total_used_;
    std::vector<Block> blocks_;
    
    void defragment();
};

// Utility functions for memory info
void printMemoryUsage();
size_t getAvailableMemory();
size_t getTotalMemory();

} // namespace CUDA
} // namespace SplatRender