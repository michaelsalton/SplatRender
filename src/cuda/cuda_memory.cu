#include "cuda_memory.h"
#include <algorithm>
#include <iostream>
#include <iomanip>

namespace SplatRender {
namespace CUDA {

CudaMemoryPool::CudaMemoryPool(size_t initial_size) 
    : pool_ptr_(nullptr), pool_size_(initial_size), total_allocated_(0), total_used_(0) {
    
    // Allocate the pool
    CUDA_CHECK(cudaMalloc(&pool_ptr_, pool_size_));
    total_allocated_ = pool_size_;
    
    // Initialize with one large free block
    blocks_.push_back({pool_ptr_, pool_size_, 0, false});
}

CudaMemoryPool::~CudaMemoryPool() {
    if (pool_ptr_) {
        cudaFree(pool_ptr_);
        pool_ptr_ = nullptr;
    }
}

void* CudaMemoryPool::allocate(size_t bytes, size_t alignment) {
    // Round up to alignment
    size_t aligned_size = ((bytes + alignment - 1) / alignment) * alignment;
    
    // Find a free block that's large enough
    for (auto& block : blocks_) {
        if (!block.in_use && block.size >= aligned_size) {
            // Split the block if it's larger than needed
            if (block.size > aligned_size + alignment) {
                // Create a new free block with the remaining space
                Block new_block;
                new_block.ptr = static_cast<char*>(block.ptr) + aligned_size;
                new_block.size = block.size - aligned_size;
                new_block.offset = block.offset + aligned_size;
                new_block.in_use = false;
                
                blocks_.push_back(new_block);
                
                // Resize the current block
                block.size = aligned_size;
            }
            
            block.in_use = true;
            total_used_ += block.size;
            return block.ptr;
        }
    }
    
    // No suitable block found, try defragmenting
    defragment();
    
    // Try again after defragmentation
    for (auto& block : blocks_) {
        if (!block.in_use && block.size >= aligned_size) {
            block.in_use = true;
            total_used_ += block.size;
            return block.ptr;
        }
    }
    
    // Still no space, allocation failed
    throw std::runtime_error("CudaMemoryPool: Out of memory");
}

void CudaMemoryPool::deallocate(void* ptr) {
    if (!ptr) return;
    
    for (auto& block : blocks_) {
        if (block.ptr == ptr && block.in_use) {
            block.in_use = false;
            total_used_ -= block.size;
            
            // Try to merge with adjacent free blocks
            defragment();
            return;
        }
    }
    
    // Pointer not found in pool (this shouldn't happen)
    std::cerr << "Warning: Attempting to deallocate pointer not in pool" << std::endl;
}

void CudaMemoryPool::reset() {
    // Mark all blocks as free and merge them
    for (auto& block : blocks_) {
        block.in_use = false;
    }
    total_used_ = 0;
    
    // Reset to single large block
    blocks_.clear();
    blocks_.push_back({pool_ptr_, pool_size_, 0, false});
}

size_t CudaMemoryPool::getFragmentation() const {
    if (total_allocated_ == 0) return 0;
    
    // Count free blocks
    size_t free_blocks = 0;
    size_t largest_free = 0;
    size_t total_free = total_allocated_ - total_used_;
    
    for (const auto& block : blocks_) {
        if (!block.in_use) {
            free_blocks++;
            largest_free = std::max(largest_free, block.size);
        }
    }
    
    if (total_free == 0) return 0;
    
    // Fragmentation metric: 1 - (largest_free_block / total_free_space)
    // 0 = no fragmentation (all free space is contiguous)
    // 1 = maximum fragmentation (free space completely scattered)
    return static_cast<size_t>(100.0 * (1.0 - static_cast<double>(largest_free) / total_free));
}

void CudaMemoryPool::defragment() {
    // Sort blocks by offset
    std::sort(blocks_.begin(), blocks_.end(), 
              [](const Block& a, const Block& b) { return a.offset < b.offset; });
    
    // Merge adjacent free blocks
    std::vector<Block> merged_blocks;
    
    for (size_t i = 0; i < blocks_.size(); ++i) {
        if (blocks_[i].in_use) {
            // Keep allocated blocks as-is
            merged_blocks.push_back(blocks_[i]);
        } else {
            // Start a new free block or merge with the last one
            if (!merged_blocks.empty() && !merged_blocks.back().in_use &&
                (static_cast<char*>(merged_blocks.back().ptr) + merged_blocks.back().size == blocks_[i].ptr)) {
                // Merge with previous free block
                merged_blocks.back().size += blocks_[i].size;
            } else {
                // Add as new free block
                merged_blocks.push_back(blocks_[i]);
            }
        }
    }
    
    blocks_ = std::move(merged_blocks);
}

// Utility functions
void printMemoryUsage() {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    
    size_t used_bytes = total_bytes - free_bytes;
    
    std::cout << "CUDA Memory Usage:" << std::endl;
    std::cout << "  Total:     " << std::setw(8) << (total_bytes / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  Used:      " << std::setw(8) << (used_bytes / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  Available: " << std::setw(8) << (free_bytes / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  Usage:     " << std::setw(8) << std::fixed << std::setprecision(1) 
              << (100.0 * used_bytes / total_bytes) << " %" << std::endl;
}

size_t getAvailableMemory() {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    
    cudaError_t error = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (error != cudaSuccess) {
        return 0;
    }
    
    return free_bytes;
}

size_t getTotalMemory() {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    
    cudaError_t error = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (error != cudaSuccess) {
        return 0;
    }
    
    return total_bytes;
}

} // namespace CUDA
} // namespace SplatRender