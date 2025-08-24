#include "kernel_profiler.h"
#include "cuda_utils.h"
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace splat {

// KernelProfiler implementation
KernelProfiler::KernelProfiler() : enabled_(true) {}

KernelProfiler::~KernelProfiler() {
    // Clean up CUDA events
    for (auto& [name, timer] : timers_) {
        if (timer.start) cudaEventDestroy(timer.start);
        if (timer.stop) cudaEventDestroy(timer.stop);
    }
}

KernelProfiler::TimerData& KernelProfiler::getOrCreateTimer(const std::string& kernel_name) {
    auto it = timers_.find(kernel_name);
    if (it == timers_.end()) {
        TimerData timer;
        CUDA_CHECK(cudaEventCreate(&timer.start));
        CUDA_CHECK(cudaEventCreate(&timer.stop));
        timer.timing = false;
        timers_[kernel_name] = timer;
        return timers_[kernel_name];
    }
    return it->second;
}

void KernelProfiler::startTimer(const std::string& kernel_name) {
    if (!enabled_) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    auto& timer = getOrCreateTimer(kernel_name);
    
    if (timer.timing) {
        std::cerr << "Warning: Timer for " << kernel_name << " already started\n";
        return;
    }
    
    CUDA_CHECK(cudaEventRecord(timer.start));
    timer.timing = true;
}

void KernelProfiler::endTimer(const std::string& kernel_name) {
    if (!enabled_) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = timers_.find(kernel_name);
    if (it == timers_.end() || !it->second.timing) {
        std::cerr << "Warning: Timer for " << kernel_name << " not started\n";
        return;
    }
    
    auto& timer = it->second;
    CUDA_CHECK(cudaEventRecord(timer.stop));
    CUDA_CHECK(cudaEventSynchronize(timer.stop));
    
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timer.start, timer.stop));
    
    kernel_stats_[kernel_name].update(elapsed_ms);
    timer.timing = false;
}

float KernelProfiler::getLastTime(const std::string& kernel_name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = kernel_stats_.find(kernel_name);
    if (it != kernel_stats_.end()) {
        return it->second.last_ms;
    }
    return 0.0f;
}

KernelProfiler::KernelStats KernelProfiler::getStats(const std::string& kernel_name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = kernel_stats_.find(kernel_name);
    if (it != kernel_stats_.end()) {
        return it->second;
    }
    return KernelStats();
}

void KernelProfiler::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    kernel_stats_.clear();
    frame_stats_ = FrameStats();
}

void KernelProfiler::printReport() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::cout << "\n========== Kernel Performance Report ==========\n";
    std::cout << std::setw(25) << "Kernel Name" 
              << std::setw(10) << "Calls"
              << std::setw(12) << "Avg (ms)"
              << std::setw(12) << "Min (ms)"
              << std::setw(12) << "Max (ms)"
              << std::setw(12) << "Total (ms)" << "\n";
    std::cout << std::string(80, '-') << "\n";
    
    for (const auto& [name, stats] : kernel_stats_) {
        std::cout << std::setw(25) << name
                  << std::setw(10) << stats.call_count
                  << std::setw(12) << std::fixed << std::setprecision(3) << stats.avg_ms
                  << std::setw(12) << stats.min_ms
                  << std::setw(12) << stats.max_ms
                  << std::setw(12) << stats.total_ms << "\n";
    }
    std::cout << "===============================================\n";
}

void KernelProfiler::printFrameReport() const {
    const auto& stats = frame_stats_;
    std::cout << "\n===== Frame Timing =====\n";
    std::cout << "Projection:     " << std::fixed << std::setprecision(3) 
              << stats.projection_ms << " ms\n";
    std::cout << "Tiling:         " << stats.tiling_ms << " ms\n";
    std::cout << "Sorting:        " << stats.sorting_ms << " ms\n";
    std::cout << "Rasterization:  " << stats.rasterization_ms << " ms\n";
    std::cout << "Total Kernels:  " << stats.getTotalKernelTime() << " ms\n";
    std::cout << "Total Frame:    " << stats.total_ms << " ms\n";
    std::cout << "Gaussians:      " << stats.rendered_gaussians 
              << " rendered / " << stats.culled_gaussians << " culled\n";
    std::cout << "========================\n";
}

// MemoryProfiler implementation
void MemoryProfiler::recordAllocation(const std::string& name, size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    allocations_[name] = bytes;
    stats_.current_usage += bytes;
    stats_.peak_usage = std::max(stats_.peak_usage, stats_.current_usage);
    stats_.allocation_count++;
    
    // Categorize allocation
    if (name.find("gaussian") != std::string::npos) {
        stats_.gaussian_memory += bytes;
    } else if (name.find("tile") != std::string::npos) {
        stats_.tile_memory += bytes;
    } else if (name.find("output") != std::string::npos) {
        stats_.output_memory += bytes;
    } else {
        stats_.temp_memory += bytes;
    }
}

void MemoryProfiler::recordDeallocation(const std::string& name, size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = allocations_.find(name);
    if (it != allocations_.end()) {
        stats_.current_usage -= it->second;
        
        // Update category
        if (name.find("gaussian") != std::string::npos) {
            stats_.gaussian_memory -= it->second;
        } else if (name.find("tile") != std::string::npos) {
            stats_.tile_memory -= it->second;
        } else if (name.find("output") != std::string::npos) {
            stats_.output_memory -= it->second;
        } else {
            stats_.temp_memory -= it->second;
        }
        
        allocations_.erase(it);
        stats_.deallocation_count++;
    }
}

void MemoryProfiler::recordTransfer(size_t bytes, float time_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (time_ms > 0) {
        float bandwidth = (bytes / (1024.0f * 1024.0f * 1024.0f)) / (time_ms / 1000.0f);
        stats_.bandwidth_gb_s = bandwidth;
    }
}

MemoryProfiler::MemoryStats MemoryProfiler::getStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void MemoryProfiler::printReport() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::cout << "\n========== Memory Usage Report ==========\n";
    std::cout << "Current Usage:     " << (stats_.current_usage / (1024.0f * 1024.0f)) << " MB\n";
    std::cout << "Peak Usage:        " << (stats_.peak_usage / (1024.0f * 1024.0f)) << " MB\n";
    std::cout << "Allocations:       " << stats_.allocation_count << "\n";
    std::cout << "Deallocations:     " << stats_.deallocation_count << "\n";
    std::cout << "\nMemory by Type:\n";
    std::cout << "  Gaussians:       " << (stats_.gaussian_memory / (1024.0f * 1024.0f)) << " MB\n";
    std::cout << "  Tiles:           " << (stats_.tile_memory / (1024.0f * 1024.0f)) << " MB\n";
    std::cout << "  Output:          " << (stats_.output_memory / (1024.0f * 1024.0f)) << " MB\n";
    std::cout << "  Temporary:       " << (stats_.temp_memory / (1024.0f * 1024.0f)) << " MB\n";
    std::cout << "Bandwidth:         " << stats_.bandwidth_gb_s << " GB/s\n";
    std::cout << "=========================================\n";
}

void MemoryProfiler::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_ = MemoryStats();
    allocations_.clear();
}

// PerformanceMonitor implementation
PerformanceMonitor::PerformanceMonitor() {
    CUDA_CHECK(cudaEventCreate(&frame_start_));
    CUDA_CHECK(cudaEventCreate(&frame_end_));
}

void PerformanceMonitor::startFrame() {
    CUDA_CHECK(cudaEventRecord(frame_start_));
}

void PerformanceMonitor::endFrame() {
    CUDA_CHECK(cudaEventRecord(frame_end_));
    CUDA_CHECK(cudaEventSynchronize(frame_end_));
    
    float frame_time_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&frame_time_ms, frame_start_, frame_end_));
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Update current metrics
    current_metrics_.frame_time_ms = frame_time_ms;
    current_metrics_.fps = (frame_time_ms > 0) ? 1000.0f / frame_time_ms : 0.0f;
    
    // Get kernel timings from profiler
    auto& profiler = KernelProfiler::getInstance();
    auto frame_stats = profiler.getFrameStats();
    float total_kernel_time = frame_stats.getTotalKernelTime();
    
    if (total_kernel_time > 0) {
        current_metrics_.projection_percent = (frame_stats.projection_ms / total_kernel_time) * 100.0f;
        current_metrics_.tiling_percent = (frame_stats.tiling_ms / total_kernel_time) * 100.0f;
        current_metrics_.sorting_percent = (frame_stats.sorting_ms / total_kernel_time) * 100.0f;
        current_metrics_.rasterization_percent = (frame_stats.rasterization_ms / total_kernel_time) * 100.0f;
    }
    
    // Update memory bandwidth
    auto mem_stats = MemoryProfiler::getInstance().getStats();
    current_metrics_.memory_bandwidth_gb_s = mem_stats.bandwidth_gb_s;
    
    // Store frame time for averaging
    frame_times_.push_back(frame_time_ms);
    if (frame_times_.size() > AVERAGING_WINDOW) {
        frame_times_.erase(frame_times_.begin());
    }
    
    updateAverages();
}

void PerformanceMonitor::updateGaussianStats(int rendered, int culled) {
    std::lock_guard<std::mutex> lock(mutex_);
    current_metrics_.rendered_gaussians = rendered;
    current_metrics_.culled_gaussians = culled;
}

void PerformanceMonitor::updateTileStats(int total, int active) {
    std::lock_guard<std::mutex> lock(mutex_);
    current_metrics_.total_tiles = total;
    current_metrics_.active_tiles = active;
}

void PerformanceMonitor::updateAverages() {
    if (frame_times_.empty()) return;
    
    float avg_frame_time = 0.0f;
    for (float time : frame_times_) {
        avg_frame_time += time;
    }
    avg_frame_time /= frame_times_.size();
    
    average_metrics_.frame_time_ms = avg_frame_time;
    average_metrics_.fps = (avg_frame_time > 0) ? 1000.0f / avg_frame_time : 0.0f;
    
    // Copy other metrics
    average_metrics_.rendered_gaussians = current_metrics_.rendered_gaussians;
    average_metrics_.culled_gaussians = current_metrics_.culled_gaussians;
    average_metrics_.total_tiles = current_metrics_.total_tiles;
    average_metrics_.active_tiles = current_metrics_.active_tiles;
    average_metrics_.memory_bandwidth_gb_s = current_metrics_.memory_bandwidth_gb_s;
}

void PerformanceMonitor::printMetrics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::cout << "\n========== Performance Metrics ==========\n";
    std::cout << "FPS:              " << std::fixed << std::setprecision(1) 
              << current_metrics_.fps << " (avg: " << average_metrics_.fps << ")\n";
    std::cout << "Frame Time:       " << std::setprecision(2) 
              << current_metrics_.frame_time_ms << " ms (avg: " 
              << average_metrics_.frame_time_ms << " ms)\n";
    std::cout << "Gaussians:        " << current_metrics_.rendered_gaussians 
              << " rendered / " << current_metrics_.culled_gaussians << " culled\n";
    std::cout << "Tiles:            " << current_metrics_.active_tiles << " / " 
              << current_metrics_.total_tiles << " active\n";
    std::cout << "Memory Bandwidth: " << current_metrics_.memory_bandwidth_gb_s << " GB/s\n";
    std::cout << "\nKernel Breakdown:\n";
    std::cout << "  Projection:     " << std::setprecision(1) 
              << current_metrics_.projection_percent << "%\n";
    std::cout << "  Tiling:         " << current_metrics_.tiling_percent << "%\n";
    std::cout << "  Sorting:        " << current_metrics_.sorting_percent << "%\n";
    std::cout << "  Rasterization:  " << current_metrics_.rasterization_percent << "%\n";
    std::cout << "=========================================\n";
}

void PerformanceMonitor::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    current_metrics_ = Metrics();
    average_metrics_ = Metrics();
    frame_times_.clear();
}

}  // namespace splat