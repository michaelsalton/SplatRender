#pragma once

#include <cuda_runtime.h>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <iomanip>
#include <iostream>
#include <float.h>

namespace splat {

class KernelProfiler {
public:
    struct KernelStats {
        float min_ms = FLT_MAX;
        float max_ms = 0.0f;
        float avg_ms = 0.0f;
        float total_ms = 0.0f;
        float last_ms = 0.0f;
        int call_count = 0;
        
        void update(float time_ms) {
            last_ms = time_ms;
            min_ms = std::min(min_ms, time_ms);
            max_ms = std::max(max_ms, time_ms);
            total_ms += time_ms;
            call_count++;
            avg_ms = total_ms / call_count;
        }
    };
    
    struct FrameStats {
        float projection_ms = 0.0f;
        float tiling_ms = 0.0f;
        float sorting_ms = 0.0f;
        float rasterization_ms = 0.0f;
        float total_ms = 0.0f;
        int rendered_gaussians = 0;
        int culled_gaussians = 0;
        
        float getTotalKernelTime() const {
            return projection_ms + tiling_ms + sorting_ms + rasterization_ms;
        }
    };
    
    static KernelProfiler& getInstance() {
        static KernelProfiler instance;
        return instance;
    }
    
    void startTimer(const std::string& kernel_name);
    void endTimer(const std::string& kernel_name);
    float getLastTime(const std::string& kernel_name) const;
    KernelStats getStats(const std::string& kernel_name) const;
    FrameStats getFrameStats() const { return frame_stats_; }
    void updateFrameStats(const FrameStats& stats) { frame_stats_ = stats; }
    
    void reset();
    void printReport() const;
    void printFrameReport() const;
    void setEnabled(bool enabled) { enabled_ = enabled; }
    bool isEnabled() const { return enabled_; }
    
    // Convenience class for RAII timing
    class ScopedTimer {
    public:
        ScopedTimer(const std::string& kernel_name) 
            : kernel_name_(kernel_name), profiler_(KernelProfiler::getInstance()) {
            if (profiler_.isEnabled()) {
                profiler_.startTimer(kernel_name_);
            }
        }
        
        ~ScopedTimer() {
            if (profiler_.isEnabled()) {
                profiler_.endTimer(kernel_name_);
            }
        }
        
    private:
        std::string kernel_name_;
        KernelProfiler& profiler_;
    };
    
private:
    KernelProfiler();
    ~KernelProfiler();
    
    KernelProfiler(const KernelProfiler&) = delete;
    KernelProfiler& operator=(const KernelProfiler&) = delete;
    
    struct TimerData {
        cudaEvent_t start;
        cudaEvent_t stop;
        bool timing = false;
    };
    
    mutable std::mutex mutex_;
    std::map<std::string, KernelStats> kernel_stats_;
    std::map<std::string, TimerData> timers_;
    FrameStats frame_stats_;
    bool enabled_ = true;
    
    TimerData& getOrCreateTimer(const std::string& kernel_name);
};

// Macro for easy kernel timing
#define PROFILE_KERNEL(name) KernelProfiler::ScopedTimer _timer(name)

// Memory profiler for tracking GPU memory usage
class MemoryProfiler {
public:
    struct MemoryStats {
        size_t peak_usage = 0;
        size_t current_usage = 0;
        size_t allocation_count = 0;
        size_t deallocation_count = 0;
        float bandwidth_gb_s = 0.0f;
        
        // Per-allocation type tracking
        size_t gaussian_memory = 0;
        size_t tile_memory = 0;
        size_t output_memory = 0;
        size_t temp_memory = 0;
    };
    
    static MemoryProfiler& getInstance() {
        static MemoryProfiler instance;
        return instance;
    }
    
    void recordAllocation(const std::string& name, size_t bytes);
    void recordDeallocation(const std::string& name, size_t bytes);
    void recordTransfer(size_t bytes, float time_ms);
    
    MemoryStats getStats() const;
    void printReport() const;
    void reset();
    
private:
    MemoryProfiler() = default;
    
    mutable std::mutex mutex_;
    MemoryStats stats_;
    std::map<std::string, size_t> allocations_;
};

// Performance monitor for overall metrics
class PerformanceMonitor {
public:
    struct Metrics {
        float fps = 0.0f;
        float frame_time_ms = 0.0f;
        float gpu_utilization = 0.0f;
        float memory_bandwidth_gb_s = 0.0f;
        int rendered_gaussians = 0;
        int culled_gaussians = 0;
        int total_tiles = 0;
        int active_tiles = 0;
        
        // Kernel breakdown (percentage of frame time)
        float projection_percent = 0.0f;
        float tiling_percent = 0.0f;
        float sorting_percent = 0.0f;
        float rasterization_percent = 0.0f;
    };
    
    static PerformanceMonitor& getInstance() {
        static PerformanceMonitor instance;
        return instance;
    }
    
    void startFrame();
    void endFrame();
    void updateGaussianStats(int rendered, int culled);
    void updateTileStats(int total, int active);
    
    Metrics getMetrics() const { return current_metrics_; }
    Metrics getAverageMetrics() const { return average_metrics_; }
    
    void printMetrics() const;
    void reset();
    
private:
    PerformanceMonitor();
    
    mutable std::mutex mutex_;
    Metrics current_metrics_;
    Metrics average_metrics_;
    
    cudaEvent_t frame_start_;
    cudaEvent_t frame_end_;
    
    std::vector<float> frame_times_;
    static constexpr int AVERAGING_WINDOW = 60;  // Average over 60 frames
    
    void updateAverages();
};

}  // namespace splat