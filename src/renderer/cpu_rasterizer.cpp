#include "renderer/cpu_rasterizer.h"
#include "core/camera.h"
#include "math/gaussian.h"
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <cmath>

namespace SplatRender {

// Helper structure for tile-based rendering
struct TileData {
    int x, y;  // Tile coordinates (in tiles, not pixels)
    std::vector<int> gaussian_indices;  // Indices of Gaussians affecting this tile
};

CPURasterizer::CPURasterizer() {
}

CPURasterizer::~CPURasterizer() {
}

void CPURasterizer::initialize(const RenderSettings& settings) {
    settings_ = settings;
    
    // Pre-allocate buffers
    int pixel_count = settings_.width * settings_.height;
    depth_buffer_.resize(pixel_count);
    
    // Reserve space for projected Gaussians (estimate)
    projected_gaussians_.reserve(100000);
}

void CPURasterizer::render(const std::vector<Gaussian3D>& gaussians,
                          const Camera& camera,
                          std::vector<float>& output_buffer) {
    auto start_time = std::chrono::steady_clock::now();
    
    // Clear buffers
    clearBuffer(output_buffer);
    std::fill(depth_buffer_.begin(), depth_buffer_.end(), std::numeric_limits<float>::infinity());
    
    // Reset stats
    stats_ = Stats();
    
    // Stage 1: Project Gaussians to screen space
    auto stage_start = std::chrono::steady_clock::now();
    projected_gaussians_.clear();
    projectGaussians(gaussians, camera, projected_gaussians_);
    auto stage_end = std::chrono::steady_clock::now();
    stats_.projection_time_ms = std::chrono::duration<float, std::milli>(stage_end - stage_start).count();
    
    // Stage 2: Cull invisible Gaussians
    stage_start = std::chrono::steady_clock::now();
    if (settings_.enable_culling) {
        cullGaussians(projected_gaussians_);
    }
    stage_end = std::chrono::steady_clock::now();
    // Culling time is included in projection time for now
    
    stats_.visible_gaussians = projected_gaussians_.size();
    stats_.culled_gaussians = gaussians.size() - projected_gaussians_.size();
    
    // Stage 3: Rasterize Gaussians
    stage_start = std::chrono::steady_clock::now();
    rasterizeGaussians(projected_gaussians_, output_buffer);
    stage_end = std::chrono::steady_clock::now();
    stats_.rasterization_time_ms = std::chrono::duration<float, std::milli>(stage_end - stage_start).count();
    
    // Total time
    auto end_time = std::chrono::steady_clock::now();
    stats_.total_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
}

void CPURasterizer::projectGaussians(const std::vector<Gaussian3D>& gaussians_3d,
                                    const Camera& camera,
                                    std::vector<Gaussian2D>& gaussians_2d) {
    // Get camera matrices
    glm::mat4 view_matrix = camera.getViewMatrix();
    glm::mat4 proj_matrix = camera.getProjectionMatrix(
        static_cast<float>(settings_.width) / static_cast<float>(settings_.height)
    );
    
    // Get camera position for view-dependent effects
    glm::vec3 camera_pos = camera.getPosition();
    
    // Project each Gaussian
    for (const auto& g3d : gaussians_3d) {
        // Transform to view space
        glm::vec4 pos_view = view_matrix * glm::vec4(g3d.position, 1.0f);
        
        // Skip if behind camera
        if (pos_view.z > -0.1f) {
            stats_.culled_gaussians++;
            continue;
        }
        
        // Project to clip space
        glm::vec4 pos_clip = proj_matrix * pos_view;
        
        // Perspective divide to NDC
        glm::vec3 pos_ndc = glm::vec3(pos_clip) / pos_clip.w;
        
        // Skip if outside NDC bounds
        if (std::abs(pos_ndc.x) > 1.5f || std::abs(pos_ndc.y) > 1.5f) {
            stats_.culled_gaussians++;
            continue;
        }
        
        // Convert to screen space
        // NDC is [-1,1], map to [0,width-1] and [0,height-1]
        float screen_x = (pos_ndc.x * 0.5f + 0.5f) * settings_.width;
        float screen_y = (0.5f - pos_ndc.y * 0.5f) * settings_.height;
        
        
        // Compute 2D covariance
        glm::mat3 cov_3d = g3d.computeCovariance3D();
        glm::mat2 cov_2d = GaussianUtils::computeCovariance2D(
            cov_3d,
            glm::vec3(pos_view),
            view_matrix,
            proj_matrix
        );
        
        // Create 2D Gaussian
        Gaussian2D g2d;
        g2d.center = glm::vec2(screen_x, screen_y);
        g2d.cov_2d = cov_2d;
        g2d.depth = -pos_view.z;  // Positive depth for sorting
        g2d.alpha = g3d.opacity;
        
        // Compute view-dependent color
        glm::vec3 view_dir = glm::normalize(camera_pos - g3d.position);
        g2d.color = g3d.evaluateColor(view_dir);
        
        // Compute screen-space radius from covariance
        Gaussian2D temp = GaussianUtils::projectToScreen(g3d, view_matrix, proj_matrix, settings_.width, settings_.height);
        g2d.radius = temp.radius;
        
        // Skip if too small
        if (g2d.radius < 0.5f) {
            stats_.culled_gaussians++;
            continue;
        }
        
        gaussians_2d.push_back(g2d);
    }
}

void CPURasterizer::cullGaussians(std::vector<Gaussian2D>& gaussians) {
    // Remove Gaussians that don't affect any pixels
    auto new_end = std::remove_if(gaussians.begin(), gaussians.end(),
        [this](const Gaussian2D& g) {
            return !isGaussianVisible(g);
        });
    
    int culled = std::distance(new_end, gaussians.end());
    stats_.culled_gaussians += culled;
    
    gaussians.erase(new_end, gaussians.end());
}

bool CPURasterizer::isGaussianVisible(const Gaussian2D& gaussian) const {
    // Check if Gaussian is within screen bounds (with radius)
    float min_x = gaussian.center.x - gaussian.radius;
    float max_x = gaussian.center.x + gaussian.radius;
    float min_y = gaussian.center.y - gaussian.radius;
    float max_y = gaussian.center.y + gaussian.radius;
    
    if (max_x < 0 || min_x >= settings_.width ||
        max_y < 0 || min_y >= settings_.height) {
        return false;
    }
    
    // Check if opacity is significant
    if (gaussian.alpha < settings_.alpha_threshold) {
        return false;
    }
    
    return true;
}

void CPURasterizer::rasterizeGaussians(const std::vector<Gaussian2D>& gaussians,
                                      std::vector<float>& output_buffer) {
    // Initialize output buffer if needed
    size_t buffer_size = settings_.width * settings_.height * 4;
    if (output_buffer.size() != buffer_size) {
        output_buffer.resize(buffer_size);
    }
    
    // Clear buffer to background color
    std::fill(output_buffer.begin(), output_buffer.end(), 0.0f);
    
    
    // Tile-based rendering for better cache performance
    int tiles_x = (settings_.width + settings_.tile_size - 1) / settings_.tile_size;
    int tiles_y = (settings_.height + settings_.tile_size - 1) / settings_.tile_size;
    
    // Build tile lists
    std::vector<TileData> tiles(tiles_x * tiles_y);
    
    // Initialize tiles
    for (int ty = 0; ty < tiles_y; ++ty) {
        for (int tx = 0; tx < tiles_x; ++tx) {
            int tile_idx = ty * tiles_x + tx;
            tiles[tile_idx].x = tx;
            tiles[tile_idx].y = ty;
        }
    }
    
    // Assign Gaussians to tiles
    for (size_t g_idx = 0; g_idx < gaussians.size(); ++g_idx) {
        const auto& gaussian = gaussians[g_idx];
        
        // Find affected tiles
        int min_tx = std::max(0, static_cast<int>((gaussian.center.x - gaussian.radius) / settings_.tile_size));
        int max_tx = std::min(tiles_x - 1, static_cast<int>((gaussian.center.x + gaussian.radius) / settings_.tile_size));
        int min_ty = std::max(0, static_cast<int>((gaussian.center.y - gaussian.radius) / settings_.tile_size));
        int max_ty = std::min(tiles_y - 1, static_cast<int>((gaussian.center.y + gaussian.radius) / settings_.tile_size));
        
        // Add to tile lists
        for (int ty = min_ty; ty <= max_ty; ++ty) {
            for (int tx = min_tx; tx <= max_tx; ++tx) {
                int tile_idx = ty * tiles_x + tx;
                tiles[tile_idx].gaussian_indices.push_back(g_idx);
            }
        }
    }
    
    // Sort Gaussians in each tile by depth
    auto sort_start = std::chrono::steady_clock::now();
    for (auto& tile : tiles) {
        std::sort(tile.gaussian_indices.begin(), tile.gaussian_indices.end(),
            [&gaussians](int a, int b) {
                return gaussians[a].depth < gaussians[b].depth;  // Front to back
            });
    }
    auto sort_end = std::chrono::steady_clock::now();
    stats_.sorting_time_ms = std::chrono::duration<float, std::milli>(sort_end - sort_start).count();
    
    // Debug: Draw solid squares at Gaussian centers
    static bool debug_squares = false;
    if (debug_squares && !gaussians.empty()) {
        std::cout << "Drawing " << gaussians.size() << " debug squares" << std::endl;
        for (size_t i = 0; i < gaussians.size(); ++i) {
            const auto& g = gaussians[i];
            // Draw 40x40 pixel square at each Gaussian center
            int cx = static_cast<int>(g.center.x);
            int cy = static_cast<int>(g.center.y);
            std::cout << "Square " << i << " at (" << cx << "," << cy << ") color=(" 
                      << g.color.r << "," << g.color.g << "," << g.color.b << ")" << std::endl;
            
            // Clamp center to ensure square is visible
            cx = std::max(20, std::min(settings_.width - 20, cx));
            cy = std::max(20, std::min(settings_.height - 20, cy));
            
            for (int dy = -20; dy <= 20; ++dy) {
                for (int dx = -20; dx <= 20; ++dx) {
                    int px = cx + dx;
                    int py = cy + dy;
                    if (px >= 0 && px < settings_.width && py >= 0 && py < settings_.height) {
                        int idx = (py * settings_.width + px) * 4;
                        output_buffer[idx + 0] = g.color.r;
                        output_buffer[idx + 1] = g.color.g;
                        output_buffer[idx + 2] = g.color.b;
                        output_buffer[idx + 3] = 1.0f;
                    }
                }
            }
        }
        return; // Skip normal rendering for debugging
    }
    
    // Rasterize each tile
    for (const auto& tile : tiles) {
        int tile_start_x = tile.x * settings_.tile_size;
        int tile_start_y = tile.y * settings_.tile_size;
        int tile_end_x = std::min(tile_start_x + settings_.tile_size, settings_.width);
        int tile_end_y = std::min(tile_start_y + settings_.tile_size, settings_.height);
        
        // Process each pixel in the tile
        for (int py = tile_start_y; py < tile_end_y; ++py) {
            for (int px = tile_start_x; px < tile_end_x; ++px) {
                int pixel_idx = py * settings_.width + px;
                
                // Initialize pixel
                glm::vec3 color(0.0f);
                float alpha = 0.0f;
                
                // Blend Gaussians front-to-back
                for (int g_idx : tile.gaussian_indices) {
                    const auto& gaussian = gaussians[g_idx];
                    
                    // Early termination if alpha is saturated
                    if (alpha > 0.99f) {
                        break;
                    }
                    
                    // Skip if pixel is outside Gaussian's bounding box
                    glm::vec2 pixel_pos(px + 0.5f, py + 0.5f);
                    glm::vec2 diff = pixel_pos - gaussian.center;
                    float dist_sq = glm::dot(diff, diff);
                    
                    if (dist_sq > gaussian.radius * gaussian.radius) {
                        continue;
                    }
                    
                    // Evaluate Gaussian at pixel
                    float gaussian_alpha = gaussian.computeAlpha(pixel_pos);
                    
                    if (gaussian_alpha < settings_.alpha_threshold) {
                        continue;
                    }
                    
                    // Alpha blending (front-to-back)
                    float weight = gaussian_alpha * (1.0f - alpha);
                    color += gaussian.color * weight;
                    alpha += weight;
                }
                
                // Write to output buffer (RGBA format)
                output_buffer[pixel_idx * 4 + 0] = color.r;
                output_buffer[pixel_idx * 4 + 1] = color.g;
                output_buffer[pixel_idx * 4 + 2] = color.b;
                output_buffer[pixel_idx * 4 + 3] = alpha;
                
            }
        }
    }
}

void CPURasterizer::clearBuffer(std::vector<float>& buffer) {
    // Ensure buffer is the right size
    size_t expected_size = settings_.width * settings_.height * 4;  // RGBA
    if (buffer.size() != expected_size) {
        buffer.resize(expected_size);
    }
    
    // Clear to transparent black
    std::fill(buffer.begin(), buffer.end(), 0.0f);
}

} // namespace SplatRender