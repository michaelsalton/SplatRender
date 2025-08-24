#pragma once

#include <cuda_runtime.h>

namespace SplatRender {
namespace CUDA {

// ============================================================================
// CUDA Math Types and Operations
// ============================================================================

// Matrix types
struct float3x3 {
    float m[3][3];
};

struct float4x4 {
    float m[4][4];
};

// Vector operations
__device__ inline float2 operator-(const float2& a, const float2& b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

__device__ inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 operator*(float s, const float3& v) {
    return make_float3(s * v.x, s * v.y, s * v.z);
}

__device__ inline float3 operator*(const float3& v, float s) {
    return make_float3(v.x * s, v.y * s, v.z * s);
}

__device__ inline float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float length(const float3& v) {
    return sqrtf(dot(v, v));
}

__device__ inline float3 normalize(const float3& v) {
    float len = length(v);
    if (len > 0.0f) {
        float inv_len = 1.0f / len;
        return make_float3(v.x * inv_len, v.y * inv_len, v.z * inv_len);
    }
    return v;
}

// ============================================================================
// Constants
// ============================================================================

// Tile configuration
constexpr int TILE_SIZE = 16;  // 16x16 pixels per tile
constexpr int TILE_PIXELS = TILE_SIZE * TILE_SIZE;  // 256 pixels per tile

// Thread block sizes
constexpr int PROJECTION_BLOCK_SIZE = 256;
constexpr int TILING_BLOCK_SIZE = 256;
constexpr int SORTING_BLOCK_SIZE = 256;
constexpr int RASTER_BLOCK_X = 16;
constexpr int RASTER_BLOCK_Y = 16;

// Memory limits
constexpr int MAX_GAUSSIANS_PER_TILE = 1024;
constexpr int MAX_SHARED_GAUSSIANS = 256;  // For rasterization kernel
constexpr int BATCH_SIZE = 32;  // Gaussians loaded per batch in rasterization

// Culling thresholds
constexpr float MIN_OPACITY = 0.01f;
constexpr float MIN_RADIUS = 0.5f;
constexpr float TRANSMITTANCE_THRESHOLD = 0.001f;
constexpr float CULLING_THRESHOLD = 3.0f;  // 3-sigma for Gaussian extent

// Mathematical constants
constexpr float PI = 3.14159265358979323846f;
constexpr float SQRT_2PI = 2.5066282746310005f;

// ============================================================================
// Device Structures (GPU-compatible)
// ============================================================================

// 3D Gaussian structure for GPU
struct __align__(16) GaussianData3D {
    float3 position;      // 12 bytes
    float opacity;        // 4 bytes
    
    float3 scale;         // 12 bytes
    float pad1;           // 4 bytes (padding for alignment)
    
    float4 rotation;      // 16 bytes (quaternion)
    
    // Spherical harmonics coefficients (DC + first 3 degrees)
    // Layout: R[0-15], G[16-31], B[32-47]
    float sh_coeffs[48];  // 192 bytes
};

// 2D Gaussian structure for GPU (after projection)
struct __align__(16) GaussianData2D {
    float2 center;        // 8 bytes - screen position
    float depth;          // 4 bytes - for sorting
    float radius;         // 4 bytes - bounding radius (3-sigma)
    
    // 2D covariance matrix (symmetric, only store 3 values)
    float cov_a;          // 4 bytes - covariance[0][0]
    float cov_b;          // 4 bytes - covariance[0][1] = covariance[1][0]
    float cov_c;          // 4 bytes - covariance[1][1]
    float alpha;          // 4 bytes - opacity after projection
    
    float3 color;         // 12 bytes - RGB after SH evaluation
    float pad;            // 4 bytes - padding for alignment
};

// Tile data structure
struct TileData {
    int gaussian_count;
    int start_offset;  // Offset into global tile list
};

// Camera parameters for kernels
struct CameraParams {
    float4x4 view_matrix;
    float4x4 proj_matrix;
    float4x4 view_proj_matrix;  // Precomputed view * proj
    float3 camera_pos;
    float tan_fovx;
    float tan_fovy;
    float focal_x;
    float focal_y;
    float cx;  // Principal point x
    float cy;  // Principal point y
};

// Render parameters
struct RenderParams {
    int screen_width;
    int screen_height;
    int tiles_x;
    int tiles_y;
    int total_tiles;
    int num_gaussians;
    float near_plane;
    float far_plane;
};

// ============================================================================
// Device Functions (inline GPU helpers)
// ============================================================================

// Compute 2D Gaussian value at a point
__device__ __forceinline__ float evaluateGaussian2D(
    const float2& point,
    const float2& center,
    const float cov_a,
    const float cov_b,
    const float cov_c
) {
    float2 d = point - center;
    
    // Compute inverse of covariance matrix
    float det = cov_a * cov_c - cov_b * cov_b;
    if (det <= 0.0f) return 0.0f;
    
    float inv_det = 1.0f / det;
    float inv_a = cov_c * inv_det;
    float inv_b = -cov_b * inv_det;
    float inv_c = cov_a * inv_det;
    
    // Compute exponent: -0.5 * d^T * inv_cov * d
    float exponent = -0.5f * (d.x * d.x * inv_a + 2.0f * d.x * d.y * inv_b + d.y * d.y * inv_c);
    
    // Avoid numerical issues
    if (exponent < -10.0f) return 0.0f;
    
    return expf(exponent);
}

// Convert quaternion to rotation matrix
__device__ __forceinline__ float3x3 quaternionToMatrix(const float4& q) {
    float3x3 R;
    
    float xx = q.x * q.x;
    float yy = q.y * q.y;
    float zz = q.z * q.z;
    float xy = q.x * q.y;
    float xz = q.x * q.z;
    float yz = q.y * q.z;
    float wx = q.w * q.x;
    float wy = q.w * q.y;
    float wz = q.w * q.z;
    
    R.m[0][0] = 1.0f - 2.0f * (yy + zz);
    R.m[0][1] = 2.0f * (xy - wz);
    R.m[0][2] = 2.0f * (xz + wy);
    
    R.m[1][0] = 2.0f * (xy + wz);
    R.m[1][1] = 1.0f - 2.0f * (xx + zz);
    R.m[1][2] = 2.0f * (yz - wx);
    
    R.m[2][0] = 2.0f * (xz - wy);
    R.m[2][1] = 2.0f * (yz + wx);
    R.m[2][2] = 1.0f - 2.0f * (xx + yy);
    
    return R;
}

// Matrix multiplication helpers
__device__ __forceinline__ float3 transformPoint(const float4x4& m, const float3& p) {
    float4 p4 = make_float4(p.x, p.y, p.z, 1.0f);
    float4 result;
    result.x = m.m[0][0] * p4.x + m.m[0][1] * p4.y + m.m[0][2] * p4.z + m.m[0][3] * p4.w;
    result.y = m.m[1][0] * p4.x + m.m[1][1] * p4.y + m.m[1][2] * p4.z + m.m[1][3] * p4.w;
    result.z = m.m[2][0] * p4.x + m.m[2][1] * p4.y + m.m[2][2] * p4.z + m.m[2][3] * p4.w;
    result.w = m.m[3][0] * p4.x + m.m[3][1] * p4.y + m.m[3][2] * p4.z + m.m[3][3] * p4.w;
    
    if (result.w != 0.0f) {
        return make_float3(result.x / result.w, result.y / result.w, result.z / result.w);
    }
    return make_float3(result.x, result.y, result.z);
}

// Compute 3D covariance matrix from scale and rotation
__device__ __forceinline__ void compute3DCovariance(
    const float3& scale,
    const float4& rotation,
    float* cov3d  // Output: 6 values for symmetric 3x3 matrix
) {
    // Create scale matrix
    float3x3 S;
    S.m[0][0] = scale.x; S.m[0][1] = 0.0f;    S.m[0][2] = 0.0f;
    S.m[1][0] = 0.0f;    S.m[1][1] = scale.y; S.m[1][2] = 0.0f;
    S.m[2][0] = 0.0f;    S.m[2][1] = 0.0f;    S.m[2][2] = scale.z;
    
    // Convert quaternion to rotation matrix
    float3x3 R = quaternionToMatrix(rotation);
    
    // Compute covariance: Σ = R * S * S^T * R^T
    // Since S is diagonal: S * S^T = S^2
    float3x3 S2;
    S2.m[0][0] = scale.x * scale.x; S2.m[0][1] = 0.0f; S2.m[0][2] = 0.0f;
    S2.m[1][0] = 0.0f; S2.m[1][1] = scale.y * scale.y; S2.m[1][2] = 0.0f;
    S2.m[2][0] = 0.0f; S2.m[2][1] = 0.0f; S2.m[2][2] = scale.z * scale.z;
    
    // Compute R * S2 * R^T
    float3x3 temp, cov;
    
    // temp = R * S2
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            temp.m[i][j] = R.m[i][0] * S2.m[0][j] + 
                          R.m[i][1] * S2.m[1][j] + 
                          R.m[i][2] * S2.m[2][j];
        }
    }
    
    // cov = temp * R^T
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            cov.m[i][j] = temp.m[i][0] * R.m[j][0] + 
                         temp.m[i][1] * R.m[j][1] + 
                         temp.m[i][2] * R.m[j][2];
        }
    }
    
    // Store as symmetric matrix (only upper triangle)
    cov3d[0] = cov.m[0][0];
    cov3d[1] = cov.m[0][1];
    cov3d[2] = cov.m[0][2];
    cov3d[3] = cov.m[1][1];
    cov3d[4] = cov.m[1][2];
    cov3d[5] = cov.m[2][2];
}

// Spherical harmonics evaluation (simplified for degree 0 and 1)
__device__ __forceinline__ float3 evaluateSH(
    const float* sh_coeffs,
    const float3& view_dir
) {
    // Degree 0 (DC component)
    float3 color = make_float3(
        0.28209479177387814f * sh_coeffs[0],   // R
        0.28209479177387814f * sh_coeffs[16],  // G
        0.28209479177387814f * sh_coeffs[32]   // B
    );
    
    // Degree 1
    float x = view_dir.x;
    float y = view_dir.y;
    float z = view_dir.z;
    
    color.x += 0.4886025119029199f * y * sh_coeffs[1];
    color.x += 0.4886025119029199f * z * sh_coeffs[2];
    color.x += 0.4886025119029199f * x * sh_coeffs[3];
    
    color.y += 0.4886025119029199f * y * sh_coeffs[17];
    color.y += 0.4886025119029199f * z * sh_coeffs[18];
    color.y += 0.4886025119029199f * x * sh_coeffs[19];
    
    color.z += 0.4886025119029199f * y * sh_coeffs[33];
    color.z += 0.4886025119029199f * z * sh_coeffs[34];
    color.z += 0.4886025119029199f * x * sh_coeffs[35];
    
    // Clamp to positive values
    color.x = fmaxf(color.x, 0.0f);
    color.y = fmaxf(color.y, 0.0f);
    color.z = fmaxf(color.z, 0.0f);
    
    return color;
}

// Compute screen space bounding box for a 2D Gaussian
__device__ __forceinline__ void computeBoundingBox(
    const float2& center,
    float radius,
    int screen_width,
    int screen_height,
    int& min_x, int& max_x,
    int& min_y, int& max_y
) {
    min_x = max(0, (int)floorf((center.x - radius) / TILE_SIZE));
    max_x = min((screen_width + TILE_SIZE - 1) / TILE_SIZE - 1, 
                (int)ceilf((center.x + radius) / TILE_SIZE));
    
    min_y = max(0, (int)floorf((center.y - radius) / TILE_SIZE));
    max_y = min((screen_height + TILE_SIZE - 1) / TILE_SIZE - 1,
                (int)ceilf((center.y + radius) / TILE_SIZE));
}

} // namespace CUDA
} // namespace SplatRender