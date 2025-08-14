#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <functional>
#include <glm/glm.hpp>

namespace SplatRender {
namespace Test {

// Test result tracking
struct TestResult {
    std::string test_name;
    bool passed;
    std::string error_message;
};

class TestFramework {
public:
    static TestFramework& getInstance() {
        static TestFramework instance;
        return instance;
    }
    
    void runTest(const std::string& name, std::function<void()> test_func) {
        std::cout << "Running: " << name << " ... ";
        TestResult result{name, true, ""};
        
        try {
            test_func();
            std::cout << "PASSED" << std::endl;
        } catch (const std::exception& e) {
            result.passed = false;
            result.error_message = e.what();
            std::cout << "FAILED: " << e.what() << std::endl;
        }
        
        results_.push_back(result);
    }
    
    void printSummary() {
        int passed = 0;
        int failed = 0;
        
        for (const auto& result : results_) {
            if (result.passed) passed++;
            else failed++;
        }
        
        std::cout << "\n========== TEST SUMMARY ==========" << std::endl;
        std::cout << "Total tests: " << results_.size() << std::endl;
        std::cout << "Passed: " << passed << std::endl;
        std::cout << "Failed: " << failed << std::endl;
        
        if (failed > 0) {
            std::cout << "\nFailed tests:" << std::endl;
            for (const auto& result : results_) {
                if (!result.passed) {
                    std::cout << "  - " << result.test_name << ": " << result.error_message << std::endl;
                }
            }
        }
        std::cout << "==================================" << std::endl;
    }
    
    void clear() {
        results_.clear();
    }

private:
    std::vector<TestResult> results_;
};

// Assertion helpers
#define ASSERT_TRUE(condition) \
    if (!(condition)) { \
        throw std::runtime_error("Assertion failed: " #condition); \
    }

#define ASSERT_FALSE(condition) \
    if (condition) { \
        throw std::runtime_error("Assertion failed: expected false for " #condition); \
    }

#define ASSERT_EQUAL(expected, actual) \
    if ((expected) != (actual)) { \
        throw std::runtime_error("Assertion failed: expected " + std::to_string(expected) + " but got " + std::to_string(actual)); \
    }

#define ASSERT_NEAR(expected, actual, tolerance) \
    if (std::abs((expected) - (actual)) > (tolerance)) { \
        throw std::runtime_error("Assertion failed: expected " + std::to_string(expected) + " but got " + std::to_string(actual) + " (tolerance: " + std::to_string(tolerance) + ")"); \
    }

// Vector/Matrix comparison helpers
inline bool isNear(float a, float b, float tolerance = 1e-6f) {
    return std::abs(a - b) <= tolerance;
}

inline bool isNear(const glm::vec2& a, const glm::vec2& b, float tolerance = 1e-6f) {
    return isNear(a.x, b.x, tolerance) && isNear(a.y, b.y, tolerance);
}

inline bool isNear(const glm::vec3& a, const glm::vec3& b, float tolerance = 1e-6f) {
    return isNear(a.x, b.x, tolerance) && isNear(a.y, b.y, tolerance) && isNear(a.z, b.z, tolerance);
}

inline bool isNear(const glm::mat2& a, const glm::mat2& b, float tolerance = 1e-6f) {
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            if (!isNear(a[i][j], b[i][j], tolerance)) return false;
        }
    }
    return true;
}

inline bool isNear(const glm::mat3& a, const glm::mat3& b, float tolerance = 1e-6f) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (!isNear(a[i][j], b[i][j], tolerance)) return false;
        }
    }
    return true;
}

#define ASSERT_VEC_NEAR(expected, actual, tolerance) \
    if (!isNear(expected, actual, tolerance)) { \
        throw std::runtime_error("Vector assertion failed"); \
    }

#define ASSERT_MAT_NEAR(expected, actual, tolerance) \
    if (!isNear(expected, actual, tolerance)) { \
        throw std::runtime_error("Matrix assertion failed"); \
    }

// Test running macro
#define RUN_TEST(test_func) \
    TestFramework::getInstance().runTest(#test_func, test_func)

} // namespace Test
} // namespace SplatRender