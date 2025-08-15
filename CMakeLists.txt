cmake_minimum_required(VERSION 3.18)
project(SplatRender LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Platform detection and configuration
if(APPLE)
    message(STATUS "Building for macOS - CUDA disabled")
    set(USE_CUDA OFF)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")
elseif(UNIX AND NOT APPLE)
    message(STATUS "Building for Linux - CUDA enabled")
    # Check for CUDA
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(USE_CUDA ON)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        find_package(CUDAToolkit REQUIRED)
    else()
        message(WARNING "CUDA not found - building CPU-only version")
        set(USE_CUDA OFF)
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")
endif()

# Find packages
find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(Threads REQUIRED)

# Find GLEW
find_package(GLEW REQUIRED)

# Find Eigen3
find_package(Eigen3 3.4 REQUIRED)

# GLM - header only library
find_path(GLM_INCLUDE_DIR glm/glm.hpp
    PATHS
    /opt/homebrew/include
    /usr/local/include
    /usr/include
)

if(NOT GLM_INCLUDE_DIR)
    message(FATAL_ERROR "GLM not found!")
else()
    message(STATUS "Found GLM: ${GLM_INCLUDE_DIR}")
endif()

# Source files
set(CORE_SOURCES
    src/core/engine.cpp
    src/core/camera.cpp
    src/core/input.cpp
)

set(MATH_SOURCES
    src/math/gaussian.cpp
    src/math/matrix_ops.cpp
    src/math/spherical_harmonics.cpp
)

set(IO_SOURCES
    src/io/ply_loader.cpp
)

set(RENDERER_SOURCES
    src/renderer/cpu_rasterizer.cpp
    src/renderer/opengl_display.cpp
    src/renderer/text_renderer.cpp
    src/renderer/axis_renderer.cpp
)

set(ALL_SOURCES
    ${CORE_SOURCES}
    ${MATH_SOURCES}
    ${IO_SOURCES}
    ${RENDERER_SOURCES}
    src/main.cpp
)

# CUDA sources (Linux only)
if(USE_CUDA)
    set(CUDA_SOURCES
        src/cuda/memory_manager.cu
        src/cuda/cuda_utils.cu
        src/cuda/kernels/projection.cu
        src/cuda/kernels/tiling.cu
        src/cuda/kernels/rasterization.cu
    )
    
    # Set CUDA architecture (adjust based on your GPU)
    set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES 70 75 80 86)
endif()

# Create executable
if(USE_CUDA)
    add_executable(${PROJECT_NAME} ${ALL_SOURCES} ${CUDA_SOURCES})
    
    # CUDA specific settings
    set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --use_fast_math
            --relocatable-device-code=true
            -lineinfo
        >
    )
else()
    add_executable(${PROJECT_NAME} ${ALL_SOURCES})
endif()

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${OPENGL_INCLUDE_DIRS}
)

# Eigen include
target_include_directories(${PROJECT_NAME} PRIVATE ${EIGEN3_INCLUDE_DIR})

# GLM include
target_include_directories(${PROJECT_NAME} PRIVATE ${GLM_INCLUDE_DIR})

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OPENGL_LIBRARIES}
    glfw
    GLEW::GLEW
    Threads::Threads
)

# Platform-specific libraries
if(APPLE)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        "-framework OpenGL"
        "-framework Cocoa"
        "-framework IOKit"
        "-framework CoreVideo"
    )
elseif(UNIX)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        GL
        X11
        pthread
        dl
    )
    
    if(USE_CUDA)
        target_link_libraries(${PROJECT_NAME} PRIVATE
            CUDA::cudart
            CUDA::cuda_driver
        )
    endif()
endif()

# Compiler definitions
if(USE_CUDA)
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_CUDA)
endif()

# Copy shaders to build directory
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/shaders $<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders
)

# Tests
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    # Only add tests subdirectory if it has a CMakeLists.txt
    if(EXISTS ${CMAKE_SOURCE_DIR}/tests/CMakeLists.txt)
        add_subdirectory(tests)
    endif()
endif()

# Installation
install(TARGETS ${PROJECT_NAME} DESTINATION bin)
install(DIRECTORY shaders DESTINATION share/${PROJECT_NAME})
install(DIRECTORY data DESTINATION share/${PROJECT_NAME})

# Print configuration summary
message(STATUS "")
message(STATUS "SplatRender Configuration Summary:")
message(STATUS "  Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "  C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA Support: ${USE_CUDA}")
if(USE_CUDA)
    message(STATUS "  CUDA Version: ${CUDAToolkit_VERSION}")
    message(STATUS "  CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
endif()
message(STATUS "")