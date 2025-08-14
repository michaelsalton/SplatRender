cmake_minimum_required(VERSION 3.18)

# Test executables
add_executable(test_gaussian test_gaussian.cpp)
add_executable(test_matrix_ops test_matrix_ops.cpp)
add_executable(test_spherical_harmonics test_spherical_harmonics.cpp)
add_executable(test_projection test_projection.cpp)
add_executable(test_camera test_camera.cpp)
add_executable(test_engine test_engine.cpp)
add_executable(test_display_integration test_display_integration.cpp)
add_executable(test_input test_input.cpp)

# Include directories
target_include_directories(test_gaussian PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
    ${GLM_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

target_include_directories(test_matrix_ops PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
    ${GLM_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

target_include_directories(test_spherical_harmonics PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
    ${GLM_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

target_include_directories(test_projection PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
    ${GLM_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

target_include_directories(test_camera PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
    ${GLM_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

target_include_directories(test_engine PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
    ${GLM_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

target_include_directories(test_display_integration PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
    ${GLM_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

target_include_directories(test_input PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
    ${GLM_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

# Link math libraries
target_link_libraries(test_gaussian PRIVATE
    ${CMAKE_BINARY_DIR}/CMakeFiles/SplatRender.dir/src/math/gaussian.cpp.o
    ${CMAKE_BINARY_DIR}/CMakeFiles/SplatRender.dir/src/math/spherical_harmonics.cpp.o
    ${CMAKE_BINARY_DIR}/CMakeFiles/SplatRender.dir/src/math/matrix_ops.cpp.o
)

target_link_libraries(test_matrix_ops PRIVATE
    ${CMAKE_BINARY_DIR}/CMakeFiles/SplatRender.dir/src/math/matrix_ops.cpp.o
)

target_link_libraries(test_spherical_harmonics PRIVATE
    ${CMAKE_BINARY_DIR}/CMakeFiles/SplatRender.dir/src/math/spherical_harmonics.cpp.o
)

target_link_libraries(test_projection PRIVATE
    ${CMAKE_BINARY_DIR}/CMakeFiles/SplatRender.dir/src/math/gaussian.cpp.o
    ${CMAKE_BINARY_DIR}/CMakeFiles/SplatRender.dir/src/math/matrix_ops.cpp.o
    ${CMAKE_BINARY_DIR}/CMakeFiles/SplatRender.dir/src/math/spherical_harmonics.cpp.o
)

target_link_libraries(test_camera PRIVATE
    ${CMAKE_BINARY_DIR}/CMakeFiles/SplatRender.dir/src/core/camera.cpp.o
)

target_link_libraries(test_engine PRIVATE
    # No dependencies needed for engine tests
)

target_link_libraries(test_display_integration PRIVATE
    ${CMAKE_BINARY_DIR}/CMakeFiles/SplatRender.dir/src/renderer/opengl_display.cpp.o
    ${OPENGL_LIBRARIES}
    glfw
    GLEW::GLEW
)

target_link_libraries(test_input PRIVATE
    ${CMAKE_BINARY_DIR}/CMakeFiles/SplatRender.dir/src/core/input.cpp.o
    ${CMAKE_BINARY_DIR}/CMakeFiles/SplatRender.dir/src/core/camera.cpp.o
    glfw
)

# Add tests
add_test(NAME GaussianTests COMMAND test_gaussian)
add_test(NAME MatrixOpsTests COMMAND test_matrix_ops)
add_test(NAME SphericalHarmonicsTests COMMAND test_spherical_harmonics)
add_test(NAME ProjectionTests COMMAND test_projection)
add_test(NAME CameraTests COMMAND test_camera)
add_test(NAME EngineTests COMMAND test_engine)
add_test(NAME DisplayIntegrationTests COMMAND test_display_integration)
add_test(NAME InputTests COMMAND test_input)

# Create a custom target to run all tests
add_custom_target(tests
    COMMAND test_gaussian
    COMMAND test_matrix_ops
    COMMAND test_spherical_harmonics
    COMMAND test_projection
    COMMAND test_camera
    COMMAND test_engine
    COMMAND test_display_integration
    COMMAND test_input
    DEPENDS test_gaussian test_matrix_ops test_spherical_harmonics test_projection test_camera test_engine test_display_integration test_input
    COMMENT "Running all unit tests..."
)