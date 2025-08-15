#include <iostream>
#include <string>
#include "core/engine.h"
#include "io/ply_loader.h"
#include "renderer/cpu_rasterizer.h"

int main(int argc, char** argv) {
    std::cout << "SplatRender - 3D Gaussian Splatting Renderer" << std::endl;
    
    // Parse command line arguments
    std::string ply_file;
    if (argc > 1) {
        ply_file = argv[1];
        std::cout << "Loading PLY file: " << ply_file << std::endl;
    } else {
        std::cout << "Usage: " << argv[0] << " [ply_file]" << std::endl;
        std::cout << "Running with test scene..." << std::endl;
    }
    
    // Load Gaussians if PLY file provided
    std::vector<SplatRender::Gaussian3D> gaussians;
    if (!ply_file.empty()) {
        SplatRender::PLYLoader loader;
        if (!loader.load(ply_file, gaussians)) {
            std::cerr << "Failed to load PLY file: " << loader.getLastError() << std::endl;
            return -1;
        }
        std::cout << "Loaded " << gaussians.size() << " Gaussians" << std::endl;
    } else {
        // Create a simple test scene with larger, more visible Gaussians
        SplatRender::Gaussian3D g;
        g.position = glm::vec3(0.0f, 0.0f, 0.0f);
        g.scale = glm::vec3(2.0f, 2.0f, 2.0f);  // Medium size
        g.rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        g.opacity = 1.0f;
        // Set extremely bright white color in SH coefficients
        g.sh_coeffs[0] = 100.0f;   // R channel DC (extremely bright)
        g.sh_coeffs[15] = 100.0f;  // G channel DC (extremely bright)
        g.sh_coeffs[30] = 100.0f;  // B channel DC (extremely bright)
        gaussians.push_back(g);
        
        // Add a white Gaussian on the right
        g.position = glm::vec3(3.0f, 0.0f, 0.0f);
        g.scale = glm::vec3(2.0f, 2.0f, 2.0f);  // Medium size
        g.sh_coeffs[0] = 100.0f;   // R channel DC (extremely bright)
        g.sh_coeffs[15] = 100.0f;  // G channel DC (extremely bright)
        g.sh_coeffs[30] = 100.0f;  // B channel DC (extremely bright)
        gaussians.push_back(g);
        
        // Add a white Gaussian on the left
        g.position = glm::vec3(-3.0f, 0.0f, 0.0f);
        g.scale = glm::vec3(2.0f, 2.0f, 2.0f);  // Medium size
        g.sh_coeffs[0] = 100.0f;   // R channel DC (extremely bright)
        g.sh_coeffs[15] = 100.0f;  // G channel DC (extremely bright)
        g.sh_coeffs[30] = 100.0f;  // B channel DC (extremely bright)
        gaussians.push_back(g);
        
        std::cout << "Created test scene with 3 white Gaussians at:" << std::endl;
        std::cout << "  White at (0, 0, 0)" << std::endl;
        std::cout << "  White at (3, 0, 0)" << std::endl;
        std::cout << "  White at (-3, 0, 0)" << std::endl;
    }
    
    // Create and initialize engine
    SplatRender::Engine engine;
    engine.setGaussians(gaussians);
    
    if (!engine.initialize(1280, 720, "SplatRender")) {
        std::cerr << "Failed to initialize engine" << std::endl;
        return -1;
    }
    
    // Run the engine
    engine.run();
    
    // Shutdown is handled by destructor
    return 0;
}