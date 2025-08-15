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
        g.scale = glm::vec3(5.0f, 5.0f, 5.0f);  // Much larger scale
        g.rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        g.opacity = 1.0f;
        // Set red color in SH coefficients
        g.sh_coeffs[0] = 3.0f;   // R channel DC (brighter)
        g.sh_coeffs[15] = 0.0f;  // G channel DC
        g.sh_coeffs[30] = 0.0f;  // B channel DC
        gaussians.push_back(g);
        
        // Add a green Gaussian
        g.position = glm::vec3(3.0f, 0.0f, 0.0f);
        g.sh_coeffs[0] = 0.0f;   // R channel DC
        g.sh_coeffs[15] = 3.0f;  // G channel DC (brighter)
        g.sh_coeffs[30] = 0.0f;  // B channel DC
        gaussians.push_back(g);
        
        // Add a blue Gaussian
        g.position = glm::vec3(-3.0f, 0.0f, 0.0f);
        g.sh_coeffs[0] = 0.0f;   // R channel DC
        g.sh_coeffs[15] = 0.0f;  // G channel DC
        g.sh_coeffs[30] = 3.0f;  // B channel DC (brighter)
        gaussians.push_back(g);
        
        std::cout << "Created test scene with 3 colored Gaussians at:" << std::endl;
        std::cout << "  Red at (0, 0, 0)" << std::endl;
        std::cout << "  Green at (3, 0, 0)" << std::endl;
        std::cout << "  Blue at (-3, 0, 0)" << std::endl;
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