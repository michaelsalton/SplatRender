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
        // Create a cube made of Gaussians
        std::cout << "Creating cube scene with Gaussians..." << std::endl;
        
        // Cube parameters
        float cube_size = 1.0f;  // Much smaller cube
        float half_size = cube_size / 2.0f;
        int points_per_edge = 5;  // Number of Gaussians per edge
        float spacing = cube_size / (points_per_edge - 1);
        
        // Template Gaussian
        SplatRender::Gaussian3D g;
        g.scale = glm::vec3(0.2f, 0.2f, 0.2f);  // Smaller Gaussians for the denser arrangement
        g.rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        g.opacity = 1.0f;
        
        // Initialize all SH coefficients to zero
        std::fill(g.sh_coeffs.begin(), g.sh_coeffs.end(), 0.0f);
        
        // Function to set color
        auto setColor = [&g](float r, float g_val, float b) {
            g.sh_coeffs[0] = r * 100.0f;    // R channel DC (brighter)
            g.sh_coeffs[15] = g_val * 100.0f;  // G channel DC (brighter)
            g.sh_coeffs[30] = b * 100.0f;   // B channel DC (brighter)
        };
        
        // Create 12 edges of the cube
        // Bottom face edges (red)
        for (int i = 0; i < points_per_edge; ++i) {
            float t = -half_size + i * spacing;
            setColor(1.0f, 0.2f, 0.2f);
            
            // Bottom edges
            g.position = glm::vec3(t, -half_size, -half_size); gaussians.push_back(g);
            g.position = glm::vec3(t, -half_size, half_size); gaussians.push_back(g);
            g.position = glm::vec3(-half_size, -half_size, t); gaussians.push_back(g);
            g.position = glm::vec3(half_size, -half_size, t); gaussians.push_back(g);
        }
        
        // Top face edges (green)
        for (int i = 0; i < points_per_edge; ++i) {
            float t = -half_size + i * spacing;
            setColor(0.2f, 1.0f, 0.2f);
            
            // Top edges
            g.position = glm::vec3(t, half_size, -half_size); gaussians.push_back(g);
            g.position = glm::vec3(t, half_size, half_size); gaussians.push_back(g);
            g.position = glm::vec3(-half_size, half_size, t); gaussians.push_back(g);
            g.position = glm::vec3(half_size, half_size, t); gaussians.push_back(g);
        }
        
        // Vertical edges (blue)
        for (int i = 1; i < points_per_edge - 1; ++i) {  // Skip corners (already have them)
            float t = -half_size + i * spacing;
            setColor(0.2f, 0.2f, 1.0f);
            
            // Vertical edges
            g.position = glm::vec3(-half_size, t, -half_size); gaussians.push_back(g);
            g.position = glm::vec3(half_size, t, -half_size); gaussians.push_back(g);
            g.position = glm::vec3(-half_size, t, half_size); gaussians.push_back(g);
            g.position = glm::vec3(half_size, t, half_size); gaussians.push_back(g);
        }
        
        // Add some Gaussians on the faces for better visibility
        // Front face (cyan)
        for (int i = 1; i < points_per_edge - 1; ++i) {
            for (int j = 1; j < points_per_edge - 1; ++j) {
                float x = -half_size + i * spacing;
                float y = -half_size + j * spacing;
                setColor(0.5f, 1.0f, 1.0f);
                g.position = glm::vec3(x, y, half_size);
                gaussians.push_back(g);
            }
        }
        
        // Back face (yellow)
        for (int i = 1; i < points_per_edge - 1; ++i) {
            for (int j = 1; j < points_per_edge - 1; ++j) {
                float x = -half_size + i * spacing;
                float y = -half_size + j * spacing;
                setColor(1.0f, 1.0f, 0.5f);
                g.position = glm::vec3(x, y, -half_size);
                gaussians.push_back(g);
            }
        }
        
        std::cout << "Created cube with " << gaussians.size() << " Gaussians" << std::endl;
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