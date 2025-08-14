#include <iostream>
#include "core/engine.h"

int main(int argc, char** argv) {
    std::cout << "SplatRender - 3D Gaussian Splatting Renderer" << std::endl;
    
    // Create and initialize engine
    SplatRender::Engine engine;
    
    if (!engine.initialize(1280, 720, "SplatRender")) {
        std::cerr << "Failed to initialize engine" << std::endl;
        return -1;
    }
    
    // Run the engine
    engine.run();
    
    // Shutdown is handled by destructor
    return 0;
}