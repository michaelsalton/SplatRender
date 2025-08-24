#!/usr/bin/env python3
"""
Simple converter to create a Gaussian splatting PLY from a mesh PLY
by sampling vertices.
"""

import numpy as np
import sys

def create_test_gaussian_goat(output_file, num_gaussians=10000):
    """Create a simple test Gaussian goat in a grid pattern."""
    
    # Create a goat-like shape with Gaussians
    points = []
    colors = []
    
    # Body (ellipsoid)
    for i in range(num_gaussians // 2):
        # Random points in ellipsoid
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, np.pi)
        r = np.random.uniform(0.5, 1.0)
        
        x = r * 2.0 * np.sin(phi) * np.cos(theta)  # Elongated body
        y = r * 1.0 * np.sin(phi) * np.sin(theta)
        z = r * 1.2 * np.cos(phi)
        
        points.append([x, y, z])
        # Brown/tan color for body
        colors.append([0.7, 0.5, 0.3])
    
    # Head (sphere at front)
    for i in range(num_gaussians // 4):
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, np.pi)
        r = np.random.uniform(0.3, 0.6)
        
        x = 2.5 + r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = 0.5 + r * np.cos(phi)
        
        points.append([x, y, z])
        # Lighter color for head
        colors.append([0.8, 0.6, 0.4])
    
    # Horns (two cones)
    for horn in [1, -1]:
        for i in range(num_gaussians // 8):
            t = np.random.uniform(0, 1)
            theta = np.random.uniform(0, 2*np.pi)
            r = 0.1 * (1 - t)  # Cone shape
            
            x = 2.8 + t * 0.5
            y = horn * (0.3 + r * np.cos(theta))
            z = 1.0 + t * 0.8 + r * np.sin(theta)
            
            points.append([x, y, z])
            # Dark gray for horns
            colors.append([0.3, 0.3, 0.3])
    
    points = np.array(points)
    colors = np.array(colors)
    
    # Write PLY file
    with open(output_file, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float scale_0\n")
        f.write("property float scale_1\n")
        f.write("property float scale_2\n")
        f.write("property float rot_0\n")
        f.write("property float rot_1\n")
        f.write("property float rot_2\n")
        f.write("property float rot_3\n")
        f.write("property float f_dc_0\n")
        f.write("property float f_dc_1\n")
        f.write("property float f_dc_2\n")
        f.write("property float opacity\n")
        
        # f_rest properties
        for i in range(45):
            f.write(f"property float f_rest_{i}\n")
        
        f.write("end_header\n")
        
        # Data
        for i in range(len(points)):
            # Position
            f.write(f"{points[i][0]:.6f} {points[i][1]:.6f} {points[i][2]:.6f} ")
            
            # Scale (small Gaussians)
            scale = 0.05
            f.write(f"{scale:.6f} {scale:.6f} {scale:.6f} ")
            
            # Rotation (identity quaternion)
            f.write("1.0 0.0 0.0 0.0 ")
            
            # Color (DC spherical harmonics)
            sh_scale = 2.0
            f.write(f"{colors[i][0]*sh_scale:.6f} {colors[i][1]*sh_scale:.6f} {colors[i][2]*sh_scale:.6f} ")
            
            # Opacity
            f.write("0.9 ")
            
            # f_rest (45 zeros)
            f.write(" ".join(["0"] * 45))
            f.write("\n")
    
    print(f"Created Gaussian goat with {len(points)} Gaussians")
    print(f"Saved to: {output_file}")

def main():
    output_file = "data/goat_gaussian.ply"
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    
    if len(sys.argv) > 2:
        num_gaussians = int(sys.argv[2])
    else:
        num_gaussians = 10000
    
    create_test_gaussian_goat(output_file, num_gaussians)
    print(f"\nRun: ./build/SplatRender {output_file}")

if __name__ == "__main__":
    main()