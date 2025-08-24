#!/usr/bin/env python3
"""
Convert a traditional mesh PLY to a simple Gaussian Splatting PLY
by placing Gaussians at vertex positions.
"""

import struct
import numpy as np
import sys

def read_mesh_ply(filename):
    """Read a traditional mesh PLY file and extract vertices and colors."""
    vertices = []
    colors = []
    
    with open(filename, 'rb') as f:
        # Read header
        header = []
        while True:
            line = f.readline().decode('utf-8').strip()
            header.append(line)
            if line == 'end_header':
                break
        
        # Parse header to find vertex count and properties
        num_vertices = 0
        has_color = False
        for line in header:
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            if 'property uchar red' in line:
                has_color = True
        
        print(f"Found {num_vertices} vertices")
        
        # Read binary data (assuming binary_little_endian)
        for i in range(num_vertices):
            # Read position (3 floats)
            x, y, z = struct.unpack('<fff', f.read(12))
            vertices.append([x, y, z])
            
            # Read color if present (3 unsigned chars)
            if has_color:
                r, g, b = struct.unpack('<BBB', f.read(3))
                colors.append([r/255.0, g/255.0, b/255.0])
            else:
                colors.append([0.5, 0.5, 0.5])  # Default gray
            
            # Skip face data or other properties
            # This is simplified - a full parser would handle all properties
            
            if i % 10000 == 0:
                print(f"  Read {i}/{num_vertices} vertices...")
    
    return np.array(vertices), np.array(colors)

def write_gaussian_ply(filename, positions, colors, scale=0.01, opacity=0.9):
    """Write a Gaussian Splatting PLY file."""
    num_gaussians = len(positions)
    
    # Prepare header
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {num_gaussians}",
        "property float x",
        "property float y", 
        "property float z",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
        "property float opacity"
    ]
    
    # Add f_rest properties (45 values for higher order SH)
    for i in range(45):
        header.append(f"property float f_rest_{i}")
    
    header.append("end_header")
    
    with open(filename, 'wb') as f:
        # Write header
        for line in header:
            f.write((line + '\n').encode('utf-8'))
        
        # Write binary data for each Gaussian
        for i in range(num_gaussians):
            # Position
            f.write(struct.pack('<fff', *positions[i]))
            
            # Scale (uniform scaling)
            f.write(struct.pack('<fff', scale, scale, scale))
            
            # Rotation quaternion (identity)
            f.write(struct.pack('<ffff', 1.0, 0.0, 0.0, 0.0))
            
            # DC spherical harmonics (color)
            # SH coefficient = color * C0 where C0 = 0.28209479177387814
            sh_scale = 3.5  # Brightness factor
            f.write(struct.pack('<fff', 
                colors[i][0] * sh_scale,
                colors[i][1] * sh_scale,
                colors[i][2] * sh_scale))
            
            # Opacity
            f.write(struct.pack('<f', opacity))
            
            # f_rest (45 zeros for higher order SH)
            for j in range(45):
                f.write(struct.pack('<f', 0.0))
            
            if i % 10000 == 0:
                print(f"  Wrote {i}/{num_gaussians} Gaussians...")
    
    print(f"Saved {num_gaussians} Gaussians to {filename}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python mesh_to_gaussian.py input.ply [output.ply] [scale] [sample_rate]")
        print("  scale: Gaussian size (default: 0.01)")
        print("  sample_rate: Use every Nth vertex (default: 1)")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.ply', '_gaussian.ply')
    scale = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
    sample_rate = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    
    print(f"Converting {input_file} to Gaussian splatting format...")
    print(f"Scale: {scale}, Sample rate: 1/{sample_rate}")
    
    # Read mesh
    vertices, colors = read_mesh_ply(input_file)
    
    # Sample vertices if requested
    if sample_rate > 1:
        vertices = vertices[::sample_rate]
        colors = colors[::sample_rate]
        print(f"Sampled to {len(vertices)} vertices")
    
    # Normalize positions to fit in view
    center = vertices.mean(axis=0)
    vertices -= center
    max_dist = np.abs(vertices).max()
    vertices /= max_dist
    vertices *= 2.0  # Scale to reasonable size
    
    # Write as Gaussians
    write_gaussian_ply(output_file, vertices, colors, scale=scale)
    
    print(f"\nConversion complete!")
    print(f"Run: ./build/SplatRender {output_file}")

if __name__ == "__main__":
    main()