"""
Convert .pli files to EPSG:4326 (WGS84) coordinate system

This script:
1. Reads all .pli files from the sea_boundary directory
2. Transforms coordinates from source CRS to EPSG:4326
3. Saves new .pli files with _4326 suffix
"""
import os
import glob
from pyproj import CRS, Transformer

def read_pli_file(pli_file):
    """
    Read a .pli file and return its components.
    Returns: (name, num_points, dimensions, coordinates_list)
    """
    with open(pli_file, 'r') as f:
        lines = f.readlines()
    
    # First line is the name
    name = lines[0].strip()
    
    # Second line is number of points and dimensions
    second_line = lines[1].strip().split()
    num_points = int(second_line[0])
    dimensions = int(second_line[1])
    
    # Remaining lines are coordinates
    coordinates = []
    for line in lines[2:]:
        line = line.strip()
        if line:
            parts = line.split()
            if len(parts) >= 2:
                x = float(parts[0])
                y = float(parts[1])
                # Check if there's a name at the end
                point_name = None
                if len(parts) >= 3:
                    point_name = parts[2]
                coordinates.append((x, y, point_name))
    
    return name, num_points, dimensions, coordinates

def transform_coordinates(coordinates, source_crs, target_crs=4326):
    """
    Transform coordinates from source CRS to target CRS using pyproj.
    
    Parameters:
    -----------
    coordinates : list of tuples
        List of (x, y, name) tuples
    source_crs : int
        Source coordinate reference system EPSG code (e.g., 28992)
    target_crs : int
        Target coordinate reference system EPSG code (default: 4326)
    
    Returns:
    --------
    list of tuples
        Transformed coordinates as (lon, lat, name) tuples
    """
    try:
        # Try using CRS.from_epsg() first
        source_crs_obj = CRS.from_epsg(source_crs)
        target_crs_obj = CRS.from_epsg(target_crs)
    except Exception:
        # Fallback: Use PROJ strings directly
        # EPSG:28992 - Amersfoort / RD New
        if source_crs == 28992:
            source_crs_obj = CRS.from_string("+proj=sterea +lat_0=52.1561605555556 +lon_0=5.38763888888889 +k=0.9999079 +x_0=155000 +y_0=463000 +ellps=bessel +towgs84=565.2369,50.0087,465.658,-0.406857330322398,0.350732676542563,-1.8703473836068,4.0812 +units=m +no_defs")
        else:
            source_crs_obj = CRS.from_epsg(source_crs)
        
        # EPSG:4326 - WGS84
        if target_crs == 4326:
            target_crs_obj = CRS.from_string("+proj=longlat +datum=WGS84 +no_defs")
        else:
            target_crs_obj = CRS.from_epsg(target_crs)
    
    # Create transformer
    transformer = Transformer.from_crs(source_crs_obj, target_crs_obj, always_xy=True)
    
    # Transform coordinates
    transformed_coords = []
    for x, y, name in coordinates:
        lon, lat = transformer.transform(x, y)
        transformed_coords.append((lon, lat, name))
    
    return transformed_coords

def write_pli_file(output_file, name, num_points, dimensions, coordinates):
    """
    Write a .pli file with the given structure.
    """
    with open(output_file, 'w') as f:
        # Write name
        f.write(f"{name}\n")
        
        # Write number of points and dimensions
        f.write(f" {num_points} {dimensions}\n")
        
        # Write coordinates
        for x, y, point_name in coordinates:
            if point_name:
                f.write(f" {x} {y} {point_name}\n")
            else:
                f.write(f" {x} {y}\n")

def main():
    # Define paths
    sea_boundary_dir = r"C:\Ocean\Work\Projects\2025\Schelde\Data\sea_boundary"
    
    # Source CRS - Dutch RD (EPSG:28992) based on coordinate values
    # Using integer EPSG code for better compatibility
    source_crs = 28992  # Amersfoort / RD New (Netherlands)
    target_crs = 4326    # WGS84
    
    print(f"Source CRS: {source_crs}")
    print(f"Target CRS: {target_crs}")
    print(f"\nReading .pli files from: {sea_boundary_dir}\n")
    
    # Find all .pli files
    pli_files = glob.glob(os.path.join(sea_boundary_dir, "*.pli"))
    
    if not pli_files:
        print("⚠ No .pli files found in the directory.")
        return
    
    print(f"Found {len(pli_files)} .pli file(s):")
    for pli_file in pli_files:
        print(f"  - {os.path.basename(pli_file)}")
    
    print("\nProcessing files...\n")
    
    for pli_file in pli_files:
        try:
            pli_name = os.path.basename(pli_file)
            print(f"Processing: {pli_name}")
            
            # Read the .pli file
            name, num_points, dimensions, coordinates = read_pli_file(pli_file)
            print(f"  Found {len(coordinates)} coordinate points")
            
            # Transform coordinates
            transformed_coords = transform_coordinates(coordinates, source_crs, target_crs)
            print(f"  Transformed coordinates to {target_crs}")
            
            # Create output filename
            base_name = os.path.splitext(pli_name)[0]
            output_file = os.path.join(sea_boundary_dir, f"{base_name}_4326.pli")
            
            # Write transformed .pli file
            write_pli_file(output_file, name, num_points, dimensions, transformed_coords)
            print(f"  ✓ Saved: {os.path.basename(output_file)}\n")
            
        except Exception as e:
            print(f"  ⚠ Error processing {pli_name}: {e}\n")
    
    print("✔ Processing complete!")

if __name__ == "__main__":
    main()

