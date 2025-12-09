"""
Read and combine .bc files from a directory into a single .bc file.

This script reads all .bc files from a specified directory and combines them
into a single .bc file in the NL_waq format where:
- Time is the first column (same for all files)
- All quantities are listed in the header
- Data rows contain time first, followed by all quantity values
"""

import os
import re
from pathlib import Path


def parse_bc_file(filepath):
    """
    Parse a .bc file and extract time, quantity name, and values.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the .bc file
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'time': list of time values
        - 'quantity': quantity name (e.g., 'tracerbndDOC')
        - 'values': list of values
        - 'unit_time': unit for time
        - 'unit_quantity': unit for quantity
        - 'name': name from the file (e.g., 'NW_0001')
        - 'function': function type
        - 'time_interpolation': interpolation method
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract name
    name_match = re.search(r'Name\s*=\s*([^\n]+)', content, re.IGNORECASE)
    name = name_match.group(1).strip() if name_match else None
    
    # Extract function
    function_match = re.search(r'Function\s*=\s*([^\n]+)', content, re.IGNORECASE)
    function = function_match.group(1).strip() if function_match else 'timeseries'
    
    # Extract time interpolation
    interp_match = re.search(r'Time-interpolation\s*=\s*([^\n]+)', content, re.IGNORECASE)
    time_interpolation = interp_match.group(1).strip() if interp_match else 'linear'
    
    # Find the second Quantity line (the one that's not "time")
    quantity_lines = re.findall(r'Quantity\s*=\s*(\w+)', content, re.IGNORECASE)
    quantity = quantity_lines[1] if len(quantity_lines) > 1 else None
    
    # Extract units
    unit_matches = re.findall(r'Unit\s*=\s*([^\n]+)', content, re.IGNORECASE)
    unit_time = unit_matches[0].strip() if len(unit_matches) > 0 else None
    unit_quantity = unit_matches[1].strip() if len(unit_matches) > 1 else None
    
    # Extract data section (lines with numeric data)
    lines = content.split('\n')
    data_lines = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('[') or line.startswith('Name') or \
           line.startswith('Function') or line.startswith('Time-interpolation') or \
           line.startswith('Quantity') or line.startswith('Unit'):
            continue
        
        # Check if line contains numeric data (scientific notation or regular numbers)
        if re.match(r'^[\d\.\+\-eE\s]+$', line):
            data_lines.append(line)
    
    # Parse data lines
    time_values = []
    values = []
    
    for line in data_lines:
        parts = line.split()
        if len(parts) >= 2:
            try:
                time_val = float(parts[0])
                value = float(parts[1])
                time_values.append(time_val)
                values.append(value)
            except ValueError:
                continue
    
    return {
        'time': time_values,
        'quantity': quantity,
        'values': values,
        'unit_time': unit_time,
        'unit_quantity': unit_quantity,
        'name': name,
        'function': function,
        'time_interpolation': time_interpolation
    }


def read_bc_files(directory):
    """
    Read all .bc files from a directory and combine them.
    
    Parameters
    ----------
    directory : str or Path
        Path to directory containing .bc files
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'time': list of time values
        - 'quantities': dict mapping quantity names to lists of values
        - 'units': dict mapping quantity names to units
        - 'metadata': dict with name, function, time_interpolation, unit_time
    """
    directory = Path(directory)
    
    # Find all .bc files
    bc_files = list(directory.glob('*.bc'))
    
    if not bc_files:
        raise ValueError(f"No .bc files found in {directory}")
    
    print(f"Found {len(bc_files)} .bc files")
    
    # Parse all files
    quantities = {}
    units = {}
    time_column = None
    metadata = {}
    
    for bc_file in sorted(bc_files):
        print(f"Reading: {bc_file.name}")
        parsed = parse_bc_file(bc_file)
        
        if parsed['quantity'] is None:
            print(f"  Warning: Could not extract quantity name from {bc_file.name}, skipping")
            continue
        
        # Store time column from first file (should be same for all)
        if time_column is None:
            time_column = parsed['time']
            metadata = {
                'name': parsed['name'],
                'function': parsed['function'],
                'time_interpolation': parsed['time_interpolation'],
                'unit_time': parsed['unit_time']
            }
        else:
            # Verify time column is the same
            if parsed['time'] != time_column:
                print(f"  Warning: Time column differs in {bc_file.name}")
        
        # Store values and units with quantity name as key
        quantities[parsed['quantity']] = parsed['values']
        units[parsed['quantity']] = parsed['unit_quantity']
    
    print(f"\nFound {len(quantities)} quantities")
    print(f"Time unit: {metadata['unit_time']}")
    print(f"Quantities: {', '.join(sorted(quantities.keys()))}")
    
    return {
        'time': time_column,
        'quantities': quantities,
        'units': units,
        'metadata': metadata
    }


def write_combined_bc_file(data, output_path, name=None):
    """
    Write a combined .bc file in NL_waq format.
    
    Parameters
    ----------
    data : dict
        Dictionary returned by read_bc_files()
    output_path : str or Path
        Path to output .bc file
    name : str, optional
        Name to use in the output file (defaults to name from first file)
    """
    output_path = Path(output_path)
    
    time_column = data['time']
    quantities = data['quantities']
    units = data['units']
    metadata = data['metadata']
    
    # Use provided name or name from metadata
    output_name = name if name else metadata.get('name', 'Combined')
    
    # Format time unit to uppercase (like in NL_waq format)
    unit_time = metadata['unit_time'].upper() if metadata['unit_time'] else 'MINUTES SINCE 2017-12-01'
    
    with open(output_path, 'w') as f:
        # Write [General] section
        f.write("[General]\n")
        f.write("fileVersion = 1.01\n")
        f.write("fileType    = boundConds\n")
        f.write("\n")
        
        # Write [Forcing] section
        f.write("[Forcing]\n")
        f.write(f"name = {output_name}\n")
        f.write(f"function = {metadata.get('function', 'timeseries')}\n")
        f.write(f"timeInterpolation = {metadata.get('time_interpolation', 'linear')}\n")
        f.write("offset = 0.0\n")
        f.write("factor = 1.0\n")
        
        # Write time quantity
        f.write("quantity = time\n")
        f.write(f"unit = {unit_time}\n")
        
        # Write all other quantities in sorted order
        for quantity in sorted(quantities.keys()):
            f.write(f"quantity = {quantity}\n")
            f.write(f"unit = {units[quantity]}\n")
        
        # Write data rows
        for i, time_val in enumerate(time_column):
            # Format time in scientific notation with 6 decimal places
            time_str = f"{time_val:.6e}"
            
            # Get all values for this time step
            values = []
            for quantity in sorted(quantities.keys()):
                if i < len(quantities[quantity]):
                    val = quantities[quantity][i]
                    # Format value with 6 decimal places (matching NL_waq format)
                    values.append(f"{val:.6f}")
                else:
                    values.append("0.000000")
            
            # Write line: time followed by all values
            f.write(f"{time_str} {' '.join(values)}\n")
    
    print(f"\n✓ Saved combined .bc file to: {output_path}")


def main():
    """Main function to read and combine .bc files."""
    # Define the directory path
    bc_directory = r"P:\ltv-natuur-schelde-slib-waq\ecolmod\02_preprocessing\boundary_conditions\freshwater_boundary\results"
    
    # Read and combine all .bc files
    data = read_bc_files(bc_directory)
    
    # Write combined .bc file
    output_bc = os.path.join(bc_directory, "combined.bc")
    write_combined_bc_file(data, output_bc)
    
    return data


if __name__ == "__main__":
    data = main()

