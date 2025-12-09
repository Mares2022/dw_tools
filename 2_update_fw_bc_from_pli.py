"""
Update .bc files with names from .pli files and export time series to .txt files

This script:
1. Reads all .pli files in the results directory
2. Extracts boundary names from .pli files
3. Updates the Name attribute in .bc files with these names (adding _0001)
4. Saves the time series data to .txt files named after the substance
"""
import os
import re
import glob
from dwq_utilities import get_all_pli_names, create_txt_file_with_all_boundaries, update_bc_file

def main():
    # Define paths
    results_dir = r"P:\ltv-natuur-schelde-slib-waq\ecolmod\02_preprocessing\boundary_conditions\freshwater_boundary\results"
    
    # Get all names from .pli files
    pli_names = get_all_pli_names(results_dir)
    
    if not pli_names:
        print("⚠ No names found in .pli files. Exiting.")
        return
    
    print(f"\nUsing names: {pli_names[:5]}..." if len(pli_names) > 5 else f"\nUsing names: {pli_names}")
    
    # Find all .bc files in the directory
    bc_files = glob.glob(os.path.join(results_dir, "*.bc"))
    
    # Filter out combined.bc if it exists (might be a special file)
    bc_files = [f for f in bc_files if not os.path.basename(f).lower() == 'combined.bc']
    
    print(f"\nProcessing {len(bc_files)} .bc files...")
    
    for bc_file in bc_files:
        bc_name = os.path.basename(bc_file)
        print(f"\nProcessing: {bc_name}")
        # Create .txt file with all boundaries
        create_txt_file_with_all_boundaries(bc_file, pli_names, results_dir)
        # Update .bc file with first boundary name
        update_bc_file(bc_file, pli_names, results_dir)
    
    print("\n✔ Processing complete!")

if __name__ == "__main__":
    main()

