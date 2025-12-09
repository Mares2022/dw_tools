"""
Freshwater Boundaries Data Processing Script

This script processes freshwater boundary data and generates .bc files
for Delwaq boundary conditions.
"""
import os
import pandas as pd
from dwq_utilities import process_time, process_value, get_bc_file

def main():
    # Define paths
    data_dir = r"P:\ltv-natuur-schelde-slib-waq\ecolmod\02_preprocessing\boundary_conditions\freshwater_boundary\results"
    data_path = os.path.join(data_dir, "freshwaterboundaries.csv")
    metadata_path = os.path.join(data_dir, "substances_attributes_modified.csv")

    # Read the data file
    df = pd.read_csv(data_path, sep=";")

    # Read substances attributes from metadata file
    substances_attributes = pd.read_csv(metadata_path, sep=';')

    # First pass: Find a reference substance with valid data to establish time series structure
    reference_time_series = None
    print("Establishing reference time series...")
    for _, attrs in substances_attributes.iterrows():
        substance = attrs["substance"]
        df_substance = df[df['name'] == substance].copy()
        
        if df_substance.empty:
            continue
            
        # Check if datetime column exists
        if 'datetime' not in df_substance.columns:
            if 'datetime' in df_substance.index.names or df_substance.index.name == 'datetime':
                df_substance = df_substance.reset_index()
            else:
                continue
        
        # Filter out null datetimes
        df_substance = df_substance[df_substance['datetime'].notna()].copy()
        if df_substance.empty:
            continue
        
        # Process time to get the time series structure
        df_substance = process_time(df_substance, 'datetime', '2017-12-01', 'minutes', 'time_since', '2018-01-01', '2019-01-01')
        
        if not df_substance.empty:
            reference_time_series = df_substance[['time_since']].copy()
            print(f"  ✓ Reference time series established from substance '{substance}' ({len(reference_time_series)} time steps)")
            break
    
    if reference_time_series is None:
        print("  ⚠ Error: Could not establish reference time series. No substances with valid datetime data found.")
        return

    # Second pass: Process each substance
    for _, attrs in substances_attributes.iterrows():
        # Read the attributes from the metadata file
        delwaq_id = attrs["delwaq"]
        substance = attrs["substance"]
        factor = attrs["factor"]
        value_col = attrs["value"]
        name = attrs["name"]
        function = attrs["function"]
        time_interpolation = attrs["time_interpolation"]
        quantity_time = attrs["quantity_time"]
        unit_time = attrs["unit_time"]
        quantity_substance = attrs["quantity_substance"]
        unit_substance = attrs["unit_substance"]
        
        print(f"Processing: {delwaq_id}, {substance}, factor={factor}")

        # Get data for substance of interest
        df_substance = df[df['name'] == substance].copy()
        
        if df_substance.empty:
            print(f"  ⚠ Warning: No data found for substance '{substance}'.")
            print(f"  Using reference time series with zero values to maintain time series frequency.")
            # Use reference time series with zero values
            df_substance = reference_time_series.copy()
            df_substance['value'] = '0.00000'
        else:
            print(f"  Found {len(df_substance)} rows before processing")

            # Check if datetime column exists (might be in index)
            has_datetime = False
            if 'datetime' not in df_substance.columns:
                # Check if datetime is in the index
                if 'datetime' in df_substance.index.names or df_substance.index.name == 'datetime':
                    df_substance = df_substance.reset_index()
                    has_datetime = True
                else:
                    print(f"  ⚠ Warning: 'datetime' column not found. Using reference time series with zero values.")
                    df_substance = reference_time_series.copy()
                    df_substance['value'] = '0.00000'
                    has_datetime = False
            else:
                has_datetime = True
            
            if has_datetime:
                # Handle null values in datetime: drop rows with null datetime
                datetime_null_count = df_substance['datetime'].isna().sum()
                if datetime_null_count > 0:
                    print(f"  ⚠ Warning: {datetime_null_count} rows have null datetime values. These will be dropped.")
                    df_substance = df_substance[df_substance['datetime'].notna()].copy()
                
                if df_substance.empty:
                    print(f"  ⚠ Warning: No data with valid datetime for substance '{substance}'.")
                    print(f"  Using reference time series with zero values to maintain time series frequency.")
                    df_substance = reference_time_series.copy()
                    df_substance['value'] = '0.00000'
                else:
                    # Handle null values in value column: set to 0 and keep the same time series frequency
                    # Fill nulls in the value column before processing
                    if value_col in df_substance.columns:
                        df_substance[value_col] = df_substance[value_col].fillna(0)
                    else:
                        print(f"  ⚠ Warning: Column '{value_col}' not found. Using reference time series with zero values.")
                        df_substance = reference_time_series.copy()
                        df_substance['value'] = '0.00000'
                    # Continue to process_time if we have valid data

        # Check if we already have time_since (using reference time series)
        if 'time_since' in df_substance.columns and 'value' in df_substance.columns:
            # Already using reference time series with zero values, skip processing
            print(f"  Using reference time series structure ({len(df_substance)} time steps)")
        else:
            # Format the time 
            df_substance = process_time(df_substance, 'datetime', '2017-12-01', 'minutes', 'time_since', '2018-01-01', '2019-01-01')
            
            if df_substance.empty:
                print(f"  ⚠ Warning: No data remaining after time processing for substance '{substance}'.")
                print(f"  Using reference time series with zero values to maintain time series frequency.")
                # Use reference time series with zero values
                df_substance = reference_time_series.copy()
                df_substance['value'] = '0.00000'
            else:
                print(f"  {len(df_substance)} rows after time processing")
                # Format the value
                df_substance = process_value(df_substance, factor, value_col, new_value_col='value', decimal_places=5)
                
                # Merge with reference time series to ensure all time steps are present
                # For missing time steps, fill with zero
                df_substance = reference_time_series.merge(
                    df_substance[['time_since', 'value']], 
                    on='time_since', 
                    how='left'
                )
                df_substance['value'] = df_substance['value'].fillna('0.00000')
        
        print(f"  {len(df_substance)} rows ready for writing")
        
        # Write the .bc file
        # Define header lines as tuples of (key, value)
        header_lines = [
            ("Name", name),
            ("Function", function),
            ("Time-interpolation", time_interpolation),
            ("Quantity", quantity_time),
            ("Unit", unit_time),
            ("Quantity", quantity_substance),
            ("Unit", unit_substance),
        ]

        get_bc_file(delwaq_id, df_substance, header_lines, data_dir, time_col='time_since', value_col='value')

    print("\n✔ Processing complete!")


if __name__ == "__main__":
    main()
    