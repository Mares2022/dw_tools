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

    # Process each substance
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

        # Format the time 
        df_substance = process_time(df_substance, 'datetime', '2017-12-01', 'minutes', 'time_since', '2018-01-01', '2019-01-01')

        # Format the value
        df_substance = process_value(df_substance, factor, value_col, new_value_col='value', decimal_places=5)
        
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
    