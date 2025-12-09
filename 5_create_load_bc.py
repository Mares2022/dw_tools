"""
Loads Data Processing Script

This script processes water quality and discharge data from gemalen (pumping stations)
and generates CSV files for boundary conditions.
"""
import os
import pandas as pd
from dw_tools.dwq_utilities import (
    process_time,
    clean_numeric,
    clean_waterkwaliteit_dataset,
    rename_columns,
    rename_highest_mean_duplicate,
    read_gemalen_sheet,
    group_columns_by_kgm,
    aggregate_by_grouped,
    save_columns_to_csv,
    read_waarde_sheet,
)


def main():
    # Define paths
    data_dir = r"P:\ltv-natuur-schelde-slib-waq\ecolmod\02_preprocessing\loads"
    substances_path = os.path.join(data_dir, "data waterkwaliteit gemalen Westerschelde.xlsx")
    discharge_path = os.path.join(data_dir, "Debieten_gemalen_stuwen_westerschelde_2018.xlsx")
    waarde_path = os.path.join(data_dir, "Concentraties_en_vrachten_in_de_waterlijn.xlsx")
    
    summary_file = os.path.join(data_dir, "summary_all_sheets.csv")

    # Time column
    time_col = "MPS.Omschrijving"
    new_time_col = "datetime"

    # Substance dictionary
    substance_dict = {
        "DOC": "koolstof organisch",
        "POC1": "Onopgeloste bestandsdelen",
        "NH4": "ammonium",
        "NO3": "som nitraat en nitriet",
        "PON1": "stikstof totaal",
        "PO4": "fosfaat",
        "POP1": "fosfor totaal",
        "Si": 0,
        "Opal": 0,
        "OXY": "zuurstof",
        "Diat": 0,
        "Green": 0
    }
    
    # Keep only substances that have meaningful names (not zeros)
    valid_substances = {k: v for k, v in substance_dict.items() if v != 0}
    
    # ============================================================================
    # Part 1: Process water quality data from Excel sheets
    # ============================================================================
    print("=" * 80)
    print("Part 1: Processing water quality data from Excel sheets")
    print("=" * 80)
    
    # Create an empty list to collect all rows
    summary_rows = []
    stations_ids = []
    
    with pd.ExcelFile(substances_path) as xls:
        for sheet_name in xls.sheet_names[:]:  # Process all sheets
            station_id = sheet_name.split('_')[-1]
            print(f"\nProcessing sheet: {station_id}")
            
            df = pd.read_excel(xls, sheet_name=sheet_name, skiprows=5)
            if df.empty:
                print("  (Sheet is empty after skipping rows)")
                continue
            
            # Transpose and clean
            df = df.transpose()
            df.reset_index(inplace=True)
            df.columns = df.iloc[0]
            df = df[1:].reset_index(drop=True)
            
            available_cols = df.columns.tolist()
            
            # Matching columns
            found = [v for v in valid_substances.values() if v in available_cols]
            not_found = [v for v in valid_substances.values() if v not in available_cols]
            
            print(f"  Found {len(found)} columns: {found}")
            print(f"  Missing {len(not_found)} columns: {not_found}")
            
            # Build the extraction column list
            extract_cols = [time_col] + found
            
            # Check MPS exists
            if time_col not in available_cols:
                print(f"  WARNING: '{time_col}' not found in this sheet.")
                continue
            
            # Extract the columns
            df_sub = df[extract_cols]
            
            # Clean and sort
            df_sub = clean_waterkwaliteit_dataset(df_sub, time_col, new_time_col)
            
            # Rename columns according to substance dictionary
            df_sub = rename_columns(df_sub, substance_dict)
            
            # Rename highest mean duplicate OXY column
            df_sub = rename_highest_mean_duplicate(df_sub, "OXY", "OXY_saturation")
            print(df_sub.columns)

            # df_sub = process_time(df_sub, new_time_col, '2017-12-01', 'minutes', 'time_since', '2018-01-01', '2019-01-01')
            
            # Save the sub dataframe to a csv file
            output_csv = os.path.join(data_dir, f"{station_id}_substance.csv")
            df_sub.reset_index(inplace=True)
            df_sub.to_csv(output_csv, index=False, sep=";")
            
            stations_ids.append(station_id)
            
            print(f"  The number of rows in the sub dataframe is: {df_sub.shape[0]}")
            print(f"  The number of columns in the sub dataframe is: {df_sub.shape[1]}")
            
            # Append summary row
            summary_rows.append({
                "sheet": sheet_name,
                "found_columns": ", ".join(found),
                "missing_columns": ", ".join(not_found),
                "n_found": len(found),
                "n_missing": len(not_found)
            })
    
    # Convert collected rows into a DataFrame
    summary_df = pd.DataFrame(summary_rows)
    
    # Save as a single CSV file using semicolon as separator
    summary_df.to_csv(summary_file, index=False, sep=";")
    
    print(f"\n✔ Combined summary saved to:\n{summary_file}")
    
    # ============================================================================
    # Part 2: Process discharge data from gemalen sheet
    # ============================================================================
    print("\n" + "=" * 80)
    print("Part 2: Processing discharge data from gemalen sheet")
    print("=" * 80)

    # Time column
    time_col = "Unnamed: 0"
    new_time_col = "datetime"
    
    # Read gemalen sheet
    df_gemalen = read_gemalen_sheet(discharge_path, time_col, new_time_col)
    
    # Clean numeric values
    df_gemalen = clean_numeric(df_gemalen)
    
    # Get columns and group by KGM
    grouped = group_columns_by_kgm(df_gemalen.columns)
    
    # Aggregate by grouped KGM stations
    df_aggregated = aggregate_by_grouped(df_gemalen, grouped)

    # df_aggregated = process_time(df_aggregated, new_time_col, '2017-12-01', 'minutes', 'time_since', '2018-01-01', '2019-01-01')
    
    # Save columns to CSV files
    print(f"\nSaving discharge CSV files for {len(stations_ids)} stations...")
    save_columns_to_csv(df_aggregated, data_dir, new_time_col, suffix="_discharge", stations_ids=stations_ids)
    
    print("\n✔ Processing complete!")

    # ============================================================================
    # Part 3: Process discharge data from Waarde dataset
    # ============================================================================

    # Read gemalen sheet
    # df_waarde = read_waarde_sheet(waarde_path, time_col, new_time_col)
    # print(df_waarde.head())


if __name__ == "__main__":
    main()
