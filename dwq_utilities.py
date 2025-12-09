import pandas as pd
import os
import numpy as np
import re
from collections import defaultdict
import requests
import xml.etree.ElementTree as ET
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import re
import glob

def get_bc_file(delwaq_id, df_substance, header_lines,  data_dir, time_col='time_since', value_col='value'):
    # Determine the width of the longest key for alignment
    max_key_len = max(len(key) for key, _ in header_lines)

    # Build header string dynamically
    header = "[forcing]\n"  # [forcing] at the top
    for key, val in header_lines:
        header += f"{key.ljust(max_key_len)} = {val}\n"

    # Build file path using var name
    output_file = os.path.join(data_dir, f"{delwaq_id}.bc")

    # Write the .bc file
    with open(output_file, "w") as f:
        f.write(header)
        for _, row in df_substance.iterrows():
            f.write(f"{row[time_col]} {row[value_col]}\n")
    
    print(f"  ✓ Saved: {output_file}")


def clean_numeric(df):
    """
    Convert all columns of a DataFrame to numeric floats.
    - Handles values like '<0.05' by taking the number and dividing by 2.
    - Non-numeric values are converted to NaN.
    """
    
    def process_value(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, str) and x.startswith('<'):
            try:
                num = float(x[1:])  # remove '<' and convert
                return num / 2
            except:
                return np.nan
        try:
            return float(x)
        except:
            return np.nan
    
    # Apply to all columns
    df_numeric = df.map(process_value)
    return df_numeric


def clean_waterkwaliteit_dataset(df, time_col, new_time_col):
    """
    Cleans date-like strings that contain suffixes like '.1', '.2',
    converts the column to datetime, and returns the dataframe sorted by it.
    """
    df = df.copy()
    
    # Ensure column exists
    if time_col not in df.columns:
        raise KeyError(f"Column '{time_col}' not found in dataframe.")
    
    # Remove anything like .1, .2, .33, etc. at the end of the string
    df[time_col] = df[time_col].astype(str).str.replace(r"\.\d+$", "", regex=True)
    
    # Convert to datetime (assuming format dd-mm-yyyy)
    df[time_col] = pd.to_datetime(df[time_col], format="%d-%m-%Y", errors="coerce")

    # Sort rows by this column
    df = df.sort_values(by=time_col).reset_index(drop=True)

    # Set column as index
    df = df.set_index(time_col)

    df  = clean_numeric(df)

    # Drop the rows where all values are NaN with the exception of the index
    df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')

    # Set index as column
    df = df.reset_index()

    counts = df[time_col].value_counts()

    # Find the dates that appear more than once
    duplicates = counts[counts > 1]

    if not duplicates.empty:
        print("WARNING: Multiple rows with the same datetime found:")
        # Group by datetime and take mean of numeric columns
        df = df.groupby(time_col).mean().reset_index()
        print(duplicates)
    else:
        print("No duplicate datetimes found.")

    # Reset index and rename date column
    df.rename(columns={time_col: new_time_col}, inplace=True)
    
    # Set new time column as index
    df.set_index(new_time_col, inplace=True)
    
    return df


def rename_columns(df, substance_dict):
    """
    Rename columns in df according to substance_dict.
    Only keeps mappings where the substance has a non-zero name.
    Handles missing columns gracefully.
    
    Parameters:
        df (pd.DataFrame): Input dataframe
        substance_dict (dict): Dictionary mapping new names to existing columns
    
    Returns:
        pd.DataFrame: DataFrame with renamed columns
    """
    # Keep only valid substances (non-zero values)
    valid_substances = {k: v for k, v in substance_dict.items() if v != 0}

    # Build a rename dictionary: only for columns that exist in df
    rename_dict = {v: k for k, v in valid_substances.items() if v in df.columns}

    # Rename columns
    df_renamed = df.rename(columns=rename_dict)

    return df_renamed


def rename_highest_mean_duplicate(df, colname, new_name):
    """
    Rename ONLY one duplicated column with highest mean.
    Handles columns with identical names by tracking occurrence index.
    """
    # Collect the positions of columns with this name
    indices = [i for i, c in enumerate(df.columns) if c == colname]

    if len(indices) < 2:
        return df  # nothing to do

    # Compute means per duplicated occurrence
    means = {}
    for occ, col_idx in enumerate(indices):
        means[occ] = df.iloc[:, col_idx].mean()

    # Which occurrence has the highest mean?
    highest_occ = max(means, key=means.get)

    # Build new column names
    new_columns = df.columns.tolist()

    # Rename only that occurrence
    col_to_rename_idx = indices[highest_occ]
    new_columns[col_to_rename_idx] = new_name

    # Apply back
    df = df.copy()
    df.columns = new_columns
    return df


def read_waarde_sheet(filepath, time_col=None, new_time_col=None):
    df = pd.read_excel(filepath, sheet_name="effluent rwzi Waarde")
    # Remove first two rows
    # df = df.iloc[3:].reset_index(drop=True)
    # df.rename(columns={time_col: new_time_col}, inplace=True)
    # df.set_index(new_time_col, inplace=True)
    return df

def read_gemalen_sheet(filepath, time_col, new_time_col):
    df = pd.read_excel(filepath, sheet_name="Gemalen", skiprows=2)
    # Remove first two rows
    df = df.iloc[2:].reset_index(drop=True)
    df.rename(columns={time_col: new_time_col}, inplace=True)
    df.set_index(new_time_col, inplace=True)
    return df


def group_columns_by_kgm(columns):
    groups = defaultdict(list)

    for col in columns:
        match = re.match(r"(KGM\d+)", col)
        if match:
            kgm = match.group(1)
            groups[kgm].append(col)

    print(f"\nFound {len(groups)} KGM station groups")

    return dict(groups)


def aggregate_by_grouped(df, grouped):
    # Initialize new DataFrame with the same index as original
    df_new = pd.DataFrame(index=df.index)
    
    for new_col, old_cols in grouped.items():
        # Sum the old columns row-wise; ignore missing columns
        existing_cols = [col for col in old_cols if col in df.columns]
        df_new[new_col] = df[existing_cols].sum(axis=1)
    
    return df_new


def save_columns_to_csv(df, output_path, time_col, suffix="_discharge", stations_ids=None):
    """
    Save columns to CSV files.
    
    Note: If stations_ids is None, all columns will be saved.
    Otherwise, only columns matching station IDs will be saved.
    """
    os.makedirs(output_path, exist_ok=True)

    # Reset index
    df.reset_index(inplace=True)
    
    # If stations_ids is not provided, save all columns
    columns_to_save = df.columns if stations_ids is None else [col for col in df.columns if col in stations_ids]
    
    for col in columns_to_save:
        filename = f"{col}{suffix}.csv"
        filepath = os.path.join(output_path, filename)
        # Save with index
        df[[time_col, col]].to_csv(filepath, index=False, sep=";")
        print(f"Saved: {filepath}")


# ============================================================================
# Freshwater Boundaries Functions
# ============================================================================

def resample_time_series(df, date_column='datetime', value_column='intercept', time_step='D'):
    """
    Resample time series to a specified time step with constant values between data points.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with date and value columns
    date_column : str
        Name of the date column (default: 'datetime')
    value_column : str
        Name of the value column to keep constant (default: 'intercept')
    time_step : str
        Pandas time step frequency (e.g., 'D' for daily, 'H' for hourly, '6H' for 6-hourly)
        See pandas offset aliases: https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases
    
    Returns:
    --------
    pandas.DataFrame
        Resampled dataframe with regular time steps and forward-filled values
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Convert date column to datetime if it's a string
    if df[date_column].dtype == 'object':
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Set date as index for resampling
    df = df.set_index(date_column)
    
    # Resample to the specified time step and forward fill values
    # This keeps values constant until the next data point
    df = df.resample(time_step).ffill()
    
    # Reset index to make date a column again
    df = df.reset_index()
    
    # Format the date column back to string with time component
    df[date_column] = df[date_column].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return df


def convert_datetime(df, date_col, ref_date, unit='minutes',
                     new_col='time_since',
                     start_date=None, end_date=None):
    """
    Convert a datetime column to time since a reference date, 
    with optional time filtering.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe
    date_col : str
        Name of the datetime column
    ref_date : str or datetime
        Reference datetime (e.g. '2017-12-01')
    unit : str
        Output unit: 'seconds', 'minutes', 'hours', 'days'
    new_col : str
        Name of the column to store the result
    start_date : str or datetime, optional
        Start date for filtering (inclusive)
    end_date : str or datetime, optional
        End date for filtering (inclusive)

    Returns
    -------
    pandas.DataFrame
    """
    df = df.copy()

    # Ensure datetime
    df[date_col] = pd.to_datetime(df[date_col])
    ref_date = pd.to_datetime(ref_date)

    # Optional filtering
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        df = df[df[date_col] >= start_date]

    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        df = df[df[date_col] <= end_date]

    # Compute timedelta
    delta = df[date_col] - ref_date

    # Convert units
    if unit == 'seconds':
        df[new_col] = delta.dt.total_seconds()
    elif unit == 'minutes':
        df[new_col] = delta.dt.total_seconds() / 60
    elif unit == 'hours':
        df[new_col] = delta.dt.total_seconds() / 3600
    elif unit == 'days':
        df[new_col] = delta.dt.total_seconds() / 86400
    else:
        raise ValueError("unit must be one of: 'seconds', 'minutes', 'hours', 'days'")

    # Format in scientific notation
    df[new_col] = df[new_col].apply(lambda x: format(x, '.6e'))

    return df


def convert_units(df, col, factor=0.001):
    """
    Convert units by multiplying by 0.001 and formatting in scientific notation.
    
    Example: df_name = convert_units(df_name, 'intercept')
    """
    df = df.copy()
    df[col] = df[col] * factor
    df[col] = df[col].apply(lambda x: format(x, '.6e'))
    return df


def process_time(df, date_col, ref_date, unit='minutes', new_col='time_since', start_date=None, end_date=None):
    """
    Process time column: convert to datetime, add record, sort, format, and convert to time since reference.
    """
    # Get columns for name, date, and intercept
    df = df.copy()

    # If date column is not in the dataframe, add it reset index to expose the index as a column
    if date_col not in df.columns:
        df.reset_index(inplace=True)

    # Ensure the date column is datetime
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df[date_col] = df[date_col].dt.strftime('%Y-%m-%d %H:%M:%S')
    df = convert_datetime(df, date_col, ref_date, unit, new_col, start_date, end_date)
    return df


def process_value(df, factor, value_col, new_value_col='value', decimal_places=5):
    """
    Process value column: multiply by factor, set negatives to 0, round to 5 decimals.
    Note: Function name has typo but kept for compatibility.
    """
    df = df.copy()
    df[new_value_col] = df[value_col]*factor
    # If value is negative, set it to 0
    df[new_value_col] = df[new_value_col].apply(lambda x: 0 if x < 0 else x)
    # Keep only decimal_places decimals. 
    df[new_value_col] = df[new_value_col].apply(lambda x: round(x, decimal_places))
    df[new_value_col] = df[new_value_col].apply(lambda x: f"{x:.{decimal_places}f}")

    return df


def filter_data(df, start_date, end_date):
    """
    Filter DataFrame by date range.
    """
    df = df.copy()
    # Filter the DataFrame
    mask = (df['datetime'] >= start_date) & (df['datetime'] <= end_date)
    filtered_df = df[mask]
    return filtered_df


def download_wfs_csv(url, output_filename, timeout=300):
    """
    Download CSV from a WFS (Web Feature Service) endpoint with proper error handling.
    
    Parameters:
    -----------
    url : str
        The WFS URL to download from
    output_filename : str
        Name of the output CSV file
    timeout : int
        Request timeout in seconds (default: 300)
    
    Returns:
    --------
    bool
        True if download successful, False otherwise
    """
    try:
        # Set headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/csv,application/csv,*/*',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        }
        
        print(f"Downloading from: {url[:100]}...")
        print(f"Output file: {output_filename}")
        
        # Make the request with timeout
        response = requests.get(url, headers=headers, timeout=timeout, stream=False)
        
        # Check content type first
        content_type = response.headers.get('Content-Type', '')
        print(f"Content-Type: {content_type}")
        
        # Check if response is XML (error) instead of CSV
        if 'xml' in content_type.lower() or response.text.strip().startswith('<?xml'):
            print("✗ Error: Server returned XML (likely an error response)")
            print("\nFull error response:")
            print("=" * 80)
            print(response.text)
            print("=" * 80)
            
            # Try to parse XML error
            try:
                root = ET.fromstring(response.text)
                # Look for exception text
                for elem in root.iter():
                    if 'ExceptionText' in elem.tag or 'exception' in elem.tag.lower():
                        print(f"\nParsed Error: {elem.text}")
            except:
                pass
            
            return False
        
        # Check if request was successful
        response.raise_for_status()
        
        # Write to file
        with open(output_filename, "wb") as f:
            f.write(response.content)
        
        file_size = os.path.getsize(output_filename)
        print(f"✓ Download successful! File size: {file_size:,} bytes ({file_size/1024:.2f} KB)")
        return True
        
    except requests.exceptions.Timeout:
        print(f"✗ Error: Request timed out after {timeout} seconds")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"✗ HTTP Error: {e}")
        print(f"Response status code: {response.status_code}")
        if hasattr(response, 'text'):
            print(f"Response content: {response.text[:1000]}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Request Error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def preprocess_scheldemonitor(data, sep, min_date, no_data, selected_columns):
    """
    Preprocess Scheldemonitor data.
    """
    # Read dataframe
    df = pd.read_csv(data, sep=sep)
    # Get data for dates after min_Date
    df = df.sort_values('datetime')
    df = df[df['datetime'] >= min_date]
    # Print min and maximun date
    print(f"Minimum date: {df['datetime'].min()}")
    print(f"Maximum date: {df['datetime'].max()}")
    # Replace records where 'value' has no data
    df['value'] = df['value'].replace(no_data, np.nan).astype(float)
    df = df[df['value'].notna()]
    # Select columns in a new dataframe
    df = df[selected_columns]
    return df


def get_scheldemonitor_map(df):
    """
    Convert DataFrame to GeoDataFrame using longitude and latitude.
    """    
    # Convert to GeoDataFrame using longitude and latitude
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326"  # Set CRS properly
    )
    return gdf


def get_summary(df, output_csv):
    """
    Get summary statistics for stations and parameters.
    """
    # Prepare list to store summary
    summary_records = []

    unique_para = df['parametername'].unique() 
    unique_station = df['stationname'].unique()

    for sta in unique_station:
        df_station = df[df['stationname'] == sta]
        for para in unique_para:
            df_parameter = df_station[df_station['parametername'] == para]
            record_count = len(df_parameter)  # number of rows, 0 if empty
            
            summary_records.append({
                'stationname': sta,
                'parametername': para,
                'record_count': record_count
            })

    # Create summary DataFrame
    location_summary = pd.DataFrame(summary_records)
    
    # Get the count of repetitions for each unique value
    value_counts = location_summary['record_count'].value_counts().sort_index()
    print(f"\nRepetitions (frequency) for each unique record_count value:")
    print(value_counts)

    location_summary = location_summary[location_summary['record_count'] > 0]
    value_counts = location_summary['parametername'].value_counts().sort_index()
    value_counts.to_csv(output_csv, sep=';')

    value_counts = value_counts.to_frame()
    value_counts['country_count'] = 1
    
    return value_counts, location_summary


def filter_station_parameter(df, stationname, parametername):
    """
    Filter DataFrame by station name and parameter name.
    """
    filtered_df = df[
        (df['stationname'] == stationname) &
        (df['parametername'] == parametername)
    ].copy() 
    return filtered_df


def plot_data(df, location_summary, records):
    """
    Plot time series data for stations and parameters with record count below threshold.
    """

    # Iterate over the summary table
    for idx, row in location_summary.iterrows():
        sta = row['stationname']
        para = row['parametername']
        count = row['record_count']
        
        if count < records:
            # Filter the original DataFrame
            filtered_df = df[
                (df['stationname'] == sta) &
                (df['parametername'] == para)
            ].copy()  # use .copy() to avoid SettingWithCopyWarning

            print(sta,"and", para)
            try:
                # Make sure datetime column is datetime type
                filtered_df['datetime'] = pd.to_datetime(filtered_df['datetime'])
                filtered_df = filtered_df.sort_values('datetime')
                
                # Plot
                plt.figure(figsize=(12, 6))
                plt.plot(filtered_df['datetime'], filtered_df['value'], marker='o', linestyle='-')
                plt.xlabel('Date')
                plt.ylabel('Value')
                plt.title(f'Time Series of {para} at {sta}')
                plt.grid(True)
                plt.tight_layout()
                plt.show()

            except:
                print(f"Error in {sta} and {para}")


def create_bbx(pli_files=None, directory=None, buffer_meters=100):
    """
    Read coordinates from .pli files and calculate bounding box with buffer.
    
    Parameters:
    -----------
    pli_files : list of str, optional
        List of paths to .pli files. If None, will search for .pli files in directory.
    directory : str, optional
        Directory path to search for .pli files. If None and pli_files is None,
        uses current directory.
    buffer_meters : float
        Buffer to add to bounding box in meters (default: 100)
    
    Returns:
    --------
    dict
        Dictionary with keys: 'xmin', 'xmax', 'ymin', 'ymax', 'width', 'height'
    """
    import glob
    
    # Collect all .pli files
    if pli_files is None:
        if directory is None:
            directory = os.getcwd()
        pli_files = glob.glob(os.path.join(directory, "*.pli"))
    
    if not pli_files:
        raise ValueError("No .pli files found")
    
    # Collect all coordinates
    all_x = []
    all_y = []
    
    for pli_file in pli_files:
        if not os.path.exists(pli_file):
            print(f"Warning: File not found: {pli_file}")
            continue
        
        with open(pli_file, 'r') as f:
            lines = f.readlines()
        
        # Skip first line (name) and second line (number of points, dimensions)
        if len(lines) < 3:
            print(f"Warning: File {pli_file} has insufficient lines")
            continue
        
        # Read coordinates (starting from line 3, index 2)
        for line in lines[2:]:
            line = line.strip()
            if not line:
                continue
            try:
                parts = line.split()
                if len(parts) >= 2:
                    x = float(parts[0])
                    y = float(parts[1])
                    all_x.append(x)
                    all_y.append(y)
            except ValueError:
                continue
    
    if not all_x or not all_y:
        raise ValueError("No valid coordinates found in .pli files")
    
    # Calculate bounding box
    xmin = min(all_x)
    xmax = max(all_x)
    ymin = min(all_y)
    ymax = max(all_y)
    
    # Add buffer
    xmin -= buffer_meters
    xmax += buffer_meters
    ymin -= buffer_meters
    ymax += buffer_meters
    
    bbx = {
        'xmin': xmin,
        'xmax': xmax,
        'ymin': ymin,
        'ymax': ymax,
        'width': xmax - xmin,
        'height': ymax - ymin
    }
    
    print(f"Bounding box calculated from {len(pli_files)} .pli file(s):")
    print(f"  X range: {xmin:.2f} to {xmax:.2f} (width: {bbx['width']:.2f} m)")
    print(f"  Y range: {ymin:.2f} to {ymax:.2f} (height: {bbx['height']:.2f} m)")
    print(f"  Buffer: {buffer_meters} m")
    
    return bbx


def read_pli_names(pli_file):
    """
    Read boundary names from a .pli file.
    Returns a list of names found in the file, prioritizing names ending with _0001.
    """
    names = []
    names_0001 = []
    try:
        with open(pli_file, 'r') as f:
            lines = f.readlines()
            
        # First line is usually the main name
        if lines:
            main_name = lines[0].strip()
            if main_name:
                names.append(main_name)
                # If it doesn't have _0001, create a version with it
                if not main_name.endswith('_0001'):
                    names_0001.append(f"{main_name}_0001")
                elif main_name.endswith('_0001'):
                    names_0001.append(main_name)
            
        # Extract names from coordinate lines (format: x y name)
        for line in lines[2:]:  # Skip first two lines (name and dimensions)
            line = line.strip()
            if line:
                # Split by whitespace and get the last part (the name)
                parts = line.split()
                if len(parts) >= 3:
                    name = parts[-1]  # Last part is the boundary name
                    if name:
                        if name not in names:
                            names.append(name)
                        # Collect names ending with _0001
                        if name.endswith('_0001') and name not in names_0001:
                            names_0001.append(name)
    except Exception as e:
        print(f"  ⚠ Error reading {pli_file}: {e}")
    
    # Return _0001 names if found, otherwise return all names
    return names_0001 if names_0001 else names


def get_all_pli_names(results_dir):
    """
    Read all .pli files in the directory and collect all boundary names.
    Returns a list of unique names.
    """
    pli_files = glob.glob(os.path.join(results_dir, "*.pli"))
    all_names = []
    
    print(f"Reading .pli files from: {results_dir}")
    for pli_file in pli_files:
        pli_name = os.path.basename(pli_file)
        names = read_pli_names(pli_file)
        print(f"  {pli_name}: Found {len(names)} names")
        all_names.extend(names)
    
    # Remove duplicates while preserving order
    unique_names = []
    seen = set()
    for name in all_names:
        if name not in seen:
            unique_names.append(name)
            seen.add(name)
    
    print(f"\nTotal unique names found: {len(unique_names)}")
    return unique_names


def extract_time_series_from_bc(bc_file):
    """
    Extract time series data and header information from a .bc file.
    Returns a tuple: (header_template, time_series_data)
    where header_template is a dict with header components
    """
    try:
        with open(bc_file, 'r') as f:
            lines = [line.rstrip('\n') for line in f.readlines()]
        
        header_template = {
            'forcing': '[forcing]',
            'function': None,
            'time_interpolation': None,
            'quantity_time': None,
            'unit_time': None,
            'quantity_substance': None,
            'unit_substance': None
        }
        time_series_data = []
        in_header = True
        found_time_quantity = False
        
        for line in lines:
            line_stripped = line.strip()
            
            # Skip empty lines
            if not line_stripped:
                if in_header:
                    continue
                else:
                    # Empty line after time series might indicate new section
                    continue
            
            # Collect header lines
            if in_header:
                if line_stripped.startswith('[forcing]'):
                    continue  # Already have this
                elif 'Function' in line_stripped and '=' in line_stripped:
                    parts = line_stripped.split('=')
                    if len(parts) == 2:
                        header_template['function'] = parts[1].strip()
                elif 'Time-interpolation' in line_stripped and '=' in line_stripped:
                    parts = line_stripped.split('=')
                    if len(parts) == 2:
                        header_template['time_interpolation'] = parts[1].strip()
                elif 'Quantity' in line_stripped and '=' in line_stripped:
                    parts = line_stripped.split('=')
                    if len(parts) == 2:
                        quantity_value = parts[1].strip()
                        if 'time' in quantity_value.lower():
                            header_template['quantity_time'] = quantity_value
                            found_time_quantity = True
                        elif 'tracerbnd' in quantity_value.lower() or found_time_quantity:
                            header_template['quantity_substance'] = quantity_value
                elif 'Unit' in line_stripped and '=' in line_stripped:
                    parts = line_stripped.split('=')
                    if len(parts) == 2:
                        unit_value = parts[1].strip()
                        if found_time_quantity and header_template['unit_time'] is None:
                            header_template['unit_time'] = unit_value
                        elif header_template['quantity_substance'] and header_template['unit_substance'] is None:
                            header_template['unit_substance'] = unit_value
                            # After substance unit, we should have time series data
                            in_header = False
                continue
            
            # After header, collect time series data
            if not in_header:
                parts = line_stripped.split()
                if len(parts) == 2:
                    try:
                        # Verify both parts are numbers
                        float(parts[0])
                        float(parts[1])
                        time_series_data.append(line_stripped)
                    except ValueError:
                        # Check if this is a new [forcing] section
                        if line_stripped.startswith('['):
                            in_header = True
                            found_time_quantity = False
        
        return header_template, time_series_data
        
    except Exception as e:
        print(f"  ⚠ Error reading {bc_file}: {e}")
        return {}, []


def create_txt_file_with_all_boundaries(bc_file, pli_names, results_dir):
    """
    Create a .txt file with time series for all boundaries from .pli files.
    Each boundary gets its own [forcing] section with the same time series data.
    """
    substance_name = os.path.splitext(os.path.basename(bc_file))[0]
    txt_file = os.path.join(results_dir, f"{substance_name}_updated.bc")
    
    # Extract time series and header info from .bc file
    header_template, time_series_data = extract_time_series_from_bc(bc_file)
    
    if not time_series_data:
        print(f"  ⚠ No time series data found in {os.path.basename(bc_file)}")
        return
    
    if not pli_names:
        print(f"  ⚠ No pli names found, skipping .txt file creation")
        return
    
    # Check if we have all required header components
    if not all([header_template.get('function'), 
                header_template.get('time_interpolation'),
                header_template.get('quantity_time'),
                header_template.get('unit_time'),
                header_template.get('quantity_substance'),
                header_template.get('unit_substance')]):
        print(f"  ⚠ Incomplete header information in {os.path.basename(bc_file)}")
        return
    
    # Write .txt file with all boundaries
    try:
        with open(txt_file, 'w') as f:
            for i, boundary_name in enumerate(pli_names):
                # Ensure boundary name ends with _0001
                if not boundary_name.endswith('_0001'):
                    boundary_name = f"{boundary_name}_0001"
                
                # Write header for this boundary
                f.write(header_template['forcing'] + '\n')
                f.write(f"Name               = {boundary_name}\n")
                f.write(f"Function           = {header_template['function']}\n")
                f.write(f"Time-interpolation = {header_template['time_interpolation']}\n")
                f.write(f"Quantity           = {header_template['quantity_time']}\n")
                f.write(f"Unit               = {header_template['unit_time']}\n")
                f.write(f"Quantity           = {header_template['quantity_substance']}\n")
                f.write(f"Unit               = {header_template['unit_substance']}\n")
                
                # Write time series data
                for ts_line in time_series_data:
                    f.write(ts_line + '\n')
                
                # Add blank line between sections (except for last one)
                if i < len(pli_names) - 1:
                    f.write('\n')
        
        print(f"  ✓ Created {txt_file} with {len(pli_names)} boundaries, {len(time_series_data)} data points each")
        
    except Exception as e:
        print(f"  ⚠ Error writing {txt_file}: {e}")


def update_bc_file(bc_file, pli_names, results_dir):
    """
    Update a .bc file with the first name from pli_names.
    """
    substance_name = os.path.splitext(os.path.basename(bc_file))[0]
    
    try:
        with open(bc_file, 'r') as f:
            content = f.read()
        
        # Update Name attribute in .bc file with first pli name
        if pli_names:
            # Use the first name from pli_names
            new_name = pli_names[0]
            if not new_name.endswith('_0001'):
                new_name = f"{new_name}_0001"
            
            # Update the Name line in header (only the first occurrence if multiple sections)
            updated_content = content
            name_pattern = r'Name\s*=\s*[^\n]+'
            # Replace first occurrence
            lines = updated_content.split('\n')
            for i, line in enumerate(lines):
                if 'Name' in line and '=' in line:
                    lines[i] = f"Name               = {new_name}"
                    break
            updated_content = '\n'.join(lines)
            
            # Write updated .bc file
            with open(bc_file, 'w') as f:
                f.write(updated_content)
            print(f"  ✓ Updated first Name in {os.path.basename(bc_file)} to: {new_name}")
        else:
            print(f"  ⚠ No pli names found, skipping Name update for {os.path.basename(bc_file)}")
            
    except Exception as e:
        print(f"  ⚠ Error processing {bc_file}: {e}")