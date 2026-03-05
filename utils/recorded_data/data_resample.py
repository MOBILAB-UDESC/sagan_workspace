import pandas as pd
import numpy as np
import os

def process_and_resample_system_time(file_path, sample_time=0.01):
    """
    Reads a CSV, normalizes system_timestamp_ns to seconds starting at 0, 
    resamples to a fixed grid, and interpolates all data columns.
    Returns the resampled DataFrame.
    """
    # 1. Read the data
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    df = pd.read_csv(file_path)

    # Check if required column exists
    if 'system_timestamp_ns' not in df.columns:
        print(f"Skipping {file_path}: 'system_timestamp_ns' column not found.")
        return None

    # 2. Extract and Convert Time
    # Convert nanoseconds to seconds (1e-9)
    t_raw = df['system_timestamp_ns'].values.astype(float) * 1e-9
    
    # Normalize Time (Make the first data t=0)
    t_start = t_raw[0]
    t_normalized = t_raw - t_start
    
    # 3. Create new Time Grid (0, 0.01, 0.02, ... up to max time)
    t_max = t_normalized[-1]
    t_new = np.arange(0, t_max, sample_time)
    
    # 4. Interpolate all columns
    new_data = {'time': t_new}
    
    for col in df.columns:
        # We generally exclude the raw timestamps from the output, keeping only the aligned 'time'
        if col in ['msg_timestamp', 'receive_timestamp', 'system_timestamp_ns']:
            continue
            
        # Ensure column is numeric before interpolating
        if pd.api.types.is_numeric_dtype(df[col]):
            # Linear interpolation
            new_data[col] = np.interp(t_new, t_normalized, df[col].values)
            
    # 5. Create new DataFrame
    df_resampled = pd.DataFrame(new_data)
    
    return df_resampled

# List of files to process
files_to_process = [
    'imu_data.csv', 
    'odom_gt_data.csv', 
    'odom_noise_data.csv'
]

# Store processed dataframes
processed_results = []

# Run the processing for each file
for file in files_to_process:
    df = process_and_resample_system_time(file)
    if df is not None:
        processed_results.append({'file': file, 'df': df})

# 6. Trim to the minimum length (The "Trim Thing")
if processed_results:
    # Find the minimum number of rows among all dataframes
    min_len = min(len(item['df']) for item in processed_results)
    print(f"Common length determined: {min_len} rows")

    for item in processed_results:
        original_file = item['file']
        df = item['df']
        
        # Trim the dataframe to the minimum length
        df_trimmed = df.iloc[:min_len]
        
        # Generate output filename (e.g., imu_data.csv -> imu_data_resampled.csv)
        base_name = os.path.splitext(original_file)[0]
        output_file = f"{base_name}_resampled.csv"
        
        # Save
        df_trimmed.to_csv(output_file, index=False)
        print(f"Saved: {original_file} -> {output_file} | Rows: {len(df_trimmed)}")
else:
    print("No data processed.")