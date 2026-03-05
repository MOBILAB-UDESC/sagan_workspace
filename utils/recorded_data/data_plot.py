import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import os

# Files to load (these are the trimmed & resampled versions from the previous step)
files_map = {
    'IMU': 'imu_data_resampled.csv',
    'Ground Truth': 'odom_gt_data_resampled.csv',
    'Odom Noise': 'odom_noise_data_resampled.csv'
}

plt.figure(figsize=(12, 6))

for label, filename in files_map.items():
    if os.path.exists(filename):
        # 1. Load the CSV
        df = pd.read_csv(filename)
        
        # 2. Check for Orientation columns
        cols = ['orientation_x', 'orientation_y', 'orientation_z', 'orientation_w']
        if all(c in df.columns for c in cols):
            # 3. Extract Quaternions
            quats = df[cols].to_numpy()
            
            # 4. Convert to Yaw (Euler 'xyz', index 2)
            r = R.from_quat(quats)
            yaw = r.as_euler('xyz', degrees=False)[:, 2]
            
            # 5. Plot
            plt.plot(df['time'], yaw, label=label, linewidth=1.5, alpha=0.8)
            print(f"Loaded and plotted: {filename}")
        else:
            print(f"Skipping {filename}: Missing orientation columns.")
    else:
        print(f"Warning: {filename} not found. (Did you run the previous step?)")

plt.title("Yaw Orientation Comparison (from Resampled Data)")
plt.xlabel("Time (s)")
plt.ylabel("Yaw (radians)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Save and Show
plt.savefig('resampled_yaw_plot.png')
plt.show()