import matplotlib.pyplot as plt
import re
import csv
import numpy as np

def parse_pose_string(cell_str):
    """
    Extracts timestamp, x, and y from a string formatted like:
    "(453.91, geometry_msgs.msg.Pose(position=geometry_msgs.msg.Point(x=..., y=... ...)))"
    """
    # 1. Extract Timestamp (at the start of the tuple)
    t_match = re.search(r"^\(([\d\.]+)", cell_str)
    t = float(t_match.group(1)) if t_match else 0.0
    
    # 2. Split string to isolate Position from Orientation to avoid confusion
    # The format is consistent: position=... followed by orientation=...
    parts = cell_str.split("orientation=")
    pos_part = parts[0]
    
    # 3. Extract X and Y using Regex handles scientific notation (e.g. 2.8e-09)
    x_match = re.search(r"x=([-\d\.eE]+)", pos_part)
    y_match = re.search(r"y=([-\d\.eE]+)", pos_part)
    
    x = float(x_match.group(1)) if x_match else 0.0
    y = float(y_match.group(1)) if y_match else 0.0
    
    return t, x, y

def plot_ros_data(csv_filename):
    # Initialize lists
    data = {
        'gazebo': {'t': [], 'x': [], 'y': []},
        'ekf':    {'t': [], 'x': [], 'y': []},
        'odom':   {'t': [], 'x': [], 'y': []}
    }

    try:
        with open(csv_filename, 'r') as f:
            reader = csv.reader(f)
            
            # Skip the header row (gazebo,ekf,odom_noisy)
            headers = next(reader, None)
            
            for row in reader:
                # Ensure row has enough columns (expecting 3)
                if not row or len(row) < 3:
                    continue
                
                # Column 0: Gazebo (Ground Truth)
                gt_t, gt_x, gt_y = parse_pose_string(row[0])
                data['gazebo']['t'].append(gt_t)
                data['gazebo']['x'].append(gt_x)
                data['gazebo']['y'].append(gt_y)

                # Column 1: EKF
                ekf_t, ekf_x, ekf_y = parse_pose_string(row[1])
                data['ekf']['t'].append(ekf_t)
                data['ekf']['x'].append(ekf_x)
                data['ekf']['y'].append(ekf_y)

                # Column 2: Odom Noisy
                odom_t, odom_x, odom_y = parse_pose_string(row[2])
                data['odom']['t'].append(odom_t)
                data['odom']['x'].append(odom_x)
                data['odom']['y'].append(odom_y)

    except FileNotFoundError:
        print(f"Error: The file '{csv_filename}' was not found.")
        return

    # --- Calculations ---
    # Convert lists to numpy arrays for easy math
    gz_x = np.array(data['gazebo']['x'])
    gz_y = np.array(data['gazebo']['y'])
    ekf_x = np.array(data['ekf']['x'])
    ekf_y = np.array(data['ekf']['y'])
    odom_x = np.array(data['odom']['x'])
    odom_y = np.array(data['odom']['y'])
    
    # Normalize time to start at 0
    t_start = data['gazebo']['t'][0]
    time_axis = np.array(data['gazebo']['t']) - t_start

    # Euclidean Error Calculation
    # Note: This assumes row-alignment (timestamps match per row)
    err_ekf = np.sqrt((gz_x - ekf_x)**2 + (gz_y - ekf_y)**2)
    err_odom = np.sqrt((gz_x - odom_x)**2 + (gz_y - odom_y)**2)

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: 2D Trajectory
    ax1.plot(gz_x, gz_y, 'k-', linewidth=2, label='Ground Truth (Gazebo)')
    ax1.plot(ekf_x, ekf_y, 'b--', linewidth=1.5, label='EKF Estimate')
    ax1.plot(odom_x, odom_y, 'r:', linewidth=1.5, alpha=0.8, label='Noisy Odom')
    ax1.set_title("Robot Trajectory (Top-Down)")
    ax1.set_xlabel("X Position (m)")
    ax1.set_ylabel("Y Position (m)")
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')

    # Plot 2: Error Analysis
    ax2.plot(time_axis, err_ekf, 'b-', label='EKF Error')
    ax2.plot(time_axis, err_odom, 'r-', label='Odom Error', alpha=0.6)
    ax2.set_title("Position Error relative to Ground Truth")
    ax2.set_xlabel("Time Elapsed (s)")
    ax2.set_ylabel("Euclidean Error (m)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Ensure data.csv exists in the same directory
    plot_ros_data('data.csv')