#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation as R
import seaborn as sns

class OdometryAnalyzer2D:
    def __init__(self, data_dir="."):
        """
        Initialize the analyzer with data directory (2D focus)
        
        Args:
            data_dir: Directory containing the resampled CSV files
        """
        self.data_dir = data_dir
        self.data = {}
        
    def load_resampled_data(self):
        """
        Load all resampled data files
        """
        files = {
            'imu': 'imu_data_resampled.csv',
            'odom_gt': 'odom_gt_data_resampled.csv',
            'odom_noise': 'odom_noise_data_resampled.csv',
            'odom_filtered': 'odom_filtered_data_resampled.csv'
        }
        
        for key, filename in files.items():
            filepath = os.path.join(self.data_dir, filename)
            if os.path.exists(filepath):
                self.data[key] = pd.read_csv(filepath)
                print(f"Loaded {filename}: {len(self.data[key])} rows")
            else:
                print(f"Warning: {filename} not found in {self.data_dir}")
    
    def calculate_2d_rmse(self):
        """
        Calculate 2D RMSE between ground truth and noisy/filtered odometry
        
        Returns:
            dict: RMSE values for 2D position and yaw angle
        """
        if 'odom_gt' not in self.data or 'odom_noise' not in self.data or 'odom_filtered' not in self.data:
            print("Error: Required data not loaded")
            return None
        
        # Extract 2D position data (x and y only)
        gt_pos_2d = self.data['odom_gt'][['position_x', 'position_y']].values
        noise_pos_2d = self.data['odom_noise'][['position_x', 'position_y']].values
        filtered_pos_2d = self.data['odom_filtered'][['position_x', 'position_y']].values
        
        # Calculate 2D position errors
        noise_pos_2d_error = noise_pos_2d - gt_pos_2d
        filtered_pos_2d_error = filtered_pos_2d - gt_pos_2d
        
        # RMSE for 2D position (Euclidean distance in XY plane)
        noise_pos_2d_rmse = np.sqrt(np.mean(np.sum(noise_pos_2d_error**2, axis=1)))
        filtered_pos_2d_rmse = np.sqrt(np.mean(np.sum(filtered_pos_2d_error**2, axis=1)))
        
        # Calculate individual axis RMSE
        noise_x_rmse = np.sqrt(np.mean(noise_pos_2d_error[:, 0]**2))
        noise_y_rmse = np.sqrt(np.mean(noise_pos_2d_error[:, 1]**2))
        filtered_x_rmse = np.sqrt(np.mean(filtered_pos_2d_error[:, 0]**2))
        filtered_y_rmse = np.sqrt(np.mean(filtered_pos_2d_error[:, 1]**2))
        
        # Extract yaw angle from quaternions
        def quaternion_to_yaw(df):
            """Convert quaternion to yaw angle (around Z-axis)"""
            quats = df[['orientation_x', 'orientation_y', 'orientation_z', 'orientation_w']].values
            r = R.from_quat(quats)
            euler = r.as_euler('xyz', degrees=False)
            return euler[:, 2]  # Yaw angle (around Z)
        
        gt_yaw = quaternion_to_yaw(self.data['odom_gt'])
        noise_yaw = quaternion_to_yaw(self.data['odom_noise'])
        filtered_yaw = quaternion_to_yaw(self.data['odom_filtered'])
        
        # Normalize angles to [-pi, pi]
        def normalize_angle(angle):
            return np.mod(angle + np.pi, 2 * np.pi) - np.pi
        
        gt_yaw_norm = normalize_angle(gt_yaw)
        noise_yaw_norm = normalize_angle(noise_yaw)
        filtered_yaw_norm = normalize_angle(filtered_yaw)
        
        # Calculate yaw errors
        noise_yaw_error = normalize_angle(noise_yaw_norm - gt_yaw_norm)
        filtered_yaw_error = normalize_angle(filtered_yaw_norm - gt_yaw_norm)
        
        # RMSE for yaw
        noise_yaw_rmse = np.sqrt(np.mean(noise_yaw_error**2))
        filtered_yaw_rmse = np.sqrt(np.mean(filtered_yaw_error**2))
        
        # Convert to degrees for display
        noise_yaw_rmse_deg = np.degrees(noise_yaw_rmse)
        filtered_yaw_rmse_deg = np.degrees(filtered_yaw_rmse)
        
        results = {
            'position_2d': {
                'noise_rmse': noise_pos_2d_rmse,
                'filtered_rmse': filtered_pos_2d_rmse,
                'improvement': (noise_pos_2d_rmse - filtered_pos_2d_rmse) / noise_pos_2d_rmse * 100
            },
            'position_x': {
                'noise_rmse': noise_x_rmse,
                'filtered_rmse': filtered_x_rmse,
                'improvement': (noise_x_rmse - filtered_x_rmse) / noise_x_rmse * 100
            },
            'position_y': {
                'noise_rmse': noise_y_rmse,
                'filtered_rmse': filtered_y_rmse,
                'improvement': (noise_y_rmse - filtered_y_rmse) / noise_y_rmse * 100
            },
            'yaw': {
                'noise_rmse_rad': noise_yaw_rmse,
                'filtered_rmse_rad': filtered_yaw_rmse,
                'noise_rmse_deg': noise_yaw_rmse_deg,
                'filtered_rmse_deg': filtered_yaw_rmse_deg,
                'improvement': (noise_yaw_rmse - filtered_yaw_rmse) / noise_yaw_rmse * 100
            }
        }
        
        return results
    
    def plot_2d_trajectory(self, save_fig=True, figsize=(14, 10)):
        """
        Plot 2D trajectory comparison (XY plane only)
        
        Args:
            save_fig: Whether to save the figure
            figsize: Figure size
        """
        if 'odom_gt' not in self.data or 'odom_noise' not in self.data or 'odom_filtered' not in self.data:
            print("Error: Required data not loaded")
            return
        
        fig = plt.figure(figsize=figsize)
        
        # Create subplots
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
        
        # Main XY trajectory plot
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.data['odom_gt']['position_x'], self.data['odom_gt']['position_y'], 
                'g-', linewidth=2.5, label='Ground Truth', alpha=0.8)
        ax1.plot(self.data['odom_noise']['position_x'], self.data['odom_noise']['position_y'], 
                'r-', linewidth=1, label='Noisy', alpha=0.4)
        ax1.plot(self.data['odom_filtered']['position_x'], self.data['odom_filtered']['position_y'], 
                'b-', linewidth=1.8, label='Filtered', alpha=0.7)
        
        # Add start and end markers
        ax1.scatter(self.data['odom_gt']['position_x'].iloc[0], 
                   self.data['odom_gt']['position_y'].iloc[0],
                   c='green', s=150, marker='o', edgecolors='black', label='Start')
        ax1.scatter(self.data['odom_gt']['position_x'].iloc[-1], 
                   self.data['odom_gt']['position_y'].iloc[-1],
                   c='red', s=150, marker='^', edgecolors='black', label='End')
        
        ax1.set_xlabel('X Position (m)', fontsize=12)
        ax1.set_ylabel('Y Position (m)', fontsize=12)
        ax1.set_title('2D Robot Trajectory (XY Plane)', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # Add arrows to show direction
        n_arrows = 10
        indices = np.linspace(0, len(self.data['odom_gt']) - 1, n_arrows, dtype=int)
        for idx in indices:
            ax1.arrow(self.data['odom_gt']['position_x'].iloc[idx],
                     self.data['odom_gt']['position_y'].iloc[idx],
                     0.1 * np.cos(self._get_yaw_at_index(self.data['odom_gt'], idx)),
                     0.1 * np.sin(self._get_yaw_at_index(self.data['odom_gt'], idx)),
                     head_width=0.05, head_length=0.05, fc='green', ec='green', alpha=0.6)
        
        # X position error over time
        ax2 = fig.add_subplot(gs[1, 0])
        time = self.data['odom_gt']['time']
        
        noise_x_error = self.data['odom_noise']['position_x'] - self.data['odom_gt']['position_x']
        filtered_x_error = self.data['odom_filtered']['position_x'] - self.data['odom_gt']['position_x']
        
        ax2.plot(time, noise_x_error, 'r-', label='Noisy Error', alpha=0.6)
        ax2.plot(time, filtered_x_error, 'b-', label='Filtered Error', alpha=0.8)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Time (s)', fontsize=10)
        ax2.set_ylabel('X Error (m)', fontsize=10)
        ax2.set_title('X Position Error', fontsize=11)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Y position error over time
        ax3 = fig.add_subplot(gs[1, 1])
        noise_y_error = self.data['odom_noise']['position_y'] - self.data['odom_gt']['position_y']
        filtered_y_error = self.data['odom_filtered']['position_y'] - self.data['odom_gt']['position_y']
        
        ax3.plot(time, noise_y_error, 'r-', label='Noisy Error', alpha=0.6)
        ax3.plot(time, filtered_y_error, 'b-', label='Filtered Error', alpha=0.8)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax3.set_xlabel('Time (s)', fontsize=10)
        ax3.set_ylabel('Y Error (m)', fontsize=10)
        ax3.set_title('Y Position Error', fontsize=11)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # 2D Euclidean error over time
        ax4 = fig.add_subplot(gs[1, 2])
        gt_pos_2d = self.data['odom_gt'][['position_x', 'position_y']].values
        noise_pos_2d = self.data['odom_noise'][['position_x', 'position_y']].values
        filtered_pos_2d = self.data['odom_filtered'][['position_x', 'position_y']].values
        
        noise_2d_error = np.sqrt(np.sum((noise_pos_2d - gt_pos_2d)**2, axis=1))
        filtered_2d_error = np.sqrt(np.sum((filtered_pos_2d - gt_pos_2d)**2, axis=1))
        
        ax4.plot(time, noise_2d_error, 'r-', label='Noisy Error', alpha=0.6)
        ax4.plot(time, filtered_2d_error, 'b-', label='Filtered Error', alpha=0.8)
        ax4.set_xlabel('Time (s)', fontsize=10)
        ax4.set_ylabel('2D Position Error (m)', fontsize=10)
        ax4.set_title('2D Euclidean Error', fontsize=11)
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig('2d_trajectory_analysis.png', dpi=300, bbox_inches='tight')
            print("Saved 2D trajectory plot as '2d_trajectory_analysis.png'")
        
        plt.show()
    
    def _get_yaw_at_index(self, df, idx):
        """Get yaw angle at specific index"""
        quat = df.iloc[idx][['orientation_x', 'orientation_y', 'orientation_z', 'orientation_w']].values
        r = R.from_quat(quat)
        euler = r.as_euler('xyz', degrees=False)
        return euler[2]  # Yaw angle
    
    def plot_yaw_comparison(self, save_fig=True, figsize=(14, 8)):
        """
        Plot yaw angle comparison and error
        
        Args:
            save_fig: Whether to save the figure
            figsize: Figure size
        """
        if 'odom_gt' not in self.data or 'odom_noise' not in self.data or 'odom_filtered' not in self.data:
            print("Error: Required data not loaded")
            return
        
        # Extract yaw angles
        def get_yaw_angles(df):
            quats = df[['orientation_x', 'orientation_y', 'orientation_z', 'orientation_w']].values
            r = R.from_quat(quats)
            euler = r.as_euler('xyz', degrees=False)
            return euler[:, 2]  # Yaw
        
        gt_yaw = get_yaw_angles(self.data['odom_gt'])
        noise_yaw = get_yaw_angles(self.data['odom_noise'])
        filtered_yaw = get_yaw_angles(self.data['odom_filtered'])
        
        # Convert to degrees for display
        gt_yaw_deg = np.degrees(gt_yaw)
        noise_yaw_deg = np.degrees(noise_yaw)
        filtered_yaw_deg = np.degrees(filtered_yaw)
        
        # Normalize angles for error calculation
        def normalize_angle(angle):
            return np.mod(angle + np.pi, 2 * np.pi) - np.pi
        
        noise_yaw_error = normalize_angle(noise_yaw - gt_yaw)
        filtered_yaw_error = normalize_angle(filtered_yaw - gt_yaw)
        
        time = self.data['odom_gt']['time']
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Yaw angle comparison (radians)
        ax1 = axes[0, 0]
        ax1.plot(time, gt_yaw, 'g-', label='Ground Truth', linewidth=2, alpha=0.8)
        ax1.plot(time, noise_yaw, 'r-', label='Noisy', linewidth=1, alpha=0.4)
        ax1.plot(time, filtered_yaw, 'b-', label='Filtered', linewidth=1.5, alpha=0.7)
        ax1.set_xlabel('Time (s)', fontsize=10)
        ax1.set_ylabel('Yaw Angle (rad)', fontsize=10)
        ax1.set_title('Yaw Angle Comparison (Radians)', fontsize=12)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Yaw angle comparison (degrees)
        ax2 = axes[0, 1]
        ax2.plot(time, gt_yaw_deg, 'g-', label='Ground Truth', linewidth=2, alpha=0.8)
        ax2.plot(time, noise_yaw_deg, 'r-', label='Noisy', linewidth=1, alpha=0.4)
        ax2.plot(time, filtered_yaw_deg, 'b-', label='Filtered', linewidth=1.5, alpha=0.7)
        ax2.set_xlabel('Time (s)', fontsize=10)
        ax2.set_ylabel('Yaw Angle (deg)', fontsize=10)
        ax2.set_title('Yaw Angle Comparison (Degrees)', fontsize=12)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Yaw error (radians)
        ax3 = axes[1, 0]
        ax3.plot(time, noise_yaw_error, 'r-', label='Noisy Error', alpha=0.6)
        ax3.plot(time, filtered_yaw_error, 'b-', label='Filtered Error', alpha=0.8)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax3.set_xlabel('Time (s)', fontsize=10)
        ax3.set_ylabel('Yaw Error (rad)', fontsize=10)
        ax3.set_title('Yaw Angle Error (Radians)', fontsize=12)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # Cumulative yaw error distribution
        ax4 = axes[1, 1]
        ax4.hist(noise_yaw_error, bins=50, alpha=0.5, color='red', label='Noisy', density=True)
        ax4.hist(filtered_yaw_error, bins=50, alpha=0.5, color='blue', label='Filtered', density=True)
        ax4.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax4.set_xlabel('Yaw Error (rad)', fontsize=10)
        ax4.set_ylabel('Density', fontsize=10)
        ax4.set_title('Yaw Error Distribution', fontsize=12)
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig('yaw_analysis.png', dpi=300, bbox_inches='tight')
            print("Saved yaw analysis plot as 'yaw_analysis.png'")
        
        plt.show()
    
    def plot_rmse_comparison(self, rmse_results, save_fig=True, figsize=(12, 8)):
        """
        Create bar plots comparing RMSE values
        
        Args:
            rmse_results: RMSE results from calculate_2d_rmse()
            save_fig: Whether to save the figure
            figsize: Figure size
        """
        if rmse_results is None:
            print("Error: No RMSE results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 2D Position RMSE comparison
        ax1 = axes[0, 0]
        labels = ['Noisy', 'Filtered']
        values = [rmse_results['position_2d']['noise_rmse'], 
                 rmse_results['position_2d']['filtered_rmse']]
        improvement = rmse_results['position_2d']['improvement']
        
        bars = ax1.bar(labels, values, color=['red', 'blue'], alpha=0.7)
        ax1.set_ylabel('RMSE (m)', fontsize=10)
        ax1.set_title(f'2D Position RMSE\nImprovement: {improvement:.1f}%', fontsize=12)
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4f}',
                    ha='center', va='bottom', fontsize=9)
        
        # X Position RMSE comparison
        ax2 = axes[0, 1]
        values_x = [rmse_results['position_x']['noise_rmse'], 
                   rmse_results['position_x']['filtered_rmse']]
        improvement_x = rmse_results['position_x']['improvement']
        
        bars = ax2.bar(labels, values_x, color=['red', 'blue'], alpha=0.7)
        ax2.set_ylabel('RMSE (m)', fontsize=10)
        ax2.set_title(f'X Position RMSE\nImprovement: {improvement_x:.1f}%', fontsize=12)
        ax2.grid(True, axis='y', alpha=0.3)
        
        for bar, value in zip(bars, values_x):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4f}',
                    ha='center', va='bottom', fontsize=9)
        
        # Y Position RMSE comparison
        ax3 = axes[1, 0]
        values_y = [rmse_results['position_y']['noise_rmse'], 
                   rmse_results['position_y']['filtered_rmse']]
        improvement_y = rmse_results['position_y']['improvement']
        
        bars = ax3.bar(labels, values_y, color=['red', 'blue'], alpha=0.7)
        ax3.set_ylabel('RMSE (m)', fontsize=10)
        ax3.set_title(f'Y Position RMSE\nImprovement: {improvement_y:.1f}%', fontsize=12)
        ax3.grid(True, axis='y', alpha=0.3)
        
        for bar, value in zip(bars, values_y):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4f}',
                    ha='center', va='bottom', fontsize=9)
        
        # Yaw RMSE comparison (in degrees)
        ax4 = axes[1, 1]
        values_yaw = [rmse_results['yaw']['noise_rmse_deg'], 
                     rmse_results['yaw']['filtered_rmse_deg']]
        improvement_yaw = rmse_results['yaw']['improvement']
        
        bars = ax4.bar(labels, values_yaw, color=['red', 'blue'], alpha=0.7)
        ax4.set_ylabel('RMSE (degrees)', fontsize=10)
        ax4.set_title(f'Yaw Angle RMSE\nImprovement: {improvement_yaw:.1f}%', fontsize=12)
        ax4.grid(True, axis='y', alpha=0.3)
        
        for bar, value in zip(bars, values_yaw):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}°',
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig('rmse_comparison.png', dpi=300, bbox_inches='tight')
            print("Saved RMSE comparison plot as 'rmse_comparison.png'")
        
        plt.show()
    
    def plot_error_scatter(self, save_fig=True, figsize=(12, 5)):
        """
        Create scatter plots of position errors
        
        Args:
            save_fig: Whether to save the figure
            figsize: Figure size
        """
        if 'odom_gt' not in self.data or 'odom_noise' not in self.data or 'odom_filtered' not in self.data:
            print("Error: Required data not loaded")
            return
        
        # Calculate errors
        noise_x_error = self.data['odom_noise']['position_x'] - self.data['odom_gt']['position_x']
        noise_y_error = self.data['odom_noise']['position_y'] - self.data['odom_gt']['position_y']
        filtered_x_error = self.data['odom_filtered']['position_x'] - self.data['odom_gt']['position_x']
        filtered_y_error = self.data['odom_filtered']['position_y'] - self.data['odom_gt']['position_y']
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Noisy error scatter
        ax1 = axes[0]
        scatter1 = ax1.scatter(noise_x_error, noise_y_error, 
                              c=self.data['odom_gt']['time'], 
                              cmap='viridis', alpha=0.6, s=20)
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax1.set_xlabel('X Error (m)', fontsize=10)
        ax1.set_ylabel('Y Error (m)', fontsize=10)
        ax1.set_title('Noisy Odometry Errors', fontsize=12)
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label='Time (s)')
        
        # Filtered error scatter
        ax2 = axes[1]
        scatter2 = ax2.scatter(filtered_x_error, filtered_y_error, 
                              c=self.data['odom_gt']['time'], 
                              cmap='viridis', alpha=0.6, s=20)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax2.set_xlabel('X Error (m)', fontsize=10)
        ax2.set_ylabel('Y Error (m)', fontsize=10)
        ax2.set_title('Filtered Odometry Errors', fontsize=12)
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='Time (s)')
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig('error_scatter_plots.png', dpi=300, bbox_inches='tight')
            print("Saved error scatter plots as 'error_scatter_plots.png'")
        
        plt.show()
    
    def generate_summary_report(self, rmse_results, save_report=True):
        """
        Generate and display a 2D-focused summary report
        
        Args:
            rmse_results: RMSE results from calculate_2d_rmse()
            save_report: Whether to save report to file
        """
        if rmse_results is None:
            print("No RMSE results to report")
            return
        
        report = []
        report.append("=" * 70)
        report.append("2D ODOMETRY FILTERING PERFORMANCE REPORT (XY Plane)")
        report.append("=" * 70)
        report.append("")
        
        report.append("POSITION ACCURACY - 2D RMSE:")
        report.append("-" * 50)
        report.append(f"Noisy Odometry:     {rmse_results['position_2d']['noise_rmse']:.6f} m")
        report.append(f"Filtered Odometry:  {rmse_results['position_2d']['filtered_rmse']:.6f} m")
        report.append(f"Improvement:        {rmse_results['position_2d']['improvement']:.2f} %")
        report.append("")
        
        report.append("POSITION ACCURACY - Individual Axes:")
        report.append("-" * 50)
        report.append("X-Axis:")
        report.append(f"  Noisy RMSE:      {rmse_results['position_x']['noise_rmse']:.6f} m")
        report.append(f"  Filtered RMSE:   {rmse_results['position_x']['filtered_rmse']:.6f} m")
        report.append(f"  Improvement:     {rmse_results['position_x']['improvement']:.2f} %")
        report.append("")
        report.append("Y-Axis:")
        report.append(f"  Noisy RMSE:      {rmse_results['position_y']['noise_rmse']:.6f} m")
        report.append(f"  Filtered RMSE:   {rmse_results['position_y']['filtered_rmse']:.6f} m")
        report.append(f"  Improvement:     {rmse_results['position_y']['improvement']:.2f} %")
        report.append("")
        
        report.append("YAW ANGLE ACCURACY:")
        report.append("-" * 50)
        report.append(f"Noisy Odometry:     {rmse_results['yaw']['noise_rmse_rad']:.6f} rad")
        report.append(f"                        ({rmse_results['yaw']['noise_rmse_deg']:.2f}°)")
        report.append(f"Filtered Odometry:  {rmse_results['yaw']['filtered_rmse_rad']:.6f} rad")
        report.append(f"                        ({rmse_results['yaw']['filtered_rmse_deg']:.2f}°)")
        report.append(f"Improvement:        {rmse_results['yaw']['improvement']:.2f} %")
        report.append("")
        
        report.append("DATA STATISTICS:")
        report.append("-" * 50)
        for key, df in self.data.items():
            if key in ['odom_gt', 'odom_noise', 'odom_filtered']:
                report.append(f"{key.upper()}:")
                report.append(f"  Duration: {df['time'].max():.2f} seconds")
                report.append(f"  Samples:  {len(df)}")
                report.append(f"  X range:  [{df['position_x'].min():.3f}, {df['position_x'].max():.3f}] m")
                report.append(f"  Y range:  [{df['position_y'].min():.3f}, {df['position_y'].max():.3f}] m")
                report.append(f"  Path length: {self._calculate_path_length(df):.3f} m")
        
        report.append("")
        report.append("=" * 70)
        
        # Print report
        for line in report:
            print(line)
        
        # Save report to file
        if save_report:
            with open('2d_odometry_analysis_report.txt', 'w') as f:
                f.write('\n'.join(report))
            print("\nReport saved as '2d_odometry_analysis_report.txt'")
    
    def _calculate_path_length(self, df):
        """Calculate total path length from position data"""
        x_diff = np.diff(df['position_x'].values)
        y_diff = np.diff(df['position_y'].values)
        segment_lengths = np.sqrt(x_diff**2 + y_diff**2)
        return np.sum(segment_lengths)
    
    def run_full_2d_analysis(self):
        """
        Run complete 2D analysis pipeline
        """
        print("Loading resampled data...")
        self.load_resampled_data()
        
        print("\nCalculating 2D RMSE...")
        rmse_results = self.calculate_2d_rmse()
        
        if rmse_results:
            print("\nGenerating 2D analysis report...")
            self.generate_summary_report(rmse_results)
            
            print("\nPlotting 2D trajectory analysis...")
            self.plot_2d_trajectory()
            
            print("\nPlotting yaw analysis...")
            self.plot_yaw_comparison()
            
            print("\nPlotting RMSE comparison...")
            self.plot_rmse_comparison(rmse_results)
            
            print("\nPlotting error scatter plots...")
            self.plot_error_scatter()
            
            print("\n2D analysis complete!")
            print("\nGenerated files:")
            print("  1. 2d_trajectory_analysis.png")
            print("  2. yaw_analysis.png")
            print("  3. rmse_comparison.png")
            print("  4. error_scatter_plots.png")
            print("  5. 2d_odometry_analysis_report.txt")
        else:
            print("Failed to calculate RMSE. Check if required data is available.")


def main():
    """
    Main function to run the 2D analysis
    """
    # Check if resampled files exist
    required_files = [
        'odom_gt_data_resampled.csv',
        'odom_noise_data_resampled.csv',
        'odom_filtered_data_resampled.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("Error: Required resampled files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease run the resampling script first:")
        print("  python your_resampling_script.py")
        return
    
    # Create analyzer and run 2D analysis
    analyzer = OdometryAnalyzer2D()
    analyzer.run_full_2d_analysis()


if __name__ == "__main__":
    # Set matplotlib style for better plots
    plt.style.use('seaborn-v0_8-darkgrid')
    
    main()