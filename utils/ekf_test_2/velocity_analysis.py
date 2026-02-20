#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter
from scipy.stats import gaussian_kde
import seaborn as sns

class SpeedAnalyzer:
    def __init__(self, data_dir="."):
        """
        Initialize the speed analyzer
        
        Args:
            data_dir: Directory containing the resampled CSV files
        """
        self.data_dir = data_dir
        self.data = {}
        
    def load_resampled_data(self):
        """
        Load resampled odometry data files
        """
        files = {
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
    
    def calculate_2d_velocity_metrics(self):
        """
        Calculate comprehensive 2D velocity metrics
        
        Returns:
            dict: Various velocity metrics for noisy and filtered data
        """
        if 'odom_gt' not in self.data or 'odom_noise' not in self.data or 'odom_filtered' not in self.data:
            print("Error: Required data not loaded")
            return None
        
        # Calculate 2D velocity magnitude (speed) for each dataset
        def calculate_speed(df):
            """Calculate 2D speed from linear velocities"""
            return np.sqrt(df['linear_velocity_x']**2 + df['linear_velocity_y']**2)
        
        # Calculate yaw rate (angular velocity around Z)
        def calculate_yaw_rate(df):
            """Extract yaw rate from angular velocity"""
            return df['angular_velocity_z']
        
        # Calculate metrics
        metrics = {}
        
        for name in ['gt', 'noise', 'filtered']:
            df = self.data[f'odom_{name}']
            time = df['time']
            
            # Speed calculations
            speed = calculate_speed(df)
            yaw_rate = calculate_yaw_rate(df)
            
            # Linear velocity components
            vx = df['linear_velocity_x']
            vy = df['linear_velocity_y']
            
            # Acceleration calculations (derivative of velocity)
            dt = np.diff(time)
            ax = np.diff(vx) / dt
            ay = np.diff(vy) / dt
            acceleration_magnitude = np.sqrt(ax**2 + ay**2)
            
            # Jerk calculations (derivative of acceleration)
            if len(ax) > 1:
                jerk_magnitude = np.sqrt(np.diff(ax)**2 + np.diff(ay)**2) / dt[1:]
            else:
                jerk_magnitude = np.array([])
            
            metrics[name] = {
                'time': time,
                'speed': speed,
                'yaw_rate': yaw_rate,
                'vx': vx,
                'vy': vy,
                'ax': ax,
                'ay': ay,
                'acceleration_magnitude': acceleration_magnitude,
                'jerk_magnitude': jerk_magnitude,
                'speed_stats': {
                    'mean': np.mean(speed),
                    'std': np.std(speed),
                    'max': np.max(speed),
                    'min': np.min(speed),
                    'rms': np.sqrt(np.mean(speed**2))
                },
                'yaw_rate_stats': {
                    'mean': np.mean(yaw_rate),
                    'std': np.std(yaw_rate),
                    'max': np.max(yaw_rate),
                    'min': np.min(yaw_rate),
                    'rms': np.sqrt(np.mean(yaw_rate**2))
                }
            }
        
        # Calculate velocity errors relative to ground truth
        metrics['errors'] = {
            'speed_noise_error': metrics['noise']['speed'] - metrics['gt']['speed'],
            'speed_filtered_error': metrics['filtered']['speed'] - metrics['gt']['speed'],
            'yaw_rate_noise_error': metrics['noise']['yaw_rate'] - metrics['gt']['yaw_rate'],
            'yaw_rate_filtered_error': metrics['filtered']['yaw_rate'] - metrics['gt']['yaw_rate'],
            'vx_noise_error': metrics['noise']['vx'] - metrics['gt']['vx'],
            'vx_filtered_error': metrics['filtered']['vx'] - metrics['gt']['vx'],
            'vy_noise_error': metrics['noise']['vy'] - metrics['gt']['vy'],
            'vy_filtered_error': metrics['filtered']['vy'] - metrics['gt']['vy']
        }
        
        # Calculate RMSE for velocity
        metrics['rmse'] = {
            'speed_noise': np.sqrt(np.mean(metrics['errors']['speed_noise_error']**2)),
            'speed_filtered': np.sqrt(np.mean(metrics['errors']['speed_filtered_error']**2)),
            'yaw_rate_noise': np.sqrt(np.mean(metrics['errors']['yaw_rate_noise_error']**2)),
            'yaw_rate_filtered': np.sqrt(np.mean(metrics['errors']['yaw_rate_filtered_error']**2)),
            'vx_noise': np.sqrt(np.mean(metrics['errors']['vx_noise_error']**2)),
            'vx_filtered': np.sqrt(np.mean(metrics['errors']['vx_filtered_error']**2)),
            'vy_noise': np.sqrt(np.mean(metrics['errors']['vy_noise_error']**2)),
            'vy_filtered': np.sqrt(np.mean(metrics['errors']['vy_filtered_error']**2))
        }
        
        # Calculate improvements
        metrics['improvement'] = {
            'speed': (metrics['rmse']['speed_noise'] - metrics['rmse']['speed_filtered']) / metrics['rmse']['speed_noise'] * 100,
            'yaw_rate': (metrics['rmse']['yaw_rate_noise'] - metrics['rmse']['yaw_rate_filtered']) / metrics['rmse']['yaw_rate_noise'] * 100,
            'vx': (metrics['rmse']['vx_noise'] - metrics['rmse']['vx_filtered']) / metrics['rmse']['vx_noise'] * 100,
            'vy': (metrics['rmse']['vy_noise'] - metrics['rmse']['vy_filtered']) / metrics['rmse']['vy_noise'] * 100
        }
        
        return metrics
    
    def plot_speed_comparison(self, metrics, save_fig=True, figsize=(14, 10)):
        """
        Plot speed comparison and analysis
        
        Args:
            metrics: Calculated velocity metrics
            save_fig: Whether to save the figure
            figsize: Figure size
        """
        if metrics is None:
            print("Error: No metrics to plot")
            return
        
        fig = plt.figure(figsize=figsize)
        
        # Create subplots
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])
        
        # Speed comparison plot
        ax1 = fig.add_subplot(gs[0, :])
        time_gt = metrics['gt']['time']
        
        ax1.plot(time_gt, metrics['gt']['speed'], 'g-', label='Ground Truth', linewidth=2.5, alpha=0.8)
        ax1.plot(time_gt, metrics['noise']['speed'], 'r-', label='Noisy', linewidth=1, alpha=0.4)
        ax1.plot(time_gt, metrics['filtered']['speed'], 'b-', label='Filtered', linewidth=1.5, alpha=0.7)
        ax1.set_xlabel('Time (s)', fontsize=12)
        ax1.set_ylabel('Speed (m/s)', fontsize=12)
        ax1.set_title('2D Speed Comparison', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Speed error plot
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(time_gt, metrics['errors']['speed_noise_error'], 'r-', label='Noisy Error', alpha=0.6)
        ax2.plot(time_gt, metrics['errors']['speed_filtered_error'], 'b-', label='Filtered Error', alpha=0.8)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Time (s)', fontsize=10)
        ax2.set_ylabel('Speed Error (m/s)', fontsize=10)
        ax2.set_title('Speed Error Over Time', fontsize=12)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Speed histogram
        ax3 = fig.add_subplot(gs[1, 1])
        bins = np.linspace(
            min(metrics['gt']['speed'].min(), metrics['noise']['speed'].min(), metrics['filtered']['speed'].min()),
            max(metrics['gt']['speed'].max(), metrics['noise']['speed'].max(), metrics['filtered']['speed'].max()),
            50
        )
        ax3.hist(metrics['gt']['speed'], bins=bins, alpha=0.5, color='green', label='Ground Truth', density=True)
        ax3.hist(metrics['noise']['speed'], bins=bins, alpha=0.5, color='red', label='Noisy', density=True)
        ax3.hist(metrics['filtered']['speed'], bins=bins, alpha=0.5, color='blue', label='Filtered', density=True)
        ax3.set_xlabel('Speed (m/s)', fontsize=10)
        ax3.set_ylabel('Density', fontsize=10)
        ax3.set_title('Speed Distribution', fontsize=12)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # Velocity vector plot (scatter of vx vs vy)
        ax4 = fig.add_subplot(gs[2, 0])
        # Sample points for cleaner visualization
        sample_step = max(1, len(time_gt) // 100)
        indices = range(0, len(time_gt), sample_step)
        
        # Ground truth velocity vectors
        for i in indices:
            ax4.arrow(0, 0, metrics['gt']['vx'].iloc[i], metrics['gt']['vy'].iloc[i], 
                     head_width=0.02, head_length=0.02, fc='green', ec='green', alpha=0.3)
        
        ax4.set_xlabel('Vx (m/s)', fontsize=10)
        ax4.set_ylabel('Vy (m/s)', fontsize=10)
        ax4.set_title('Velocity Vector Field (Ground Truth)', fontsize=12)
        ax4.grid(True, alpha=0.3)
        ax4.axis('equal')
        
        # Speed power spectral density
        ax5 = fig.add_subplot(gs[2, 1])
        
        # Calculate PSD using FFT
        def calculate_psd(signal, dt):
            n = len(signal)
            fft_vals = np.fft.fft(signal)
            psd = np.abs(fft_vals)**2 / (n * dt)
            freqs = np.fft.fftfreq(n, dt)
            return freqs[:n//2], psd[:n//2]
        
        dt = np.mean(np.diff(time_gt))
        if dt > 0:
            freqs_gt, psd_gt = calculate_psd(metrics['gt']['speed'], dt)
            freqs_noise, psd_noise = calculate_psd(metrics['noise']['speed'], dt)
            freqs_filtered, psd_filtered = calculate_psd(metrics['filtered']['speed'], dt)
            
            ax5.loglog(freqs_gt[1:], psd_gt[1:], 'g-', label='Ground Truth', alpha=0.8)
            ax5.loglog(freqs_noise[1:], psd_noise[1:], 'r-', label='Noisy', alpha=0.5)
            ax5.loglog(freqs_filtered[1:], psd_filtered[1:], 'b-', label='Filtered', alpha=0.7)
            ax5.set_xlabel('Frequency (Hz)', fontsize=10)
            ax5.set_ylabel('PSD', fontsize=10)
            ax5.set_title('Speed Power Spectral Density', fontsize=12)
            ax5.legend(fontsize=9)
            ax5.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig('speed_analysis.png', dpi=300, bbox_inches='tight')
            print("Saved speed analysis plot as 'speed_analysis.png'")
        
        plt.show()
    
    def plot_yaw_rate_comparison(self, metrics, save_fig=True, figsize=(14, 8)):
        """
        Plot yaw rate comparison and analysis
        
        Args:
            metrics: Calculated velocity metrics
            save_fig: Whether to save the figure
            figsize: Figure size
        """
        if metrics is None:
            print("Error: No metrics to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        time_gt = metrics['gt']['time']
        
        # Yaw rate comparison
        ax1 = axes[0, 0]
        ax1.plot(time_gt, metrics['gt']['yaw_rate'], 'g-', label='Ground Truth', linewidth=2, alpha=0.8)
        ax1.plot(time_gt, metrics['noise']['yaw_rate'], 'r-', label='Noisy', linewidth=1, alpha=0.4)
        ax1.plot(time_gt, metrics['filtered']['yaw_rate'], 'b-', label='Filtered', linewidth=1.5, alpha=0.7)
        ax1.set_xlabel('Time (s)', fontsize=10)
        ax1.set_ylabel('Yaw Rate (rad/s)', fontsize=10)
        ax1.set_title('Yaw Rate Comparison', fontsize=12)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Yaw rate error
        ax2 = axes[0, 1]
        ax2.plot(time_gt, metrics['errors']['yaw_rate_noise_error'], 'r-', label='Noisy Error', alpha=0.6)
        ax2.plot(time_gt, metrics['errors']['yaw_rate_filtered_error'], 'b-', label='Filtered Error', alpha=0.8)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Time (s)', fontsize=10)
        ax2.set_ylabel('Yaw Rate Error (rad/s)', fontsize=10)
        ax2.set_title('Yaw Rate Error', fontsize=12)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Yaw rate histogram
        ax3 = axes[1, 0]
        bins = np.linspace(
            min(metrics['gt']['yaw_rate'].min(), metrics['noise']['yaw_rate'].min(), metrics['filtered']['yaw_rate'].min()),
            max(metrics['gt']['yaw_rate'].max(), metrics['noise']['yaw_rate'].max(), metrics['filtered']['yaw_rate'].max()),
            50
        )
        ax3.hist(metrics['gt']['yaw_rate'], bins=bins, alpha=0.5, color='green', label='Ground Truth', density=True)
        ax3.hist(metrics['noise']['yaw_rate'], bins=bins, alpha=0.5, color='red', label='Noisy', density=True)
        ax3.hist(metrics['filtered']['yaw_rate'], bins=bins, alpha=0.5, color='blue', label='Filtered', density=True)
        ax3.set_xlabel('Yaw Rate (rad/s)', fontsize=10)
        ax3.set_ylabel('Density', fontsize=10)
        ax3.set_title('Yaw Rate Distribution', fontsize=12)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # Yaw rate vs speed scatter
        ax4 = axes[1, 1]
        scatter1 = ax4.scatter(metrics['noise']['speed'], metrics['noise']['yaw_rate'], 
                              c='red', alpha=0.3, s=10, label='Noisy')
        scatter2 = ax4.scatter(metrics['filtered']['speed'], metrics['filtered']['yaw_rate'], 
                              c='blue', alpha=0.5, s=10, label='Filtered')
        scatter3 = ax4.scatter(metrics['gt']['speed'], metrics['gt']['yaw_rate'], 
                              c='green', alpha=0.5, s=10, label='Ground Truth')
        ax4.set_xlabel('Speed (m/s)', fontsize=10)
        ax4.set_ylabel('Yaw Rate (rad/s)', fontsize=10)
        ax4.set_title('Speed vs Yaw Rate', fontsize=12)
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig('yaw_rate_analysis.png', dpi=300, bbox_inches='tight')
            print("Saved yaw rate analysis plot as 'yaw_rate_analysis.png'")
        
        plt.show()
    
    def plot_velocity_components(self, metrics, save_fig=True, figsize=(14, 8)):
        """
        Plot individual velocity components (vx, vy)
        
        Args:
            metrics: Calculated velocity metrics
            save_fig: Whether to save the figure
            figsize: Figure size
        """
        if metrics is None:
            print("Error: No metrics to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        time_gt = metrics['gt']['time']
        
        # Vx comparison
        ax1 = axes[0, 0]
        ax1.plot(time_gt, metrics['gt']['vx'], 'g-', label='Ground Truth', linewidth=2, alpha=0.8)
        ax1.plot(time_gt, metrics['noise']['vx'], 'r-', label='Noisy', linewidth=1, alpha=0.4)
        ax1.plot(time_gt, metrics['filtered']['vx'], 'b-', label='Filtered', linewidth=1.5, alpha=0.7)
        ax1.set_xlabel('Time (s)', fontsize=10)
        ax1.set_ylabel('Vx (m/s)', fontsize=10)
        ax1.set_title('X Velocity Component', fontsize=12)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Vy comparison
        ax2 = axes[0, 1]
        ax2.plot(time_gt, metrics['gt']['vy'], 'g-', label='Ground Truth', linewidth=2, alpha=0.8)
        ax2.plot(time_gt, metrics['noise']['vy'], 'r-', label='Noisy', linewidth=1, alpha=0.4)
        ax2.plot(time_gt, metrics['filtered']['vy'], 'b-', label='Filtered', linewidth=1.5, alpha=0.7)
        ax2.set_xlabel('Time (s)', fontsize=10)
        ax2.set_ylabel('Vy (m/s)', fontsize=10)
        ax2.set_title('Y Velocity Component', fontsize=12)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Vx error
        ax3 = axes[1, 0]
        ax3.plot(time_gt, metrics['errors']['vx_noise_error'], 'r-', label='Noisy Error', alpha=0.6)
        ax3.plot(time_gt, metrics['errors']['vx_filtered_error'], 'b-', label='Filtered Error', alpha=0.8)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax3.set_xlabel('Time (s)', fontsize=10)
        ax3.set_ylabel('Vx Error (m/s)', fontsize=10)
        ax3.set_title('X Velocity Error', fontsize=12)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # Vy error
        ax4 = axes[1, 1]
        ax4.plot(time_gt, metrics['errors']['vy_noise_error'], 'r-', label='Noisy Error', alpha=0.6)
        ax4.plot(time_gt, metrics['errors']['vy_filtered_error'], 'b-', label='Filtered Error', alpha=0.8)
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax4.set_xlabel('Time (s)', fontsize=10)
        ax4.set_ylabel('Vy Error (m/s)', fontsize=10)
        ax4.set_title('Y Velocity Error', fontsize=12)
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig('velocity_components_analysis.png', dpi=300, bbox_inches='tight')
            print("Saved velocity components plot as 'velocity_components_analysis.png'")
        
        plt.show()
    
    def plot_acceleration_analysis(self, metrics, save_fig=True, figsize=(14, 10)):
        """
        Plot acceleration analysis
        
        Args:
            metrics: Calculated velocity metrics
            save_fig: Whether to save the figure
            figsize: Figure size
        """
        if metrics is None:
            print("Error: No metrics to plot")
            return
        
        fig = plt.figure(figsize=figsize)
        
        # Create subplots
        gs = fig.add_gridspec(2, 2)
        
        # Acceleration magnitude comparison
        ax1 = fig.add_subplot(gs[0, 0])
        time_acc = metrics['gt']['time'].iloc[:-1]  # Acceleration has one less point
        
        ax1.plot(time_acc, metrics['gt']['acceleration_magnitude'], 'g-', 
                label='Ground Truth', linewidth=2, alpha=0.8)
        ax1.plot(time_acc, metrics['noise']['acceleration_magnitude'], 'r-', 
                label='Noisy', linewidth=1, alpha=0.4)
        ax1.plot(time_acc, metrics['filtered']['acceleration_magnitude'], 'b-', 
                label='Filtered', linewidth=1.5, alpha=0.7)
        ax1.set_xlabel('Time (s)', fontsize=10)
        ax1.set_ylabel('Acceleration (m/s²)', fontsize=10)
        ax1.set_title('Acceleration Magnitude', fontsize=12)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Acceleration histogram
        ax2 = fig.add_subplot(gs[0, 1])
        bins = np.linspace(0, max(
            metrics['gt']['acceleration_magnitude'].max(),
            metrics['noise']['acceleration_magnitude'].max(),
            metrics['filtered']['acceleration_magnitude'].max()
        ), 50)
        
        ax2.hist(metrics['gt']['acceleration_magnitude'], bins=bins, 
                alpha=0.5, color='green', label='Ground Truth', density=True)
        ax2.hist(metrics['noise']['acceleration_magnitude'], bins=bins, 
                alpha=0.5, color='red', label='Noisy', density=True)
        ax2.hist(metrics['filtered']['acceleration_magnitude'], bins=bins, 
                alpha=0.5, color='blue', label='Filtered', density=True)
        ax2.set_xlabel('Acceleration (m/s²)', fontsize=10)
        ax2.set_ylabel('Density', fontsize=10)
        ax2.set_title('Acceleration Distribution', fontsize=12)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Jerk magnitude (if available)
        if len(metrics['gt']['jerk_magnitude']) > 0:
            ax3 = fig.add_subplot(gs[1, 0])
            time_jerk = metrics['gt']['time'].iloc[:-2]  # Jerk has two less points
            
            ax3.plot(time_jerk, metrics['gt']['jerk_magnitude'], 'g-', 
                    label='Ground Truth', linewidth=2, alpha=0.8)
            ax3.plot(time_jerk, metrics['noise']['jerk_magnitude'], 'r-', 
                    label='Noisy', linewidth=1, alpha=0.4)
            ax3.plot(time_jerk, metrics['filtered']['jerk_magnitude'], 'b-', 
                    label='Filtered', linewidth=1.5, alpha=0.7)
            ax3.set_xlabel('Time (s)', fontsize=10)
            ax3.set_ylabel('Jerk (m/s³)', fontsize=10)
            ax3.set_title('Jerk Magnitude', fontsize=12)
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)
        
        # Acceleration vs Speed scatter
        ax4 = fig.add_subplot(gs[1, 1])
        scatter1 = ax4.scatter(metrics['noise']['speed'].iloc[:-1], 
                              metrics['noise']['acceleration_magnitude'],
                              c='red', alpha=0.3, s=10, label='Noisy')
        scatter2 = ax4.scatter(metrics['filtered']['speed'].iloc[:-1], 
                              metrics['filtered']['acceleration_magnitude'],
                              c='blue', alpha=0.5, s=10, label='Filtered')
        scatter3 = ax4.scatter(metrics['gt']['speed'].iloc[:-1], 
                              metrics['gt']['acceleration_magnitude'],
                              c='green', alpha=0.5, s=10, label='Ground Truth')
        ax4.set_xlabel('Speed (m/s)', fontsize=10)
        ax4.set_ylabel('Acceleration (m/s²)', fontsize=10)
        ax4.set_title('Speed vs Acceleration', fontsize=12)
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig('acceleration_analysis.png', dpi=300, bbox_inches='tight')
            print("Saved acceleration analysis plot as 'acceleration_analysis.png'")
        
        plt.show()
    
    def plot_velocity_rmse_comparison(self, metrics, save_fig=True, figsize=(12, 8)):
        """
        Create bar plots comparing velocity RMSE values
        
        Args:
            metrics: Calculated velocity metrics
            save_fig: Whether to save the figure
            figsize: Figure size
        """
        if metrics is None or 'rmse' not in metrics:
            print("Error: No RMSE data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Speed RMSE
        ax1 = axes[0, 0]
        labels = ['Noisy', 'Filtered']
        values = [metrics['rmse']['speed_noise'], metrics['rmse']['speed_filtered']]
        improvement = metrics['improvement']['speed']
        
        bars = ax1.bar(labels, values, color=['red', 'blue'], alpha=0.7)
        ax1.set_ylabel('RMSE (m/s)', fontsize=10)
        ax1.set_title(f'Speed RMSE\nImprovement: {improvement:.1f}%', fontsize=12)
        ax1.grid(True, axis='y', alpha=0.3)
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4f}',
                    ha='center', va='bottom', fontsize=9)
        
        # Yaw Rate RMSE
        ax2 = axes[0, 1]
        values = [metrics['rmse']['yaw_rate_noise'], metrics['rmse']['yaw_rate_filtered']]
        improvement = metrics['improvement']['yaw_rate']
        
        bars = ax2.bar(labels, values, color=['red', 'blue'], alpha=0.7)
        ax2.set_ylabel('RMSE (rad/s)', fontsize=10)
        ax2.set_title(f'Yaw Rate RMSE\nImprovement: {improvement:.1f}%', fontsize=12)
        ax2.grid(True, axis='y', alpha=0.3)
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4f}',
                    ha='center', va='bottom', fontsize=9)
        
        # Vx RMSE
        ax3 = axes[1, 0]
        values = [metrics['rmse']['vx_noise'], metrics['rmse']['vx_filtered']]
        improvement = metrics['improvement']['vx']
        
        bars = ax3.bar(labels, values, color=['red', 'blue'], alpha=0.7)
        ax3.set_ylabel('RMSE (m/s)', fontsize=10)
        ax3.set_title(f'Vx RMSE\nImprovement: {improvement:.1f}%', fontsize=12)
        ax3.grid(True, axis='y', alpha=0.3)
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4f}',
                    ha='center', va='bottom', fontsize=9)
        
        # Vy RMSE
        ax4 = axes[1, 1]
        values = [metrics['rmse']['vy_noise'], metrics['rmse']['vy_filtered']]
        improvement = metrics['improvement']['vy']
        
        bars = ax4.bar(labels, values, color=['red', 'blue'], alpha=0.7)
        ax4.set_ylabel('RMSE (m/s)', fontsize=10)
        ax4.set_title(f'Vy RMSE\nImprovement: {improvement:.1f}%', fontsize=12)
        ax4.grid(True, axis='y', alpha=0.3)
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4f}',
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig('velocity_rmse_comparison.png', dpi=300, bbox_inches='tight')
            print("Saved velocity RMSE comparison plot as 'velocity_rmse_comparison.png'")
        
        plt.show()
    
    def generate_speed_report(self, metrics, save_report=True):
        """
        Generate comprehensive speed analysis report
        
        Args:
            metrics: Calculated velocity metrics
            save_report: Whether to save report to file
        """
        if metrics is None:
            print("No metrics to report")
            return
        
        report = []
        report.append("=" * 70)
        report.append("ROBOT SPEED ANALYSIS REPORT")
        report.append("=" * 70)
        report.append("")
        
        report.append("SPEED (2D VELOCITY MAGNITUDE) STATISTICS:")
        report.append("-" * 50)
        for name, label in [('gt', 'Ground Truth'), ('noise', 'Noisy'), ('filtered', 'Filtered')]:
            stats = metrics[name]['speed_stats']
            report.append(f"{label}:")
            report.append(f"  Mean:       {stats['mean']:.4f} m/s")
            report.append(f"  Std Dev:    {stats['std']:.4f} m/s")
            report.append(f"  Max:        {stats['max']:.4f} m/s")
            report.append(f"  Min:        {stats['min']:.4f} m/s")
            report.append(f"  RMS:        {stats['rms']:.4f} m/s")
        
        report.append("")
        report.append("YAW RATE STATISTICS:")
        report.append("-" * 50)
        for name, label in [('gt', 'Ground Truth'), ('noise', 'Noisy'), ('filtered', 'Filtered')]:
            stats = metrics[name]['yaw_rate_stats']
            report.append(f"{label}:")
            report.append(f"  Mean:       {stats['mean']:.4f} rad/s")
            report.append(f"  Std Dev:    {stats['std']:.4f} rad/s")
            report.append(f"  Max:        {stats['max']:.4f} rad/s")
            report.append(f"  Min:        {stats['min']:.4f} rad/s")
            report.append(f"  RMS:        {stats['rms']:.4f} rad/s")
        
        report.append("")
        report.append("VELOCITY ACCURACY (RMSE):")
        report.append("-" * 50)
        report.append(f"Speed:")
        report.append(f"  Noisy RMSE:      {metrics['rmse']['speed_noise']:.6f} m/s")
        report.append(f"  Filtered RMSE:   {metrics['rmse']['speed_filtered']:.6f} m/s")
        report.append(f"  Improvement:     {metrics['improvement']['speed']:.2f} %")
        report.append("")
        report.append(f"Yaw Rate:")
        report.append(f"  Noisy RMSE:      {metrics['rmse']['yaw_rate_noise']:.6f} rad/s")
        report.append(f"  Filtered RMSE:   {metrics['rmse']['yaw_rate_filtered']:.6f} rad/s")
        report.append(f"  Improvement:     {metrics['improvement']['yaw_rate']:.2f} %")
        report.append("")
        report.append(f"X Velocity:")
        report.append(f"  Noisy RMSE:      {metrics['rmse']['vx_noise']:.6f} m/s")
        report.append(f"  Filtered RMSE:   {metrics['rmse']['vx_filtered']:.6f} m/s")
        report.append(f"  Improvement:     {metrics['improvement']['vx']:.2f} %")
        report.append("")
        report.append(f"Y Velocity:")
        report.append(f"  Noisy RMSE:      {metrics['rmse']['vy_noise']:.6f} m/s")
        report.append(f"  Filtered RMSE:   {metrics['rmse']['vy_filtered']:.6f} m/s")
        report.append(f"  Improvement:     {metrics['improvement']['vy']:.2f} %")
        
        report.append("")
        report.append("MOTION CHARACTERISTICS:")
        report.append("-" * 50)
        # Calculate motion statistics
        total_time = metrics['gt']['time'].iloc[-1] - metrics['gt']['time'].iloc[0]
        total_distance = np.trapz(metrics['gt']['speed'], metrics['gt']['time'])
        average_speed = total_distance / total_time if total_time > 0 else 0
        
        report.append(f"Total Travel Time:      {total_time:.2f} s")
        report.append(f"Total Distance:         {total_distance:.2f} m")
        report.append(f"Average Speed:          {average_speed:.4f} m/s")
        report.append(f"Maximum Acceleration:   {metrics['gt']['acceleration_magnitude'].max():.4f} m/s²")
        if len(metrics['gt']['jerk_magnitude']) > 0:
            report.append(f"Maximum Jerk:           {metrics['gt']['jerk_magnitude'].max():.4f} m/s³")
        
        report.append("")
        report.append("=" * 70)
        
        # Print report
        for line in report:
            print(line)
        
        # Save report to file
        if save_report:
            with open('speed_analysis_report.txt', 'w') as f:
                f.write('\n'.join(report))
            print("\nReport saved as 'speed_analysis_report.txt'")
    
    def run_full_speed_analysis(self):
        """
        Run complete speed analysis pipeline
        """
        print("Loading resampled data...")
        self.load_resampled_data()
        
        print("\nCalculating velocity metrics...")
        metrics = self.calculate_2d_velocity_metrics()
        
        if metrics:
            print("\nGenerating speed analysis report...")
            self.generate_speed_report(metrics)
            
            print("\nPlotting speed analysis...")
            self.plot_speed_comparison(metrics)
            
            print("\nPlotting yaw rate analysis...")
            self.plot_yaw_rate_comparison(metrics)
            
            print("\nPlotting velocity components...")
            self.plot_velocity_components(metrics)
            
            print("\nPlotting acceleration analysis...")
            self.plot_acceleration_analysis(metrics)
            
            print("\nPlotting velocity RMSE comparison...")
            self.plot_velocity_rmse_comparison(metrics)
            
            print("\nSpeed analysis complete!")
            print("\nGenerated files:")
            print("  1. speed_analysis.png")
            print("  2. yaw_rate_analysis.png")
            print("  3. velocity_components_analysis.png")
            print("  4. acceleration_analysis.png")
            print("  5. velocity_rmse_comparison.png")
            print("  6. speed_analysis_report.txt")
        else:
            print("Failed to calculate metrics. Check if required data is available.")


def main():
    """
    Main function to run the speed analysis
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
    
    # Create analyzer and run speed analysis
    analyzer = SpeedAnalyzer()
    analyzer.run_full_speed_analysis()


if __name__ == "__main__":
    # Set matplotlib style for better plots
    plt.style.use('seaborn-v0_8-darkgrid')
    
    main()