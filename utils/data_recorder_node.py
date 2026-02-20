#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
import csv
from datetime import datetime
import os
import time
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class UltraPreciseDataRecorderNode(Node):
    def __init__(self):
        super().__init__('ultra_precise_data_recorder')
        
        # Create base directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.base_dir = f'recorded_data_{timestamp}'
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Create CSV files with high precision
        self.imu_file = self.create_csv_file('imu')
        self.odom_noise_file = self.create_csv_file('odom_noise')
        self.odom_gt_file = self.create_csv_file('odom_gt')
        self.odom_filtered_file = self.create_csv_file('odom_filtered')  # Added filtered odometry
        
        # Setup subscribers
        self.setup_subscribers()
        
        self.get_logger().info('Ultra Precise Data Recorder started!')
    
    def create_csv_file(self, topic_name):
        """Create CSV file with appropriate header"""
        csv_file = os.path.join(self.base_dir, f'{topic_name}_data.csv')
        file_handle = open(csv_file, 'w', newline='')
        writer = csv.writer(file_handle)
        
        if topic_name == 'imu':
            header = [
                'msg_timestamp',           # Float64: seconds with nanosecond precision
                'receive_timestamp',       # Float64: ROS time when received
                'system_timestamp_ns',     # Int64: system nanosecond timestamp
                'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w',
                'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z',
                'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z'
            ]
        else:
            header = [
                'msg_timestamp',           # Float64: seconds with nanosecond precision
                'receive_timestamp',       # Float64: ROS time when received
                'system_timestamp_ns',     # Int64: system nanosecond timestamp
                'position_x', 'position_y', 'position_z',
                'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w',
                'linear_velocity_x', 'linear_velocity_y', 'linear_velocity_z',
                'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z'
            ]
        
        writer.writerow(header)
        file_handle.flush()
        
        return {
            'file': file_handle,
            'writer': writer,
            'count': 0,
            'name': topic_name
        }
    
    def setup_subscribers(self):
        """Setup subscribers with QoS profiles"""
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )
        
        self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_callback, qos_profile)
        self.odom_noise_sub = self.create_subscription(Odometry, '/odom/with_noise', self.odom_noise_callback, qos_profile)
        self.odom_gt_sub = self.create_subscription(Odometry, '/odom_gz', self.odom_gt_callback, qos_profile)
        self.odom_filtered_sub = self.create_subscription(Odometry, '/odom/filtered', self.odom_filtered_callback, qos_profile)  # Added filtered subscriber
    
    def get_timestamps(self):
        """Get all timestamps with maximum precision"""
        # Get ROS time first (less overhead)
        receive_time = self.get_clock().now()
        receive_timestamp = receive_time.nanoseconds * 1e-9
        
        # Get system time with maximum precision
        system_timestamp_ns = time.time_ns()
        
        return {
            'receive_timestamp': receive_timestamp,  # Float64 with nanosecond precision
            'system_timestamp_ns': system_timestamp_ns  # Int64 nanoseconds
        }
    
    def imu_callback(self, msg):
        """Callback for IMU data"""
        try:
            # Get message timestamp with nanosecond precision
            msg_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            
            # Get reception timestamps
            timestamps = self.get_timestamps()
            
            # Prepare row
            row = [
                msg_timestamp,                    # Message timestamp (float)
                timestamps['receive_timestamp'],  # Reception timestamp (float)
                timestamps['system_timestamp_ns'], # System timestamp (int64)
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z
            ]
            
            # Write to CSV
            self.imu_file['writer'].writerow(row)
            self.imu_file['file'].flush()
            
            # Update counter
            self.imu_file['count'] += 1
            if self.imu_file['count'] % 100 == 0:
                latency = (timestamps['system_timestamp_ns'] - 
                          (msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec)) * 1e-6
                self.get_logger().info(f'IMU: {self.imu_file["count"]} records | Latency: {latency:.3f} ms')
                
        except Exception as e:
            self.get_logger().error(f'IMU error: {e}')
    
    def odom_noise_callback(self, msg):
        """Callback for noisy odometry data"""
        try:
            # Get message timestamp
            msg_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            
            # Get reception timestamps
            timestamps = self.get_timestamps()
            
            # Prepare row
            row = [
                msg_timestamp,
                timestamps['receive_timestamp'],
                timestamps['system_timestamp_ns'],
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z,
                msg.twist.twist.angular.x,
                msg.twist.twist.angular.y,
                msg.twist.twist.angular.z
            ]
            
            # Write to CSV
            self.odom_noise_file['writer'].writerow(row)
            self.odom_noise_file['file'].flush()
            
            # Update counter
            self.odom_noise_file['count'] += 1
            if self.odom_noise_file['count'] % 100 == 0:
                latency = (timestamps['system_timestamp_ns'] - 
                          (msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec)) * 1e-6
                self.get_logger().info(f'Noisy Odom: {self.odom_noise_file["count"]} records | Latency: {latency:.3f} ms')
                
        except Exception as e:
            self.get_logger().error(f'Noisy odom error: {e}')
    
    def odom_gt_callback(self, msg):
        """Callback for ground truth odometry data"""
        try:
            # Get message timestamp
            msg_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            
            # Get reception timestamps
            timestamps = self.get_timestamps()
            
            # Prepare row
            row = [
                msg_timestamp,
                timestamps['receive_timestamp'],
                timestamps['system_timestamp_ns'],
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z,
                msg.twist.twist.angular.x,
                msg.twist.twist.angular.y,
                msg.twist.twist.angular.z
            ]
            
            # Write to CSV
            self.odom_gt_file['writer'].writerow(row)
            self.odom_gt_file['file'].flush()
            
            # Update counter
            self.odom_gt_file['count'] += 1
            if self.odom_gt_file['count'] % 100 == 0:
                latency = (timestamps['system_timestamp_ns'] - 
                          (msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec)) * 1e-6
                self.get_logger().info(f'Ground Truth: {self.odom_gt_file["count"]} records | Latency: {latency:.3f} ms')
                
        except Exception as e:
            self.get_logger().error(f'Ground truth error: {e}')
    
    def odom_filtered_callback(self, msg):
        """Callback for filtered odometry data"""
        try:
            # Get message timestamp
            msg_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            
            # Get reception timestamps
            timestamps = self.get_timestamps()
            
            # Prepare row
            row = [
                msg_timestamp,
                timestamps['receive_timestamp'],
                timestamps['system_timestamp_ns'],
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z,
                msg.twist.twist.angular.x,
                msg.twist.twist.angular.y,
                msg.twist.twist.angular.z
            ]
            
            # Write to CSV
            self.odom_filtered_file['writer'].writerow(row)
            self.odom_filtered_file['file'].flush()
            
            # Update counter
            self.odom_filtered_file['count'] += 1
            if self.odom_filtered_file['count'] % 100 == 0:
                latency = (timestamps['system_timestamp_ns'] - 
                          (msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec)) * 1e-6
                self.get_logger().info(f'Filtered Odom: {self.odom_filtered_file["count"]} records | Latency: {latency:.3f} ms')
                
        except Exception as e:
            self.get_logger().error(f'Filtered odom error: {e}')
    
    def destroy_node(self):
        """Clean up"""
        self.get_logger().info('Shutting down...')
        
        for file_info in [self.imu_file, self.odom_noise_file, self.odom_gt_file, self.odom_filtered_file]:
            if file_info:
                file_info['file'].close()
                self.get_logger().info(f"  {file_info['name']}: {file_info['count']} records")
        
        self.get_logger().info(f'Data saved to: {self.base_dir}')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = UltraPreciseDataRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted, shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()