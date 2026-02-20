#!/usr/bin/env python3
import rclpy
import math
from rclpy.action import ActionClient
from sagan_interfaces.action import FollowPath
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point

def main():
    rclpy.init()
    node = rclpy.create_node('test_action_client')
    
    client = ActionClient(node, FollowPath, 'follow_path')
    
    # Wait for server
    if not client.wait_for_server(timeout_sec=5.0):
        node.get_logger().error('Action server not available')
        return
    
    # Create a simple path
    # goal_msg = FollowPath.Goal()
    # path = Path()
    # path.header.frame_id = "map"
    
    # # Add some poses
    # for i in range(1000):
    #     pose = PoseStamped()
    #     pose.pose.position.x = 4 * ma.cos(float(i) * 2 * 3.1415/1000)
    #     pose.pose.position.y = 4 * ma.sin(float(i) * 2 * 3.1415/1000)
    #     path.poses.append(pose)
    
    # goal_msg.path = path

    
    path_msg = path_creator(node)
    goal_msg = FollowPath.Goal()
    goal_msg.path = path_msg
    
    # Send goal
    future = client.send_goal_async(goal_msg)
    rclpy.spin_until_future_complete(node, future)
    
    if future.result() is not None:
        node.get_logger().info('Goal sent successfully')
    else:
        node.get_logger().error('Failed to send goal')
    
    node.destroy_node()
    rclpy.shutdown()

def path_creator(self):
    path_msg = Path()
    path_msg.header.stamp = self.get_clock().now().to_msg()
    path_msg.header.frame_id = "odom"
    
    # Lemniscate parameters
    a = 5.0  # Size parameter
    num_points = 2000  # Number of waypoints
    
    # Start at t = π/2 to ensure we begin at origin
    # For Bernoulli's lemniscate: x = a*cos(t)/(1+sin²(t)), y = a*sin(t)*cos(t)/(1+sin²(t))
    # At t = π/2: cos(π/2) = 0, sin(π/2) = 1 → x = 0, y = 0
    t_start = math.pi / 2  # Start at π/2 to begin at origin
    t_end = math.pi / 2 + 2.0 * math.pi  # One full cycle (2π) from starting point
    
    # Rotation angle in radians (135°)
    rotation_angle = 135.0 * math.pi / 180.0
    cos_theta = math.cos(rotation_angle)
    sin_theta = math.sin(rotation_angle)
    
    # Generate lemniscate waypoints
    for i in range(num_points + 1):
        # Parameter t from t_start to t_end
        t = t_start + (t_end - t_start) * i / num_points
        
        # Parametric equations for lemniscate (Bernoulli's lemniscate)
        denominator = 1.0 + math.sin(t)**2
        x_original = a * math.cos(t) / denominator
        y_original = a * math.sin(t) * math.cos(t) / denominator
        
        # Apply 135° rotation
        x = x_original * cos_theta - y_original * sin_theta
        y = x_original * sin_theta + y_original * cos_theta
        
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.header.frame_id = "odom"
        pose_stamped.pose.position.x = x
        pose_stamped.pose.position.y = y
        pose_stamped.pose.position.z = 0.0
        
        # Calculate orientation (tangent to the path)
        # Use next point for calculating direction
        if i < num_points:
            next_t = t_start + (t_end - t_start) * (i + 1) / num_points
        else:
            # For the last point, use first point (closed loop)
            next_t = t_start
        
        # Calculate next point
        next_denominator = 1.0 + math.sin(next_t)**2
        next_x_original = a * math.cos(next_t) / next_denominator
        next_y_original = a * math.sin(next_t) * math.cos(next_t) / next_denominator
        
        # Apply same rotation to next point
        next_x = next_x_original * cos_theta - next_y_original * sin_theta
        next_y = next_x_original * sin_theta + next_y_original * cos_theta
        
        # Calculate yaw angle (tangent to the rotated curve)
        yaw = math.atan2(next_y - y, next_x - x)
        
        # Convert yaw to quaternion
        pose_stamped.pose.orientation.x = 0.0
        pose_stamped.pose.orientation.y = 0.0
        pose_stamped.pose.orientation.z = math.sin(yaw / 2.0)
        pose_stamped.pose.orientation.w = math.cos(yaw / 2.0)
        
        path_msg.poses.append(pose_stamped)

    # Force the first point to be exactly at origin
    if len(path_msg.poses) > 0:
        path_msg.poses[0].pose.position.x = 0.0
        path_msg.poses[0].pose.position.y = 0.0

    self.get_logger().info(f"Created 135° rotated lemniscate path with {len(path_msg.poses)} waypoints")
    return path_msg

if __name__ == '__main__':
    main()