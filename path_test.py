#!/usr/bin/env python3
import rclpy
import math as ma
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
    goal_msg = FollowPath.Goal()
    path = Path()
    path.header.frame_id = "map"
    
    # Add some poses
    for i in range(100):
        pose = PoseStamped()
        pose.pose.position.x = 4 * ma.cos(float(i) * 2 * 3.1415/100)
        pose.pose.position.y = 4 * ma.sin(float(i) * 2 * 3.1415/100)
        path.poses.append(pose)
    
    goal_msg.path = path
    
    # Send goal
    future = client.send_goal_async(goal_msg)
    rclpy.spin_until_future_complete(node, future)
    
    if future.result() is not None:
        node.get_logger().info('Goal sent successfully')
    else:
        node.get_logger().error('Failed to send goal')
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()