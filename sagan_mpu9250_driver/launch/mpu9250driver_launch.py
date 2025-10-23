from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

import os


def generate_launch_description():
    ld = LaunchDescription()
    share_dir = get_package_share_directory('sagan_mpu9250_driver')
    parameter_file = LaunchConfiguration('params_file')

    params_declare = DeclareLaunchArgument('params_file',
                                           default_value=os.path.join(
                                               share_dir, 'params', 'mpu9250.yaml'),
                                           description='Path to the ROS2 parameters file to use.')

    mpu9250driver_node = Node(
        package='sagan_mpu9250_driver',
        executable='sagan_mpu9250_driver',
        name='mpu9250driver_node',
        output="screen",
        emulate_tty=True,
        parameters=[parameter_file]
    )

    ld.add_action(params_declare)
    ld.add_action(mpu9250driver_node)
    return ld
