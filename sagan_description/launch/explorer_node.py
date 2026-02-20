import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.parameter_descriptions import ParameterValue
import xacro

def generate_launch_description():

    # 1. Declare the launch argument
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )
    
    # 2. Create a LaunchConfiguration to hold the value of the argument
    use_sim_time = LaunchConfiguration('use_sim_time')

    m_explore_launch_path = os.path.join(get_package_share_directory('explore_lite'), 'launch', 'explore.launch.py')
    m_explore_params_path = os.path.join(get_package_share_directory('sagan_description'), 'parameters', 'explore.yaml')

    nodeExplore = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(m_explore_launch_path),
        launch_arguments={
            'params_file': m_explore_params_path,
            'use_sim_time': use_sim_time
        }.items()
    )
    
    launchDescriptionObject = LaunchDescription()
    launchDescriptionObject.add_action(use_sim_time_arg)
    launchDescriptionObject.add_action(nodeExplore)

    return launchDescriptionObject