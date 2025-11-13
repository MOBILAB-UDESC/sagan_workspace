import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
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
    
    robotXacroName="Sagan"
    namePackage="sagan_description"
    modelFileRelativePath="model/Sagan.xacro.urdf"
    
    pathModelFile = os.path.join(get_package_share_directory(namePackage), modelFileRelativePath)
    
    robotDescription= xacro.process_file(pathModelFile, mappings={'use_sim_time': use_sim_time}).toxml()

    controllers_yaml_path = os.path.join(
        get_package_share_directory("sagan_description"), 
        "parameters/sagan_controllers_parameters.yaml"
    )

    ekf_params_path = os.path.join(
        get_package_share_directory('sagan_description'), 
        'parameters', 'ekf_params.yaml'
    )

    bridge_params_path = os.path.join(
        get_package_share_directory(namePackage),
        "parameters",
        "bridge_parameters.yaml"
    )

    # --- Simulation-Only Nodes ---

    gazebo_rosPackageLaunch=PythonLaunchDescriptionSource(
        os.path.join(get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")
    )

    gazeboLaunch = IncludeLaunchDescription(
        gazebo_rosPackageLaunch, 
        launch_arguments={
            "gz_args": [" -r -v -v4 " + os.path.join(get_package_share_directory("sagan_description"), "worlds/empty_world.sdf")],
            "on_exit_shutdown": "true"
        }.items(),
        condition=IfCondition(use_sim_time)
    )
    
    spawnModelNodeGazebo = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name", robotXacroName,
            "-topic", "robot_description", 
            "-z", "1"
        ],
        output="screen",    
        condition=IfCondition(use_sim_time)  
    )
    
    start_gazebo_ros_bridge_cmd = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/world/default/set_pose@ros_gz_interfaces/srv/SetEntityPose",
            "--ros-args",
            "-p",
            f"config_file:={bridge_params_path}",
        ],
        output="screen",
        condition=IfCondition(use_sim_time) 
    )       

    # --- Real-Robot-Only Node ---

    # This is the new node. It launches the controller_manager, which will
    # load your "sagan_control_hardware_interface" plugin from the URDF.
    control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[
            {'robot_description': robotDescription},
            controllers_yaml_path
        ],
        output='screen',
        condition=UnlessCondition(use_sim_time) # Only run this if *not* in simulation
    )

    nodeSaganImuDriver = Node(
        package='sagan_mpu9250_driver',
        executable='sagan_mpu9250_driver',
        output='screen',
        parameters=[{"use_sim_time": use_sim_time}],
        condition=UnlessCondition(use_sim_time) # Only run this if *not* in simulation
    )
    # --- Common Nodes (Sim & Real) ---
    
    nodeRobotStatePublisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robotDescription, "use_sim_time": use_sim_time}]
    )

    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        parameters=[{"use_sim_time": use_sim_time}]
    )

    diff_drive_base_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['sagan_drive_controller', '--controller-manager', '/controller_manager'],
        parameters=[{"use_sim_time": use_sim_time}]
    )

    nodeSaganOdometry = Node(
        package="sagan_odometry",
        executable="sagan_odometry",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}]
    )

    nodeSaganEKF = Node(
        package='sagan_kalman_filter',
        executable='sagan_kalman_filter',
        output='screen',
        parameters=[{"use_sim_time": use_sim_time}, ekf_params_path],
    )

    nodeSaganDiffDriver = Node(
        package='sagan_differential_driver',
        executable='sagan_differential_driver',
        output='screen',
        parameters=[{"use_sim_time": use_sim_time}],
    )

    nodeSaganPath = Node(
        package='sagan_kanayama_controller',
        executable='sagan_kanayama_controller',
        name='sagan_kanayama_controller',
        output='screen',
        parameters=[{
            'k_x': 10.0,
            'k_y': 25.0,
            'k_theta': 10.0,
            'max_linear_velocity': 1.0,
            'max_angular_velocity': 1.0,
            'lookahead_distance': 0.5,
            'robot_base_frame': 'base_footprint',
            'odom_frame': 'odom',
            'use_sim_time': use_sim_time
        }]
    )


    # --- Launch Description ---

    launchDescriptionObject = LaunchDescription()
    
    # 1. Add the use_sim_time argument
    launchDescriptionObject.add_action(use_sim_time_arg)
    
    # 2. Add Simulation-Only nodes
    # launchDescriptionObject.add_action(gazeboLaunch)
    # launchDescriptionObject.add_action(spawnModelNodeGazebo)
    # launchDescriptionObject.add_action(start_gazebo_ros_bridge_cmd)

    # 3. Add Real-Robot-Only node
    launchDescriptionObject.add_action(control_node)
    launchDescriptionObject.add_action(nodeSaganImuDriver)  

    # 4. Add Common nodes
    launchDescriptionObject.add_action(nodeRobotStatePublisher)
    launchDescriptionObject.add_action(joint_state_broadcaster_spawner)
    #launchDescriptionObject.add_action(diff_drive_base_controller_spawner)
    launchDescriptionObject.add_action(nodeSaganOdometry)
    launchDescriptionObject.add_action(nodeSaganEKF)
    launchDescriptionObject.add_action(nodeSaganDiffDriver)
    launchDescriptionObject.add_action(nodeSaganPath)

    return launchDescriptionObject
