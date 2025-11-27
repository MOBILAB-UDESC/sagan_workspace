cmake_minimum_required(VERSION 3.8)
project(sagan_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(action_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/SaganCmd.msg"
  "msg/SteeringCmd.msg"
  "msg/WheelCmd.msg"
  "msg/SaganStates.msg"
  "msg/SteeringState.msg"
  "msg/WheelState.msg"
  "action/FollowPath.action"
  DEPENDENCIES 
  std_msgs
  geometry_msgs 
  nav_msgs 
  action_msgs 
)

ament_export_dependencies(
  rosidl_default_runtime
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
  action_msgs
)

  ament_package()
