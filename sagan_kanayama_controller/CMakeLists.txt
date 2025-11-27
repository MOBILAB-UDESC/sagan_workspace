cmake_minimum_required(VERSION 3.8)
project(sagan_kanayama_controller)

# Default to C++17
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(sagan_interfaces REQUIRED)

# Include directories
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Create the node
add_executable(sagan_kanayama_controller src/sagan_kanayama_controller.cpp)

# Add dependencies
ament_target_dependencies(sagan_kanayama_controller
  rclcpp
  rclcpp_action
  geometry_msgs
  nav_msgs
  tf2_ros
  tf2_geometry_msgs
  sagan_interfaces
)

# Install the node
install(TARGETS
  sagan_kanayama_controller
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_dependencies(
  rclcpp
  rclcpp_action
  geometry_msgs
  nav_msgs
  tf2_ros
  tf2_geometry_msgs
  sagan_interfaces
)

ament_package()