cmake_minimum_required(VERSION 3.8)
project(sagan_control_hardware_interface)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(controller_interface REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(sagan_interfaces REQUIRED)
find_package(std_msgs REQUIRED)

# --- Find I2C library ---
find_library(I2C_LIBRARY NAMES i2c)
find_path(I2C_INCLUDE_DIRS NAMES linux/i2c-dev.h)

if(NOT I2C_LIBRARY)
    message(FATAL_ERROR "I2C library not found")
endif()

if(NOT I2C_INCLUDE_DIRS)
    message(FATAL_ERROR "I2C headers not found")
endif()

message(STATUS "Found I2C library: ${I2C_LIBRARY}")
message(STATUS "Found I2C headers: ${I2C_INCLUDE_DIRS}")

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_library(sagan_control_hardware_interface SHARED
  src/sagan_control_hardware_interface.cpp
)

target_include_directories(sagan_control_hardware_interface
  PUBLIC
    include
    ${I2C_INCLUDE_DIRS}
)

target_link_libraries(sagan_control_hardware_interface PUBLIC
  ${I2C_LIBRARY}
)

ament_target_dependencies(sagan_control_hardware_interface PUBLIC
  controller_interface
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
  sagan_interfaces
  std_msgs
)

pluginlib_export_plugin_description_file(
  hardware_interface sagan_control_hardware_interface.xml
)

# Add install directives
install(TARGETS sagan_control_hardware_interface
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

install(FILES "sagan_control_hardware_interface.xml"
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(
    include
)

ament_export_dependencies(
    controller_interface
    hardware_interface
    pluginlib
    rclcpp
    rclcpp_lifecycle
)

ament_export_libraries(sagan_control_hardware_interface)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()