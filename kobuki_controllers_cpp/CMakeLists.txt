cmake_minimum_required(VERSION 3.8)
project(kobuki_controllers_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(casadi REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# add include directories to the build.
include_directories(
  include/${PROJECT_NAME}
)

# create variable SRC_FILES to hold the source files
set(SRC_FILES
  src/mpc.cpp
  src/trajectory.cpp
  src/main.cpp
)

# add an executable target to be built
add_executable(main 
  ${SRC_FILES}
)

# add dependecies to the target
ament_target_dependencies(main
  rclcpp
  nav_msgs
  geometry_msgs
  tf2_geometry_msgs
)
target_link_libraries(main
  casadi::casadi
)

# install the executable target
install(
  TARGETS main
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
