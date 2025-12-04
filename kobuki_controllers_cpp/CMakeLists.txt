cmake_minimum_required(VERSION 3.8)
project(kobuki_controllers_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(casadi REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(sagan_interfaces REQUIRED)


# # add include directories to the build.
# include_directories(
#   include/${PROJECT_NAME}
# )

# # create variable SRC_FILES to hold the source files
# set(SRC_FILES
#   src/mpc.cpp
#   src/trajectory.cpp
#   src/main.cpp
# )

# # add an executable target to be built
# add_executable(main 
#   ${SRC_FILES}
# )

# # add dependecies to the target
# ament_target_dependencies(main
#   rclcpp
#   nav_msgs
#   geometry_msgs
#   tf2_geometry_msgs
#   sagan_interfaces
# )
# target_link_libraries(main
#   casadi::casadi
# )

# # install the executable target
# install(
#   TARGETS main
#   DESTINATION lib/${PROJECT_NAME}
# )

# if(BUILD_TESTING)
#   find_package(ament_lint_auto REQUIRED)
#   set(ament_cmake_copyright_FOUND TRUE)
#   set(ament_cmake_cpplint_FOUND TRUE)
#   ament_lint_auto_find_test_dependencies()
# endif()

add_library(main SHARED
  src/main.cpp
  src/mpc.cpp
  src/trajectory.cpp)

target_include_directories(main PRIVATE
  include/${PROJECT_NAME}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_compile_definitions(main
  PRIVATE "CUSTOM_ACTION_CPP_BUILDING_DLL")

ament_target_dependencies(main
  "rclcpp"
  "rclcpp_action"
  "rclcpp_components"
  "sagan_interfaces"
  "nav_msgs"
  "geometry_msgs"
  "tf2_geometry_msgs"
  "sagan_interfaces")

  target_link_libraries(main
  casadi::casadi
  )

rclcpp_components_register_node(main PLUGIN "custom_action_cpp::ControlNode" EXECUTABLE path_controller)
install(TARGETS
  main
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

ament_package()
