#include "sagan_kanayama_controller.hpp"
#include "nav2_util/node_utils.hpp"
#include "nav2_util/geometry_utils.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace sagan_controllers
{

void KanayamaController::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  std::string name, std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> /*costmap_ros*/)
{
  node_ = parent;
  auto node = node_.lock();
  tf_ = tf;

  nav2_util::declare_parameter_if_not_declared(node, name + ".k_x", rclcpp::ParameterValue(10.0));
  nav2_util::declare_parameter_if_not_declared(node, name + ".k_y", rclcpp::ParameterValue(25.0));
  nav2_util::declare_parameter_if_not_declared(node, name + ".k_theta", rclcpp::ParameterValue(10.0));
  nav2_util::declare_parameter_if_not_declared(node, name + ".desired_v", rclcpp::ParameterValue(0.3));
  nav2_util::declare_parameter_if_not_declared(node, name + ".max_v", rclcpp::ParameterValue(1.0));
  nav2_util::declare_parameter_if_not_declared(node, name + ".max_omega", rclcpp::ParameterValue(0.5));

  node->get_parameter(name + ".k_x", k_x_);
  node->get_parameter(name + ".k_y", k_y_);
  node->get_parameter(name + ".k_theta", k_theta_);
  node->get_parameter(name + ".desired_v", desired_v_);
  node->get_parameter(name + ".max_v", max_v_);
  node->get_parameter(name + ".max_omega", max_omega_);
}

void KanayamaController::setPlan(const nav_msgs::msg::Path & path)
{
  global_plan_ = path;
}

geometry_msgs::msg::TwistStamped KanayamaController::computeVelocityCommands(
  const geometry_msgs::msg::PoseStamped & pose,
  const geometry_msgs::msg::Twist & /*velocity*/,
  nav2_core::GoalChecker * /*goal_checker*/)
{
  auto node = node_.lock();
  geometry_msgs::msg::TwistStamped cmd_vel;
  cmd_vel.header.frame_id = pose.header.frame_id;
  cmd_vel.header.stamp = node->now();

  if (global_plan_.poses.empty()) return cmd_vel;

  // 1. Target Point (Simplification: tracking the nearest point on the path)
  // In a production setup, you'd use a lookahead distance.
  const auto& ref_pose = global_plan_.poses.front().pose;

  // 2. State Extraction
  double x_c = pose.pose.position.x;
  double y_c = pose.pose.position.y;
  double theta_c = tf2::getYaw(pose.pose.orientation);

  double x_r = ref_pose.position.x;
  double y_r = ref_pose.position.y;
  double theta_r = tf2::getYaw(ref_pose.orientation);

  // 3. Error in Robot Frame
  double dx = x_r - x_c;
  double dy = y_r - y_c;
  
  double e_x = std::cos(theta_c) * dx + std::sin(theta_c) * dy;
  double e_y = -std::sin(theta_c) * dx + std::cos(theta_c) * dy;
  
  double e_theta = theta_r - theta_c;
  while (e_theta > M_PI) e_theta -= 2.0 * M_PI;
  while (e_theta < -M_PI) e_theta += 2.0 * M_PI;

  // 4. Reference velocities (assuming constant for the path segment)
  double v_r = desired_v_;
  double omega_r = 0.0; // Can be derived from path curvature

  // 5. Kanayama Law
  double v = v_r * std::cos(e_theta) + k_x_ * e_x;
  double omega = omega_r + v_r * (k_y_ * e_y + k_theta_ * std::sin(e_theta));

  cmd_vel.twist.linear.x = std::clamp(v, -max_v_, max_v_);
  cmd_vel.twist.angular.z = std::clamp(omega, -max_omega_, max_omega_);

  return cmd_vel;
}

void KanayamaController::activate() {}
void KanayamaController::deactivate() {}
void KanayamaController::cleanup() {}
void KanayamaController::setSpeedLimit(const double & speed_limit, const bool & percentage) 
{
  if (percentage) max_v_ *= speed_limit / 100.0;
  else max_v_ = speed_limit;
}

} // namespace sagan_controllers

PLUGINLIB_EXPORT_CLASS(sagan_controllers::KanayamaController, nav2_core::Controller)