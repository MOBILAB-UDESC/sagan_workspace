#include "sagan_kanayama_controller.hpp"
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

  // Declare Parameters
  nav2_util::declare_parameter_if_not_declared(node, name + ".k_x", rclcpp::ParameterValue(1.5));
  nav2_util::declare_parameter_if_not_declared(node, name + ".k_y", rclcpp::ParameterValue(5.0));
  nav2_util::declare_parameter_if_not_declared(node, name + ".k_theta", rclcpp::ParameterValue(2.0));
  nav2_util::declare_parameter_if_not_declared(node, name + ".desired_v", rclcpp::ParameterValue(0.3));
  nav2_util::declare_parameter_if_not_declared(node, name + ".max_v", rclcpp::ParameterValue(0.8));
  nav2_util::declare_parameter_if_not_declared(node, name + ".max_omega", rclcpp::ParameterValue(1.0));
  nav2_util::declare_parameter_if_not_declared(node, name + ".lookahead_dist", rclcpp::ParameterValue(0.4));

  node->get_parameter(name + ".k_x", k_x_);
  node->get_parameter(name + ".k_y", k_y_);
  node->get_parameter(name + ".k_theta", k_theta_);
  node->get_parameter(name + ".desired_v", desired_v_);
  node->get_parameter(name + ".max_v", max_v_);
  node->get_parameter(name + ".max_omega", max_omega_);
  node->get_parameter(name + ".lookahead_dist", lookahead_dist_);

  // Frequency check
  double controller_frequency;
  node->get_parameter("controller_frequency", controller_frequency);
  control_period_ = 1.0 / controller_frequency;
}

void KanayamaController::setPlan(const nav_msgs::msg::Path & path)
{
  global_plan_ = path;
}

double KanayamaController::normalize_angle(double angle)
{
  while (angle > M_PI) angle -= 2.0 * M_PI;
  while (angle < -M_PI) angle += 2.0 * M_PI;
  return angle;
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

  // 1. Find Lookahead Point (The Carrot)
  size_t target_idx = 0;
  for (size_t i = 0; i < global_plan_.poses.size(); ++i) {
    target_idx = i;
    double d = nav2_util::geometry_utils::euclidean_distance(pose.pose, global_plan_.poses[i].pose);
    if (d > lookahead_dist_) break;
  }
  const auto & ref_pose = global_plan_.poses[target_idx].pose;

  // 2. Extract State
  double x_c = pose.pose.position.x;
  double y_c = pose.pose.position.y;
  double theta_c = tf2::getYaw(pose.pose.orientation);

  double x_r = ref_pose.position.x;
  double y_r = ref_pose.position.y;
  double theta_r = tf2::getYaw(ref_pose.orientation);

  // 3. Transform Error to Robot Frame
  double dx = x_r - x_c;
  double dy = y_r - y_c;
  double e_x = std::cos(theta_c) * dx + std::sin(theta_c) * dy;
  double e_y = -std::sin(theta_c) * dx + std::cos(theta_c) * dy;
  double e_theta = normalize_angle(theta_r - theta_c);

  // 4. Extract Reference Velocities (Reconstructing Trajectory)
  double v_r = desired_v_;
  double omega_r = 0.0;
  if (target_idx + 1 < global_plan_.poses.size()) {
    double next_theta = tf2::getYaw(global_plan_.poses[target_idx + 1].pose.orientation);
    omega_r = normalize_angle(next_theta - theta_r) / control_period_;
  }

  // 5. Kanayama Law
  double v = v_r * std::cos(e_theta) + k_x_ * e_x;
  double omega = omega_r + v_r * (k_y_ * e_y + k_theta_ * std::sin(e_theta));

  // 6. Output and Clamping
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