#ifndef SAGAN_CONTROLLERS_KANAYAMA_CONTROLLER_HPP_
#define SAGAN_CONTROLLERS_KANAYAMA_CONTROLLER_HPP_

#include <memory>
#include <string>
#include <vector>

#include "nav2_core/controller.hpp"
#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"

namespace sagan_controllers
{
class KanayamaController : public nav2_core::Controller
{
public:
  KanayamaController() = default;
  ~KanayamaController() override = default;

  void configure(const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name, std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

  void activate() override;
  void deactivate() override;
  void cleanup() override;

  void setPlan(const nav_msgs::msg::Path & path) override;

  geometry_msgs::msg::TwistStamped computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped & pose,
    const geometry_msgs::msg::Twist & velocity,
    nav2_core::GoalChecker * goal_checker) override;

  void setSpeedLimit(const double & speed_limit, const bool & percentage) override;

protected:
  rclcpp_lifecycle::LifecycleNode::WeakPtr node_;
  std::shared_ptr<tf2_ros::Buffer> tf_;
  nav_msgs::msg::Path global_plan_;
  
  // Gains and limits
  double k_x_, k_y_, k_theta_;
  double max_v_, max_omega_, desired_v_;
};
} 

#endif