#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <cmath>
#include <vector>

class SaganPathPublisher : public rclcpp::Node
{
public:
    SaganPathPublisher() : Node("sagan_path_publisher")
    {
        // Declare parameters
        this->declare_parameter("center_x", 0.0);
        this->declare_parameter("center_y", 0.0);
        this->declare_parameter("radius", 4.0);
        this->declare_parameter("num_points", 500);
        this->declare_parameter("frame_id", "map");
        this->declare_parameter("publish_rate", 1.0);

        // Get parameters
        center_x_ = this->get_parameter("center_x").as_double();
        center_y_ = this->get_parameter("center_y").as_double();
        radius_ = this->get_parameter("radius").as_double();
        num_points_ = this->get_parameter("num_points").as_int();
        frame_id_ = this->get_parameter("frame_id").as_string();
        double publish_rate = this->get_parameter("publish_rate").as_double();

        // Create publishers
        path_publisher_ = this->create_publisher<nav_msgs::msg::Path>("/path", 10);
        marker_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>("/marker", 10);

        // Create timer
        auto timer_period = std::chrono::duration<double>(1.0 / publish_rate);
        timer_ = this->create_wall_timer(
            std::chrono::duration_cast<std::chrono::nanoseconds>(timer_period),
            std::bind(&SaganPathPublisher::timer_callback, this));

        // Generate circle points
        generate_circle_points();

        RCLCPP_INFO(this->get_logger(), "Circle path publisher started");
        RCLCPP_INFO(this->get_logger(), "Center: (%.2f, %.2f), Radius: %.2f, Points: %d", 
                   center_x_, center_y_, radius_, num_points_);
    }

private:
    void generate_circle_points()
    {
        circle_points_.clear();
        
        for (int i = 0; i < num_points_; ++i) {
            double angle = 2.0 * M_PI * i / num_points_;
            double x = center_x_ + radius_ * std::cos(angle);
            double y = center_y_ + radius_ * std::sin(angle);
            
            geometry_msgs::msg::Point point;
            point.x = x;
            point.y = y;
            point.z = 0.0;
            
            circle_points_.push_back(point);
        }
        
        // Add first point at the end to close the circle
        circle_points_.push_back(circle_points_[0]);
    }

    void timer_callback()
    {
        publish_path();
        publish_marker();
    }

    void publish_path()
    {
        auto path_msg = std::make_unique<nav_msgs::msg::Path>();
        
        // Set header
        path_msg->header.stamp = this->now();
        path_msg->header.frame_id = frame_id_;
        
        // Create poses for each point
        for (const auto& point : circle_points_) {
            geometry_msgs::msg::PoseStamped pose;
            pose.header = path_msg->header;
            pose.pose.position = point;
            
            // Calculate orientation tangent to the circle
            double dx = -point.y + center_y_;  // derivative of circle equation
            double dy = point.x - center_x_;
            double yaw = std::atan2(dy, dx);
            
            tf2::Quaternion quat;
            quat.setRPY(0.0, 0.0, yaw);
            pose.pose.orientation = tf2::toMsg(quat);
            
            path_msg->poses.push_back(pose);
        }
        
        path_publisher_->publish(std::move(path_msg));
    }

    void publish_marker()
    {
        auto marker_msg = std::make_unique<visualization_msgs::msg::Marker>();
        
        marker_msg->header.stamp = this->now();
        marker_msg->header.frame_id = frame_id_;
        marker_msg->ns = "circle_path";
        marker_msg->id = 0;
        marker_msg->type = visualization_msgs::msg::Marker::LINE_STRIP;
        marker_msg->action = visualization_msgs::msg::Marker::ADD;
        
        // Set marker scale
        marker_msg->scale.x = 0.05;  // Line width
        
        // Set marker color (green)
        marker_msg->color.r = 0.0;
        marker_msg->color.g = 1.0;
        marker_msg->color.b = 0.0;
        marker_msg->color.a = 1.0;
        
        // Set marker points
        marker_msg->points = circle_points_;
        
        marker_publisher_->publish(std::move(marker_msg));
    }

    // Parameters
    double center_x_;
    double center_y_;
    double radius_;
    int num_points_;
    std::string frame_id_;
    
    // Circle points storage
    std::vector<geometry_msgs::msg::Point> circle_points_;
    
    // ROS2 components
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SaganPathPublisher>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}