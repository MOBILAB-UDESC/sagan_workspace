#include "sagan_interfaces/action/follow_path.hpp"

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <memory>
#include <vector>
#include <cmath>
#include <thread>
#include <chrono>
#include <limits>
#include <algorithm>


using FollowPath = sagan_interfaces::action::FollowPath;
using GoalHandleFollowPath = rclcpp_action::ServerGoalHandle<FollowPath>;

class SaganKanayamaController : public rclcpp::Node
{
public:
    SaganKanayamaController() : Node("sagan_kanayama_controller")
    {
        // Parâmetros do controlador (valores baseados na seção 6.3.2)
        this->declare_parameter("k_x", 10.0);
        this->declare_parameter("k_y", 25.0);
        this->declare_parameter("k_theta", 10.0);
        this->declare_parameter("max_linear_velocity", 1.0);
        this->declare_parameter("max_angular_velocity", 0.25);
        this->declare_parameter("lookahead_distance", 0.2);
        this->declare_parameter("robot_base_frame", "base_link");
        this->declare_parameter("odom_frame", "odom");
        this->declare_parameter("goal_tolerance", 0.1);
        this->declare_parameter("control_frequency", 100.0);

        // Obter parâmetros
        k_x_ = this->get_parameter("k_x").as_double();
        k_y_ = this->get_parameter("k_y").as_double();
        k_theta_ = this->get_parameter("k_theta").as_double();
        max_linear_velocity_ = this->get_parameter("max_linear_velocity").as_double();
        max_angular_velocity_ = this->get_parameter("max_angular_velocity").as_double();
        lookahead_distance_ = this->get_parameter("lookahead_distance").as_double();
        robot_base_frame_ = this->get_parameter("robot_base_frame").as_string();
        odom_frame_ = this->get_parameter("odom_frame").as_string();
        goal_tolerance_ = this->get_parameter("goal_tolerance").as_double();
        double control_frequency = this->get_parameter("control_frequency").as_double();

        // Inicializar TF
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // Publicadores
        cmd_vel_publisher_ = this->create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 10);

        // Subscritores
        odom_subscriber_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "odom/filtered", 10, std::bind(&SaganKanayamaController::odomCallback, this, std::placeholders::_1));

        // Action Server
        action_server_ = rclcpp_action::create_server<FollowPath>(
            this,
            "follow_path",
            std::bind(&SaganKanayamaController::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
            std::bind(&SaganKanayamaController::handle_cancel, this, std::placeholders::_1),
            std::bind(&SaganKanayamaController::handle_accepted, this, std::placeholders::_1));

        // Timer para controle
        control_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(1000.0 / control_frequency)),
            std::bind(&SaganKanayamaController::controlLoop, this));

        RCLCPP_INFO(this->get_logger(), "Controlador Kanayama com Action Server inicializado");
        RCLCPP_INFO(this->get_logger(), "Ganhos: Kx=%.2f, Ky=%.2f, Kθ=%.2f", k_x_, k_y_, k_theta_);
    }

private:
    // Parâmetros do controlador
    double k_x_, k_y_, k_theta_;
    double max_linear_velocity_, max_angular_velocity_;
    double lookahead_distance_, goal_tolerance_;
    std::string robot_base_frame_, odom_frame_;

    // Estado atual do robô
    geometry_msgs::msg::Pose current_pose_;
    bool pose_received_ = false;
    
    // Trajetória de referência e estado do action
    nav_msgs::msg::Path current_path_;
    bool path_active_ = false;
    size_t current_waypoint_index_ = 0;
    std::shared_ptr<GoalHandleFollowPath> current_goal_handle_;

    // TF
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // Publicadores, Subscritores e Action Server
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_publisher_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscriber_;
    rclcpp::TimerBase::SharedPtr control_timer_;
    rclcpp_action::Server<FollowPath>::SharedPtr action_server_;

    // Callbacks do Action Server
    rclcpp_action::GoalResponse handle_goal(
        const rclcpp_action::GoalUUID & uuid,
        std::shared_ptr<const FollowPath::Goal> goal)
    {
        RCLCPP_INFO(this->get_logger(), "Recebido goal para seguir path com %zu pontos", goal->path.poses.size());
        (void)uuid;
        
        if (goal->path.poses.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Path vazio rejeitado");
            return rclcpp_action::GoalResponse::REJECT;
        }
        
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    }

    rclcpp_action::CancelResponse handle_cancel(
        const std::shared_ptr<GoalHandleFollowPath> goal_handle)
    {
        RCLCPP_INFO(this->get_logger(), "Goal cancelado");
        
        // Check if this is the current active goal
        if (goal_handle != current_goal_handle_) {
            RCLCPP_WARN(this->get_logger(), "Cancel request for non-active goal");
            return rclcpp_action::CancelResponse::REJECT;
        }

        // Stop the robot immediately
        geometry_msgs::msg::Twist stop_msg;
        cmd_vel_publisher_->publish(stop_msg);
        
        // Mark path as inactive
        path_active_ = false;
        
        RCLCPP_INFO(this->get_logger(), "Goal cancellation accepted");
        return rclcpp_action::CancelResponse::ACCEPT;
    }

    void handle_accepted(const std::shared_ptr<GoalHandleFollowPath> goal_handle)
    {
        // If already executing another goal, cancel the previous one
        if (path_active_ && current_goal_handle_) {
            RCLCPP_INFO(this->get_logger(), "Preempting previous goal");
            auto result = std::make_shared<FollowPath::Result>();
            result->success = false;
            result->message = "Preempted by new goal";
            current_goal_handle_->abort(result);
        }

        current_goal_handle_ = goal_handle;
        current_path_ = goal_handle->get_goal()->path;
        current_waypoint_index_ = 0;
        path_active_ = true;

        RCLCPP_INFO(this->get_logger(), "Execução do path iniciada com %zu waypoints", current_path_.poses.size());
    }

    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        current_pose_ = msg->pose.pose;
        pose_received_ = true;
    }

    geometry_msgs::msg::Pose getRobotPose()
    {
        // Tentar obter a pose do robô usando TF como fallback
        if (!pose_received_) {
            try {
                geometry_msgs::msg::TransformStamped transform;
                transform = tf_buffer_->lookupTransform(
                    odom_frame_, robot_base_frame_,
                    tf2::TimePointZero);
                
                current_pose_.position.x = transform.transform.translation.x;
                current_pose_.position.y = transform.transform.translation.y;
                current_pose_.position.z = transform.transform.translation.z;
                current_pose_.orientation = transform.transform.rotation;
                pose_received_ = true;
            }
            catch (tf2::TransformException &ex) {
                RCLCPP_WARN(this->get_logger(), "Não foi possível obter transformação: %s", ex.what());
            }
        }
        return current_pose_;
    }

    size_t findTargetWaypoint(const geometry_msgs::msg::Pose& robot_pose)
    {
        if (current_path_.poses.empty()) return 0;

        // Encontrar o waypoint mais próximo
        size_t closest_index = 0;
        double min_distance = std::numeric_limits<double>::max();

        for (size_t i = current_waypoint_index_; i < current_path_.poses.size(); ++i) {
            double dx = current_path_.poses[i].pose.position.x - robot_pose.position.x;
            double dy = current_path_.poses[i].pose.position.y - robot_pose.position.y;
            double distance = std::sqrt(dx*dx + dy*dy);

            if (distance < min_distance) {
                min_distance = distance;
                closest_index = i;
            }
        }

        // Encontrar waypoint à frente baseado na distância de lookahead
        for (size_t i = closest_index; i < current_path_.poses.size(); ++i) {
            double dx = current_path_.poses[i].pose.position.x - robot_pose.position.x;
            double dy = current_path_.poses[i].pose.position.y - robot_pose.position.y;
            double distance = std::sqrt(dx*dx + dy*dy);

            if (distance >= lookahead_distance_) {
                return i;
            }
        }

        return current_path_.poses.size() - 1;
    }

    void computeControl(const geometry_msgs::msg::Pose& robot_pose,
                       const geometry_msgs::msg::Pose& target_pose,
                       double& v, double& omega)
    {
        // Extrair orientação do robô
        double theta = 2.0 * std::atan2(robot_pose.orientation.z, robot_pose.orientation.w);

        // Calcular erro de posição no frame global
        double x_error = target_pose.position.x - robot_pose.position.x;
        double y_error = target_pose.position.y - robot_pose.position.y;

        // Transformar erro para frame local do robô (Equação 6.42)
        double x_e = std::cos(theta) * x_error + std::sin(theta) * y_error;
        double y_e = -std::sin(theta) * x_error + std::cos(theta) * y_error;

        // Calcular erro de orientação
        double target_theta = 2.0 * std::atan2(target_pose.orientation.z, target_pose.orientation.w);
        double theta_e = target_theta - theta;

        // Normalizar ângulo para [-pi, pi]
        while (theta_e > M_PI) theta_e -= 2.0 * M_PI;
        while (theta_e < -M_PI) theta_e += 2.0 * M_PI;

        // Velocidades de referência (simplificado - assumindo velocidade constante)
        double v_r = max_linear_velocity_;
        double omega_r = 0.0; // Para trajetória reta

        // Lei de controle de Kanayama (Equação 6.45)
        v = v_r * std::cos(theta_e) + k_x_ * x_e;
        omega = omega_r + v_r * (k_y_ * y_e + k_theta_ * std::sin(theta_e));

        // Saturação das velocidades
        v = std::clamp(v, -max_linear_velocity_, max_linear_velocity_);
        omega = std::clamp(omega, -max_angular_velocity_, max_angular_velocity_);

        RCLCPP_INFO(this->get_logger(), 
                    "Erros: x_e=%.3f, y_e=%.3f, θ_e=%.3f | Comando: v=%.3f, ω=%.3f", 
                    x_e, y_e, theta_e, v, omega);
    }

    void controlLoop()
    {
        if (!path_active_ || !pose_received_ || !current_goal_handle_) {
            return;
        }

        // Check if goal was cancelled
        if (!current_goal_handle_->is_canceling()) {
            // Normal execution path
            
            // Check if path is completed
            if (current_waypoint_index_ >= current_path_.poses.size()) {
                auto result = std::make_shared<FollowPath::Result>();
                result->success = true;
                result->message = "Path completed successfully";
                current_goal_handle_->succeed(result);
                RCLCPP_INFO(this->get_logger(), "Path completado com sucesso");
                
                // Stop the robot
                geometry_msgs::msg::Twist stop_msg;
                cmd_vel_publisher_->publish(stop_msg);
                
                path_active_ = false;
                current_goal_handle_.reset();
                return;
            }

            // Execute normal control logic
            geometry_msgs::msg::Pose robot_pose = getRobotPose();
            current_waypoint_index_ = findTargetWaypoint(robot_pose);
            geometry_msgs::msg::Pose target_pose = current_path_.poses[current_waypoint_index_].pose;

            double v, omega;
            computeControl(robot_pose, target_pose, v, omega);

            // Publish velocity command
            geometry_msgs::msg::Twist cmd_vel;
            cmd_vel.linear.x = v;
            cmd_vel.angular.z = omega;
            cmd_vel_publisher_->publish(cmd_vel);

            // Publish feedback
            auto feedback = std::make_shared<FollowPath::Feedback>();
            feedback->current_waypoint = current_waypoint_index_;
            feedback->total_waypoints = current_path_.poses.size();
            current_goal_handle_->publish_feedback(feedback);

            // Check waypoint completion
            if (current_waypoint_index_ == current_path_.poses.size() - 1) {
                double dx = target_pose.position.x - robot_pose.position.x;
                double dy = target_pose.position.y - robot_pose.position.y;
                double distance = std::sqrt(dx*dx + dy*dy);

                if (distance < goal_tolerance_) {
                    current_waypoint_index_++;
                    RCLCPP_INFO(this->get_logger(), "Goal final atingido");
                }
            } else if (current_waypoint_index_ < current_path_.poses.size() - 1) {
                double dx = target_pose.position.x - robot_pose.position.x;
                double dy = target_pose.position.y - robot_pose.position.y;
                double distance = std::sqrt(dx*dx + dy*dy);

                if (distance < goal_tolerance_) {
                    current_waypoint_index_++;
                    RCLCPP_INFO(this->get_logger(), "Waypoint %zu atingido", current_waypoint_index_ - 1);
                }
            }
        } else {
            // Goal is being cancelled - send cancelled result
            RCLCPP_INFO(this->get_logger(), "Sending cancellation result");
            auto result = std::make_shared<FollowPath::Result>();
            result->success = false;
            result->message = "Goal cancelled";
            current_goal_handle_->canceled(result);
            
            // Stop the robot
            geometry_msgs::msg::Twist stop_msg;
            cmd_vel_publisher_->publish(stop_msg);
            
            path_active_ = false;
            current_goal_handle_.reset();
        }
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SaganKanayamaController>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}