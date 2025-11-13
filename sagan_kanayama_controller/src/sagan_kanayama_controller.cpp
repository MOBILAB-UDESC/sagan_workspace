#include <rclcpp/rclcpp.hpp>
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
        this->declare_parameter("max_angular_velocity", 1.0);
        this->declare_parameter("lookahead_distance", 0.5);
        this->declare_parameter("robot_base_frame", "base_link");
        this->declare_parameter("odom_frame", "odom");

        // Obter parâmetros
        k_x_ = this->get_parameter("k_x").as_double();
        k_y_ = this->get_parameter("k_y").as_double();
        k_theta_ = this->get_parameter("k_theta").as_double();
        max_linear_velocity_ = this->get_parameter("max_linear_velocity").as_double();
        max_angular_velocity_ = this->get_parameter("max_angular_velocity").as_double();
        lookahead_distance_ = this->get_parameter("lookahead_distance").as_double();
        robot_base_frame_ = this->get_parameter("robot_base_frame").as_string();
        odom_frame_ = this->get_parameter("odom_frame").as_string();

        // Inicializar TF
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // Publicadores
        cmd_vel_publisher_ = this->create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 10);

        // Subscritores
        path_subscriber_ = this->create_subscription<nav_msgs::msg::Path>(
            "path", 10, std::bind(&SaganKanayamaController::pathCallback, this, std::placeholders::_1));
        
        odom_subscriber_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "odom", 10, std::bind(&SaganKanayamaController::odomCallback, this, std::placeholders::_1));

        // Timer para controle
        control_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(50),  // 20 Hz
            std::bind(&SaganKanayamaController::controlLoop, this));

        RCLCPP_INFO(this->get_logger(), "Controlador Kanayama inicializado");
        RCLCPP_INFO(this->get_logger(), "Ganhos: Kx=%.2f, Ky=%.2f, Kθ=%.2f", k_x_, k_y_, k_theta_);
    }

private:
    // Parâmetros do controlador
    double k_x_, k_y_, k_theta_;
    double max_linear_velocity_, max_angular_velocity_;
    double lookahead_distance_;
    std::string robot_base_frame_, odom_frame_;

    // Estado atual do robô
    geometry_msgs::msg::Pose current_pose_;
    bool pose_received_ = false;
    
    // Trajetória de referência
    nav_msgs::msg::Path current_path_;
    bool path_received_ = false;
    size_t current_waypoint_index_ = 0;

    // TF
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // Publicadores e Subscritores
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_publisher_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_subscriber_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscriber_;
    rclcpp::TimerBase::SharedPtr control_timer_;

    void pathCallback(const nav_msgs::msg::Path::SharedPtr msg)
    {
        if (!msg->poses.empty()) {
            current_path_ = *msg;
            current_waypoint_index_ = 0;
            path_received_ = true;
            RCLCPP_INFO(this->get_logger(), "Nova trajetória recebida com %zu pontos", msg->poses.size());
        }
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

        RCLCPP_DEBUG(this->get_logger(), 
                    "Erros: x_e=%.3f, y_e=%.3f, θ_e=%.3f | Comando: v=%.3f, ω=%.3f", 
                    x_e, y_e, theta_e, v, omega);
    }

    void controlLoop()
    {
        if (!path_received_ || !pose_received_) {
            return;
        }

        if (current_waypoint_index_ >= current_path_.poses.size()) {
            // Parar quando atingir o final da trajetória
            geometry_msgs::msg::Twist stop_msg;
            cmd_vel_publisher_->publish(stop_msg);
            return;
        }

        // Obter pose atual do robô
        geometry_msgs::msg::Pose robot_pose = getRobotPose();

        // Encontrar waypoint alvo
        current_waypoint_index_ = findTargetWaypoint(robot_pose);
        geometry_msgs::msg::Pose target_pose = current_path_.poses[current_waypoint_index_].pose;

        // Calcular comando de controle
        double v, omega;
        computeControl(robot_pose, target_pose, v, omega);

        // Publicar comando de velocidade
        geometry_msgs::msg::Twist cmd_vel;
        cmd_vel.linear.x = v;
        cmd_vel.angular.z = omega;
        cmd_vel_publisher_->publish(cmd_vel);

        // Verificar se atingiu o waypoint atual
        double dx = target_pose.position.x - robot_pose.position.x;
        double dy = target_pose.position.y - robot_pose.position.y;
        double distance = std::sqrt(dx*dx + dy*dy);

        if (distance < 0.1 && current_waypoint_index_ < current_path_.poses.size() - 1) {
            current_waypoint_index_++;
            RCLCPP_INFO(this->get_logger(), "Waypoint %zu atingido, indo para %zu", 
                       current_waypoint_index_ - 1, current_waypoint_index_);
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