#include "sagan_kalman_filter/sagan_kalman_filter.hpp"
#include <memory>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/parameter.hpp>
#include <cmath>

// Helper para configurar sensores (Mantido conforme original)
void configure_sensor(
    rclcpp::Node* node, 
    const std::string& sensor_name,
    std::vector<bool>& measurement_map, 
    Eigen::MatrixXd& R_matrix)
{
    node->declare_parameter<std::vector<bool>>(sensor_name + ".measurement_map", std::vector<bool>(STATE_SIZE, false));
    node->declare_parameter<std::vector<double>>(sensor_name + ".R_diag", {});

    measurement_map = node->get_parameter(sensor_name + ".measurement_map").as_bool_array();
    auto r_diag = node->get_parameter(sensor_name + ".R_diag").as_double_array();

    int measurement_count = 0;
    for(bool enabled : measurement_map) if(enabled) measurement_count++;

    R_matrix = Eigen::MatrixXd::Zero(measurement_count, measurement_count);
    for (int i = 0; i < measurement_count; ++i) R_matrix(i, i) = r_diag[i];
    
    RCLCPP_INFO(node->get_logger(), "Configured sensor '%s' with %d active measurements.", sensor_name.c_str(), measurement_count);
}

SaganKalmanFilter::SaganKalmanFilter() : Node("sagan_kalman_filter"), is_initialized_(false)
{
    // --- Declaração de Parâmetros ---
    this->declare_parameter<std::vector<double>>("Q_diag", std::vector<double>(STATE_SIZE, 0.01));
    this->declare_parameter<std::string>("odom_frame_id", "odom");
    this->declare_parameter<std::string>("base_frame_id", "base_footprint"); // Alterado para base_footprint conforme Nav2

    odom_frame_id_ = this->get_parameter("odom_frame_id").as_string();
    base_frame_id_ = this->get_parameter("base_frame_id").as_string();
    std::vector<double> q_diag = this->get_parameter("Q_diag").as_double_array();

    // --- Inicialização de Estado ---
    x_ = StateVector::Zero();
    P_ = StateCovariance::Identity() * 0.1; // Reduzido incerteza inicial (1000 era muito alto para o AG)
    
    Q_ = ProcessNoiseCovariance::Zero();
    for(int i = 0; i < STATE_SIZE; ++i) Q_(i, i) = q_diag[i];
    
    configure_sensor(this, "odom_sensor", odom_measurement_map_, R_odom_);
    configure_sensor(this, "imu_sensor", imu_measurement_map_, R_imu_);

    // --- Comunicação ---
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/odom/with_noise", 10, std::bind(&SaganKalmanFilter::odom_callback, this, std::placeholders::_1));
    
    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
        "/imu", 10, std::bind(&SaganKalmanFilter::imu_callback, this, std::placeholders::_1));

    fused_odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/odom/filtered", 10);
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    RCLCPP_INFO(this->get_logger(), "Kalman Filter Node started (Event-driven mode).");
}

void SaganKalmanFilter::predict(double dt)
{
    if (dt <= 0.0) return;

    double vx = x_(2); double vy = x_(3);
    double ax = x_(4); double ay = x_(5);
    double theta = x_(6); double omega = x_(7);

    double ct = std::cos(theta); double st = std::sin(theta);
    double dt2 = 0.5 * dt * dt;

    // --- Matriz F (Jacobiano) Corrigida ---
    Eigen::Matrix<double, STATE_SIZE, STATE_SIZE> F = Eigen::Matrix<double, STATE_SIZE, STATE_SIZE>::Identity();
    F(0, 2) = ct * dt;  F(0, 3) = -st * dt; F(0, 4) = ct * dt2; F(0, 5) = -st * dt2;
    F(0, 6) = (-vx * st - vy * ct) * dt + (-ax * st - ay * ct) * dt2; // Sensibilidade do Yaw
    
    F(1, 2) = st * dt;  F(1, 3) = ct * dt;  F(1, 4) = st * dt2; F(1, 5) = ct * dt2;
    F(1, 6) = (vx * ct - vy * st) * dt + (ax * ct - ay * st) * dt2;

    F(2, 4) = dt; F(3, 5) = dt; F(6, 7) = dt;

    // --- Evolução do Estado (Modelo Cinemático) ---
    x_(0) += (vx * ct - vy * st) * dt + (ax * ct - ay * st) * dt2;
    x_(1) += (vx * st + vy * ct) * dt + (ax * st + ay * ct) * dt2;
    x_(2) += ax * dt;
    x_(3) += ay * dt;
    x_(6) += omega * dt;

    // *** NORMALIZAÇÃO PREDIÇÃO ***
    while (x_(6) > M_PI) x_(6) -= 2.0 * M_PI;
    while (x_(6) < -M_PI) x_(6) += 2.0 * M_PI;

    P_ = F * P_ * F.transpose() + Q_;
}

void SaganKalmanFilter::update(const Eigen::VectorXd& z, const Eigen::MatrixXd& H, const Eigen::MatrixXd& R)
{
    Eigen::VectorXd y = z - H * x_;

    // *** ANGLE WRAPPING INOVAÇÃO ***
    for (int i = 0; i < H.rows(); ++i) {
        if (H(i, 6) == 1.0) { 
            while (y(i) > M_PI) y(i) -= 2.0 * M_PI;
            while (y(i) < -M_PI) y(i) += 2.0 * M_PI;
        }
    }

    Eigen::MatrixXd S = H * P_ * H.transpose() + R;
    Eigen::MatrixXd K = P_ * H.transpose() * S.inverse();

    x_ = x_ + K * y;
    P_ = (Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE) - K * H) * P_;
    
    // *** NORMALIZAÇÃO FINAL DO ESTADO ***
    while (x_(6) > M_PI) x_(6) -= 2.0 * M_PI;
    while (x_(6) < -M_PI) x_(6) += 2.0 * M_PI;
}

void SaganKalmanFilter::odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    rclcpp::Time current_time = msg->header.stamp;
    if (!is_initialized_) {
        last_predict_time_ = current_time;
        is_initialized_ = true;
        return;
    }

    double dt = (current_time - last_predict_time_).seconds();
    this->predict(dt);
    last_predict_time_ = current_time;

    tf2::Quaternion q; tf2::fromMsg(msg->pose.pose.orientation, q);
    double r, p, yaw; tf2::Matrix3x3(q).getRPY(r, p, yaw);

    double measurements[STATE_SIZE] = {
        msg->pose.pose.position.x, msg->pose.pose.position.y,
        msg->twist.twist.linear.x, msg->twist.twist.linear.y,
        0.0, 0.0, yaw, msg->twist.twist.angular.z
    };

    execute_sensor_update(measurements, odom_measurement_map_, R_odom_);
    publish_fused_odometry(current_time);
}

void SaganKalmanFilter::imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
    rclcpp::Time current_time = msg->header.stamp;
    if (!is_initialized_) return;

    double dt = (current_time - last_predict_time_).seconds();
    this->predict(dt);
    last_predict_time_ = current_time;

    tf2::Quaternion q; tf2::fromMsg(msg->orientation, q);
    double r, p, yaw; tf2::Matrix3x3(q).getRPY(r, p, yaw);

    double measurements[STATE_SIZE] = {
        0.0, 0.0, 0.0, 0.0,
        msg->linear_acceleration.x, msg->linear_acceleration.y, // Removido sinal negativo para teste
        yaw, msg->angular_velocity.z
    };

    execute_sensor_update(measurements, imu_measurement_map_, R_imu_);
}

void SaganKalmanFilter::execute_sensor_update(double measurements[], std::vector<bool>& map, Eigen::MatrixXd& R)
{
    int active = R.rows();
    if (active == 0) return;

    Eigen::VectorXd z(active);
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(active, STATE_SIZE);
    int row = 0;
    for (int i = 0; i < STATE_SIZE; ++i) {
        if (map[i]) {
            z(row) = measurements[i];
            H(row, i) = 1.0;
            row++;
        }
    }
    update(z, H, R);
}

void SaganKalmanFilter::publish_fused_odometry(rclcpp::Time stamp)
{
    auto msg = nav_msgs::msg::Odometry();
    msg.header.stamp = stamp;
    msg.header.frame_id = odom_frame_id_;
    msg.child_frame_id = base_frame_id_;

    msg.pose.pose.position.x = x_(0);
    msg.pose.pose.position.y = x_(1);
    
    tf2::Quaternion q; q.setRPY(0, 0, x_(6));
    msg.pose.pose.orientation = tf2::toMsg(q);

    // Mapeamento de Covariância simplificado (x, y, yaw)
    msg.pose.covariance[0] = P_(0,0); msg.pose.covariance[7] = P_(1,1); msg.pose.covariance[35] = P_(6,6);

    msg.twist.twist.linear.x = x_(2);
    msg.twist.twist.linear.y = x_(3);
    msg.twist.twist.angular.z = x_(7);

    fused_odom_pub_->publish(msg);

    // TF Broadcaster
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = stamp;
    t.header.frame_id = odom_frame_id_;
    t.child_frame_id = base_frame_id_;
    t.transform.translation.x = x_(0);
    t.transform.translation.y = x_(1);
    t.transform.rotation = tf2::toMsg(q);
    tf_broadcaster_->sendTransform(t);
}

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SaganKalmanFilter>());
    rclcpp::shutdown();
    return 0;
}