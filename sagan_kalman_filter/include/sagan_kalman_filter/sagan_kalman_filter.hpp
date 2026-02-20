#ifndef SAGAN_KALMAN_FILTER_HPP_
#define SAGAN_KALMAN_FILTER_HPP_

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <Eigen/Dense>
#include <vector>
#include <string>

// Definição do tamanho do estado: [x, y, vx, vy, ax, ay, theta, omega]
const int STATE_SIZE = 8;

class SaganKalmanFilter : public rclcpp::Node
{
public:
    SaganKalmanFilter();

private:
    // --- Tipos Eigen para facilitar a leitura ---
    using StateVector = Eigen::Matrix<double, STATE_SIZE, 1>;
    using StateCovariance = Eigen::Matrix<double, STATE_SIZE, STATE_SIZE>;
    using ProcessNoiseCovariance = Eigen::Matrix<double, STATE_SIZE, STATE_SIZE>;

    // --- Métodos do Filtro ---
    void predict(double dt);
    void update(const Eigen::VectorXd& z, const Eigen::MatrixXd& H, const Eigen::MatrixXd& R);
    
    // Auxiliar para processar as medições de forma dinâmica
    void execute_sensor_update(double measurements[], std::vector<bool>& map, Eigen::MatrixXd& R);

    // --- Callbacks de Sensores ---
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg);

    // --- Publicação e Transformadas ---
    void publish_fused_odometry(rclcpp::Time stamp);

    // --- Variáveis de Estado e Covariância ---
    StateVector x_;           // Vetor de estado
    StateCovariance P_;       // Matriz de covariância do erro
    ProcessNoiseCovariance Q_; // Matriz de ruído de processo

    // --- Configuração de Sensores ---
    std::vector<bool> odom_measurement_map_;
    std::vector<bool> imu_measurement_map_;
    Eigen::MatrixXd R_odom_;
    Eigen::MatrixXd R_imu_;

    // --- Sincronização de Tempo ---
    rclcpp::Time last_predict_time_;
    bool is_initialized_;

    // --- Parâmetros ---
    std::string odom_frame_id_;
    std::string base_frame_id_;

    // --- Objetos ROS 2 ---
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr fused_odom_pub_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

#endif // SAGAN_KALMAN_FILTER_HPP_