#ifndef MPU9250DRIVER_H
#define MPU9250DRIVER_H

#include "mpu9250sensor.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "std_srvs/srv/trigger.hpp"

#include <array>
#include <fstream>
#include <filesystem>
#include <thread>
#include <atomic>

class MPU9250Driver : public rclcpp::Node
{
public:
  MPU9250Driver();

private:
  // --- Sensor ---
  std::unique_ptr<MPU9250Sensor> mpu9250_;
  int consecutive_error_count_;

  // --- Publisher & Timer ---
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;

  // --- Magnetometer calibration state ---
  std::array<float, 3> mag_scale_  = {1.0f, 1.0f, 1.0f};
  std::array<float, 3> mag_offset_ = {35.0f, 35.0f, 35.0f};
  std::atomic<bool> is_calibrating_{false};

  // --- Calibration service ---
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr calibration_service_;

  // --- Calibration thread ---
  std::thread calibration_thread_;

  // --- Methods ---
  void handleInput();
  void declareParameters();
  void calculateOrientation(sensor_msgs::msg::Imu& imu_message);

  // Called by the service — spawns calibration_thread_
  void handle_calibration(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response);

  // Runs in background thread — contains your existing calibration logic
  void execute_calibration(int num_samples);
};

#endif  // MPU9250DRIVER_H