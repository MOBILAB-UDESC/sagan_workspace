#include "mpu9250driver/mpu9250driver.h"

#include <chrono>
#include <memory>
#include <stdexcept>
#include <thread>
#include <algorithm>

#include "LinuxI2cCommunicator.h"

using namespace std::chrono_literals;

MPU9250Driver::MPU9250Driver() : Node("mpu9250publisher"), consecutive_error_count_(0)
{
  try {
  // Create concrete I2C communicator and pass to sensor
  std::unique_ptr<I2cCommunicator> i2cBus = std::make_unique<LinuxI2cCommunicator>();
  mpu9250_ = std::make_unique<MPU9250Sensor>(std::move(i2cBus));
  // Declare parameters
  declareParameters();
  // Set parameters
  mpu9250_->setGyroscopeRange(
      static_cast<MPU9250Sensor::GyroRange>(this->get_parameter("gyro_range").as_int()));
  mpu9250_->setAccelerometerRange(
      static_cast<MPU9250Sensor::AccelRange>(this->get_parameter("accel_range").as_int()));
  mpu9250_->setDlpfBandwidth(
      static_cast<MPU9250Sensor::DlpfBandwidth>(this->get_parameter("dlpf_bandwidth").as_int()));
  mpu9250_->setGyroscopeOffset(this->get_parameter("gyro_x_offset").as_double(),
                               this->get_parameter("gyro_y_offset").as_double(),
                               this->get_parameter("gyro_z_offset").as_double());
  mpu9250_->setAccelerometerOffset(this->get_parameter("accel_x_offset").as_double(),
                                   this->get_parameter("accel_y_offset").as_double(),
                                   this->get_parameter("accel_z_offset").as_double());
  // Check if we want to calibrate the sensor
  if (this->get_parameter("calibrate").as_bool()) {
    RCLCPP_INFO(this->get_logger(), "Calibrating...");
    mpu9250_->calibrate();
  }
  this->declare_parameter<double>("mag_x_offset", 0.0);
  this->declare_parameter<double>("mag_y_offset", 0.0);
  this->declare_parameter<double>("mag_z_offset", 0.0);
  this->declare_parameter<double>("mag_x_scale",  1.0);
  this->declare_parameter<double>("mag_y_scale",  1.0);
  this->declare_parameter<double>("mag_z_scale",  1.0);

  mpu9250_->setMagnetometerOffset(
      this->get_parameter("mag_x_offset").as_double(),
      this->get_parameter("mag_y_offset").as_double(),
      this->get_parameter("mag_z_offset").as_double());

  mpu9250_->printConfig();
  mpu9250_->printOffsets();
  } catch (const std::runtime_error& e) {
      RCLCPP_FATAL(this->get_logger(), "IMU initialization failed: %s", e.what());
      rclcpp::shutdown();
      return;
  }
  // Create publisher
  publisher_ = this->create_publisher<sensor_msgs::msg::Imu>("imu", 10);
  std::chrono::duration<int64_t, std::milli> frequency =
      1000ms / this->get_parameter("gyro_range").as_int();
  timer_ = this->create_wall_timer(frequency, std::bind(&MPU9250Driver::handleInput, this));

  calibration_service_ = this->create_service<std_srvs::srv::Trigger>(
      "calibrate_magnetometer",
      std::bind(&MPU9250Driver::handle_calibration, this,
                std::placeholders::_1, std::placeholders::_2));
}

void MPU9250Driver::handleInput()
{
  const int ERROR_LIMIT = 5;
  try {
    auto message = sensor_msgs::msg::Imu();
    message.header.stamp = this->get_clock()->now();
    message.header.frame_id = "base_link";
    // Direct measurements
    message.linear_acceleration_covariance = {0};
    message.linear_acceleration.x = mpu9250_->getAccelerationX();
    message.linear_acceleration.y = mpu9250_->getAccelerationY();
    message.linear_acceleration.z = mpu9250_->getAccelerationZ();
    message.angular_velocity_covariance[0] = {0};
    message.angular_velocity.x = mpu9250_->getAngularVelocityX();
    message.angular_velocity.y = mpu9250_->getAngularVelocityY();
    message.angular_velocity.z = mpu9250_->getAngularVelocityZ();
    // Calculate euler angles, convert to quaternion and store in message
    message.orientation_covariance = {0};
    calculateOrientation(message);
    publisher_->publish(message);
    consecutive_error_count_ = 0;
  } catch (const std::runtime_error& e) {
    RCLCPP_ERROR(this->get_logger(), "IMU communication error: %s", e.what());
    consecutive_error_count_++;
    if (consecutive_error_count_ >= ERROR_LIMIT) {
      RCLCPP_FATAL(this->get_logger(), "IMU failed to read data %d times. Shutting down.", ERROR_LIMIT);
      rclcpp::shutdown();
    }
  }
}

void MPU9250Driver::declareParameters()
{
  this->declare_parameter<bool>("calibrate", true);
  this->declare_parameter<int>("gyro_range", MPU9250Sensor::GyroRange::GYR_250_DEG_S);
  this->declare_parameter<int>("accel_range", MPU9250Sensor::AccelRange::ACC_2_G);
  this->declare_parameter<int>("dlpf_bandwidth", MPU9250Sensor::DlpfBandwidth::DLPF_260_HZ);
  this->declare_parameter<double>("gyro_x_offset", 0.0);
  this->declare_parameter<double>("gyro_y_offset", 0.0);
  this->declare_parameter<double>("gyro_z_offset", 0.0);
  this->declare_parameter<double>("accel_x_offset", 0.0);
  this->declare_parameter<double>("accel_y_offset", 0.0);
  this->declare_parameter<double>("accel_z_offset", 0.0);
  this->declare_parameter<int>("frequency", 0);
}

void MPU9250Driver::calculateOrientation(sensor_msgs::msg::Imu& imu_message)
{
    double ax = imu_message.linear_acceleration.x;
    double ay = imu_message.linear_acceleration.y;
    double az = imu_message.linear_acceleration.z;

    double roll  = atan2(ay, az);
    double pitch = atan2(-ax, sqrt(ay*ay + az*az));  // ✅ more stable than /az

    try {
        // Apply soft iron scale correction
        double mx = mpu9250_->getMagneticFluxDensityX() * mag_scale_[0];
        double my = mpu9250_->getMagneticFluxDensityY() * mag_scale_[1];
        double mz = mpu9250_->getMagneticFluxDensityZ() * mag_scale_[2];

        RCLCPP_INFO(get_logger(), "Magnectic Flux: X: %f, Y: %f, Z: %f", mx, my, mz);
        // (hard iron offset is already applied inside setMagnetometerOffset)

        // ✅ Tilt-compensated yaw
        double mx_h = mx * cos(pitch)
                    + my * sin(pitch) * sin(roll)
                    + mz * sin(pitch) * cos(roll);
        double my_h = my * cos(roll)
                    - mz * sin(roll);

        double yaw = atan2(-my_h, mx_h);

        // Euler to quaternion
        double cy = cos(yaw   * 0.5), sy = sin(yaw   * 0.5);
        double cp = cos(pitch * 0.5), sp = sin(pitch * 0.5);
        double cr = cos(roll  * 0.5), sr = sin(roll  * 0.5);

        imu_message.orientation.w = cy*cp*cr + sy*sp*sr;
        imu_message.orientation.x = cy*cp*sr - sy*sp*cr;
        imu_message.orientation.y = sy*cp*sr + cy*sp*cr;
        imu_message.orientation.z = sy*cp*cr - cy*sp*sr;

    } catch (const std::runtime_error& e) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
            "Mag not responding, roll/pitch only: %s", e.what());
        // fallback: yaw = 0
        double cp = cos(pitch*0.5), sp = sin(pitch*0.5);
        double cr = cos(roll *0.5), sr = sin(roll *0.5);
        imu_message.orientation.w = cp*cr;
        imu_message.orientation.x = cp*sr;
        imu_message.orientation.y = sp*cr;
        imu_message.orientation.z = -sp*sr;
    }
}

void MPU9250Driver::handle_calibration(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
    if (is_calibrating_) {
        response->success = false;
        response->message = "Calibration already in progress";
        return;
    }
    // Spawn thread so service call returns immediately
    if (calibration_thread_.joinable()) calibration_thread_.join();
    calibration_thread_ = std::thread(
        &MPU9250Driver::execute_calibration, this, 300); // 300 samples
    response->success = true;
    response->message = "Calibration started — rotate the sensor in all directions!";
}

void MPU9250Driver::execute_calibration(int num_samples)
{
    is_calibrating_ = true;
    float min_x = 1e9,  max_x = -1e9;
    float min_y = 1e9,  max_y = -1e9;
    float min_z = 1e9,  max_z = -1e9;

    std::vector<float> magn_x_samples(num_samples, 0.0);
    std::vector<float> magn_y_samples(num_samples, 0.0);
    std::vector<float> magn_z_samples(num_samples, 0.0);

    RCLCPP_INFO(get_logger(), "Calibration started: %d samples at 20Hz (~%.0fs)",
        num_samples, num_samples / 20.0);

    rclcpp::Rate rate(20);
    for (int i = 0; i < num_samples && rclcpp::ok(); i++) {
        float x = static_cast<float>(mpu9250_->getMagneticFluxDensityX());
        float y = static_cast<float>(mpu9250_->getMagneticFluxDensityY());
        float z = static_cast<float>(mpu9250_->getMagneticFluxDensityZ());

        magn_x_samples[i] = x;
        magn_y_samples[i] = y;
        magn_z_samples[i] = z;

        if (i % 20 == 0) // log every second
            RCLCPP_INFO(get_logger(), "Collecting... %d/%d", i, num_samples);
        rate.sleep();
    }

    std::sort(magn_x_samples.begin(), magn_x_samples.end());
    std::sort(magn_y_samples.begin(), magn_y_samples.end());
    std::sort(magn_z_samples.begin(), magn_z_samples.end());

    //Find the index for the 2% and 98% marks
    int lower_index = magn_x_samples.size() * 0.02; // Skips the bottom 2%
    int upper_index = magn_x_samples.size() * 0.98; // Skips the top 2%  

    max_x = magn_x_samples[upper_index]; min_x = magn_x_samples[lower_index];
    max_y = magn_y_samples[upper_index]; min_y = magn_y_samples[lower_index];
    max_z = magn_z_samples[upper_index]; min_z = magn_z_samples[lower_index];

    // Hard iron offsets
    float ox = (max_x + min_x) / 2.0f;
    float oy = (max_y + min_y) / 2.0f;
    float oz = (max_z + min_z) / 2.0f;

    // Soft iron scales
    float avg = ((max_x-min_x) + (max_y-min_y) + (max_z-min_z)) / 3.0f;
    float sx = avg / (max_x - min_x);
    float sy = avg / (max_y - min_y);
    float sz = avg / (max_z - min_z);

    mpu9250_->setMagnetometerOffset(ox, oy, oz);
    mag_scale_  = {sx, sy, sz};
    mag_offset_ = {ox, oy, oz};

    this->set_parameter(rclcpp::Parameter("mag_x_offset", (double)ox));
    this->set_parameter(rclcpp::Parameter("mag_y_offset", (double)oy));
    this->set_parameter(rclcpp::Parameter("mag_z_offset", (double)oz));
    this->set_parameter(rclcpp::Parameter("mag_x_scale",  (double)sx));
    this->set_parameter(rclcpp::Parameter("mag_y_scale",  (double)sy));
    this->set_parameter(rclcpp::Parameter("mag_z_scale",  (double)sz));

    RCLCPP_INFO(get_logger(),
        "Calibration done! Offsets:[%.3f,%.3f,%.3f] Scales:[%.3f,%.3f,%.3f]",
        ox, oy, oz, sx, sy, sz);

    is_calibrating_ = false;
}

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MPU9250Driver>());
  rclcpp::shutdown();
  return 0;
}
