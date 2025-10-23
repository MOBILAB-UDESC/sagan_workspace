#ifndef SAGAN_CONTROL_HARDWARE_INTERFACE_HPP_
#define SAGAN_CONTROL_HARDWARE_INTERFACE_HPP_

#include <memory>
#include <string>
#include <vector>

#include "hardware_interface/handle.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "rclcpp/clock.hpp"
#include "rclcpp/duration.hpp"
#include "rclcpp/macros.hpp"
#include "rclcpp/time.hpp"
#include "rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp"
#include "rclcpp_lifecycle/state.hpp"

namespace sagan_control_hardware_interface
{

// --- Define the I2C protocol structs (must match Pico firmware) ---
#pragma pack(push, 1)
// Master -> Slave
struct ControlData {
    uint8_t cmd;         // Command (e.g., 0xA1)
    int16_t front_velocity;   // Target velocity * 100
    int16_t rear_velocity;    // Target velocity * 100
};

// Slave -> Master
struct SensorData {
    int16_t front_velocity;     // Actual velocity * 100
    int16_t front_current;      // Current in mA
    int16_t rear_velocity;     // Actual velocity * 100
    int16_t rear_current;      // Current in mA
};
#pragma pack(pop)

// --- Conversion factor (must match Pico firmware) ---
const float DATA_SCALE_FACTOR = 100.0f;
const float CURRENT_SCALE_FACTOR = 1000.0f; // From mA to A

class SaganControlHardwareInterface : public hardware_interface::SystemInterface
{
public:
  RCLCPP_SHARED_PTR_DEFINITIONS(SaganControlHardwareInterface)

  hardware_interface::CallbackReturn on_init(
    const hardware_interface::HardwareInfo & info) override;

  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;

  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  hardware_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::return_type read(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

  hardware_interface::return_type write(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

  hardware_interface::CallbackReturn on_shutdown(
    const rclcpp_lifecycle::State & previous_state) override;

private:
  int i2c_left_handle_ = -1;
  int i2c_right_handle_ = -1;
  
  // Parameters from URDF
  std::string i2c_bus_path_;
  int left_pico_addr_ = 0;
  int right_pico_addr_ = 0;

  // Store for joint and sensor names
  std::string front_left_joint_name_, rear_left_joint_name_, front_right_joint_name_, rear_right_joint_name_;
  std::string front_left_sensor_name_, rear_left_sensor_name_, front_right_sensor_name_, rear_right_sensor_name_;

  // Store for ros2_control interfaces
  std::vector<double> hw_commands_velocities_;
  std::vector<double> hw_states_positions_; // We will integrate velocity to get position
  std::vector<double> hw_states_velocities_;
  std::vector<double> hw_states_currents_;
};

}  // namespace sagan_control_hardware_interface

#endif  // SAGAN_CONTROL_HARDWARE_INTERFACE

