#include "sagan_control_hardware_interface/sagan_control_hardware_interface.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <vector>

#include "hardware_interface/lexical_casts.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/rclcpp.hpp"

// --- Linux I2C Libraries ---
#include <fcntl.h>       // For open()
#include <sys/ioctl.h>   // For ioctl()
#include <linux/i2c-dev.h> // For I2C_SLAVE
extern "C" {
  #include <i2c/smbus.h> // For i2c_smbus_... functions
}
#include <unistd.h>      // For close()

namespace sagan_control_hardware_interface
{
hardware_interface::CallbackReturn SaganControlHardwareInterface::on_init(
  const hardware_interface::HardwareInfo & info)
{
  if (
    hardware_interface::SystemInterface::on_init(info) !=
    hardware_interface::CallbackReturn::SUCCESS)
  {
    return hardware_interface::CallbackReturn::ERROR;
  }

  RCLCPP_INFO(get_logger(), "Initializing Sagan Pico Hardware Interface...");

  // --- Get Parameters from URDF ---
  // Note: We use info_ (a member of the base class) because we're using
  // the 'HardwareInfo' struct in our on_init signature.
  i2c_bus_path_ = info_.hardware_parameters["i2c_bus_path"];
  left_pico_addr_ = std::stoi(info_.hardware_parameters["left_pico_addr"], 0, 16);
  right_pico_addr_ = std::stoi(info_.hardware_parameters["right_pico_addr"], 0, 16);

  RCLCPP_INFO(get_logger(), "Hardware Parameters:");
  RCLCPP_INFO(get_logger(), " - I2C Bus: %s", i2c_bus_path_.c_str());
  RCLCPP_INFO(get_logger(), " - Left Pico Addr: 0x%X", left_pico_addr_);
  RCLCPP_INFO(get_logger(), " - Right Pico Addr: 0x%X", right_pico_addr_);
  
  // --- Open I2C Bus for Left Pico ---
  i2c_left_handle_ = open(i2c_bus_path_.c_str(), O_RDWR);
  if (i2c_left_handle_ < 0) {
      RCLCPP_ERROR(get_logger(), "Failed to open I2C bus %s for left Pico", i2c_bus_path_.c_str());
      return hardware_interface::CallbackReturn::ERROR;
  }
  if (ioctl(i2c_left_handle_, I2C_SLAVE, left_pico_addr_) < 0) {
      RCLCPP_ERROR(get_logger(), "Failed to set I2C slave address 0x%X for left Pico", left_pico_addr_);
      return hardware_interface::CallbackReturn::ERROR;
  }
  
  // --- Open I2C Bus for Right Pico ---
  i2c_right_handle_ = open(i2c_bus_path_.c_str(), O_RDWR);
  if (i2c_right_handle_ < 0) {
      RCLCPP_ERROR(get_logger(), "Failed to open I2C bus %s for right Pico", i2c_bus_path_.c_str());
      return hardware_interface::CallbackReturn::ERROR;
  }
  if (ioctl(i2c_right_handle_, I2C_SLAVE, right_pico_addr_) < 0) {
      RCLCPP_ERROR(get_logger(), "Failed to set I2C slave address 0x%X for right Pico", right_pico_addr_);
      return hardware_interface::CallbackReturn::ERROR;
  }

  // --- Check for expected 4 joints and 4 sensors ---
  if (info_.joints.size() != 4) {
    RCLCPP_ERROR(get_logger(), "Expected 4 joints, but got %zu", info_.joints.size());
    return hardware_interface::CallbackReturn::ERROR;
  }
  if (info_.sensors.size() != 4) {
    RCLCPP_ERROR(get_logger(), "Expected 4 sensors, but got %zu", info_.sensors.size());
    return hardware_interface::CallbackReturn::ERROR;
  }

  // Initialize state and command vectors
  hw_commands_velocities_.resize(4, 0.0);
  hw_states_positions_.resize(4, 0.0);
  hw_states_velocities_.resize(4, 0.0);
  hw_states_currents_.resize(4, 0.0);

  // Store joint and sensor names from URDF
  front_left_joint_name_ = info_.joints[0].name;
  rear_left_joint_name_ = info_.joints[1].name;
  front_right_joint_name_ = info_.joints[2].name;
  rear_right_joint_name_ = info_.joints[3].name;

  front_left_sensor_name_ = info_.sensors[0].name;
  rear_left_sensor_name_ = info_.sensors[1].name;
  front_right_sensor_name_ = info_.sensors[2].name;
  rear_right_sensor_name_ = info_.sensors[3].name; // Corrected from info_.joints[3]

  // --- Validate joint interfaces (from your provided code) ---
  for (const hardware_interface::ComponentInfo & joint : info_.joints)
  {
    if (joint.command_interfaces.size() != 1)
    {
      RCLCPP_FATAL(
        get_logger(), "Joint '%s' has %zu command interfaces found. 1 expected.",
        joint.name.c_str(), joint.command_interfaces.size());
      return hardware_interface::CallbackReturn::ERROR;
    }
    if (joint.command_interfaces[0].name != hardware_interface::HW_IF_VELOCITY)
    {
      RCLCPP_FATAL(
        get_logger(), "Joint '%s' have %s command interfaces found. '%s' expected.",
        joint.name.c_str(), joint.command_interfaces[0].name.c_str(),
        hardware_interface::HW_IF_VELOCITY);
      return hardware_interface::CallbackReturn::ERROR;
    }
    if (joint.state_interfaces.size() != 2) // We expect position and velocity
    {
      RCLCPP_FATAL(
        get_logger(), "Joint '%s' has %zu state interface. 2 expected (position, velocity).", joint.name.c_str(),
        joint.state_interfaces.size());
      return hardware_interface::CallbackReturn::ERROR;
    }
  }

  RCLCPP_INFO(get_logger(), "Hardware Initialized Successfully.");
  return hardware_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> SaganControlHardwareInterface::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> state_interfaces;

  // Export 4 joints, each with 2 states (position, velocity)
  state_interfaces.emplace_back(front_left_joint_name_, "position", &hw_states_positions_[0]);
  state_interfaces.emplace_back(front_left_joint_name_, "velocity", &hw_states_velocities_[0]);
  state_interfaces.emplace_back(rear_left_joint_name_, "position", &hw_states_positions_[1]);
  state_interfaces.emplace_back(rear_left_joint_name_, "velocity", &hw_states_velocities_[1]);
  state_interfaces.emplace_back(front_right_joint_name_, "position", &hw_states_positions_[2]);
  state_interfaces.emplace_back(front_right_joint_name_, "velocity", &hw_states_velocities_[2]);
  state_interfaces.emplace_back(rear_right_joint_name_, "position", &hw_states_positions_[3]);
  state_interfaces.emplace_back(rear_right_joint_name_, "velocity", &hw_states_velocities_[3]);

  // Export 4 sensors, each with 1 state (current)
  state_interfaces.emplace_back(front_left_sensor_name_, "current", &hw_states_currents_[0]);
  state_interfaces.emplace_back(rear_left_sensor_name_, "current", &hw_states_currents_[1]);
  state_interfaces.emplace_back(front_right_sensor_name_, "current", &hw_states_currents_[2]);
  state_interfaces.emplace_back(rear_right_sensor_name_, "current", &hw_states_currents_[3]);

  return state_interfaces;
}

std::vector<hardware_interface::CommandInterface> SaganControlHardwareInterface::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> command_interfaces;
  // Export 4 joints, each with 1 command (velocity)
  command_interfaces.emplace_back(front_left_joint_name_, "velocity", &hw_commands_velocities_[0]);
  command_interfaces.emplace_back(rear_left_joint_name_, "velocity", &hw_commands_velocities_[1]);
  command_interfaces.emplace_back(front_right_joint_name_, "velocity", &hw_commands_velocities_[2]);
  command_interfaces.emplace_back(rear_right_joint_name_, "velocity", &hw_commands_velocities_[3]);
  return command_interfaces;
}

hardware_interface::CallbackReturn SaganControlHardwareInterface::on_configure(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  // reset values always when configuring hardware
  for (uint i = 0; i < hw_states_positions_.size(); i++) {
    hw_states_positions_[i] = 0.0;
    hw_states_velocities_[i] = 0.0;
    hw_commands_velocities_[i] = 0.0;
    hw_states_currents_[i] = 0.0;
  }
  RCLCPP_INFO(get_logger(), "Successfully configured!");
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn SaganControlHardwareInterface::on_activate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  // command and state should be equal when starting
  for (uint i = 0; i < hw_commands_velocities_.size(); i++) {
    hw_commands_velocities_[i] = 0.0;
  }
  RCLCPP_INFO(get_logger(), "Successfully activated!");
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn SaganControlHardwareInterface::on_deactivate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  // Stop the robot
  for (uint i = 0; i < hw_commands_velocities_.size(); i++) {
    hw_commands_velocities_[i] = 0.0;
  }
  // Send the stop command to the hardware
  write(rclcpp::Time{}, rclcpp::Duration(0, 0));

  RCLCPP_INFO(get_logger(), "Successfully deactivated!");
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::return_type SaganControlHardwareInterface::read(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & period)
{
  uint8_t request_cmd = 0xB1; // Command to request sensor data
  SensorData left_data, right_data;

  // --- Read from Left Pico ---
  if (i2c_smbus_read_i2c_block_data(i2c_left_handle_, request_cmd, sizeof(SensorData), (uint8_t*)&left_data) < 0) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000, "Failed to read from left Pico");
      return hardware_interface::return_type::ERROR;
  }

  // --- Read from Right Pico ---
  if (i2c_smbus_read_i2c_block_data(i2c_right_handle_, request_cmd, sizeof(SensorData), (uint8_t*)&right_data) < 0) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000, "Failed to read from right Pico");
      return hardware_interface::return_type::ERROR;
  }

  // --- Process and Store Left Side Data ---
  double left_vel_front = (double)left_data.front_velocity / DATA_SCALE_FACTOR;
  double left_vel_rear = (double)left_data.rear_velocity / DATA_SCALE_FACTOR;
  
  hw_states_velocities_[0] = left_vel_front;
  hw_states_velocities_[1] = left_vel_rear;
  hw_states_positions_[0] += left_vel_front * period.seconds();
  hw_states_positions_[1] += left_vel_rear * period.seconds();
  hw_states_currents_[0] = (double)left_data.front_current / CURRENT_SCALE_FACTOR;
  hw_states_currents_[1] = (double)left_data.rear_current / CURRENT_SCALE_FACTOR;

  // --- Process and Store Right Side Data ---
  double right_vel_front = (double)right_data.front_velocity / DATA_SCALE_FACTOR;
  double right_vel_rear = (double)right_data.rear_velocity / DATA_SCALE_FACTOR;

  hw_states_velocities_[2] = right_vel_front;
  hw_states_velocities_[3] = right_vel_rear;
  hw_states_positions_[2] += right_vel_front * period.seconds();
  hw_states_positions_[3] += right_vel_rear * period.seconds();
  hw_states_currents_[2] = (double)right_data.front_current / CURRENT_SCALE_FACTOR;
  hw_states_currents_[3] = (double)right_data.rear_current / CURRENT_SCALE_FACTOR;

  return hardware_interface::return_type::OK;
}

hardware_interface::return_type SaganControlHardwareInterface::write(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  // --- Prepare Left Command ---
  ControlData left_cmd_data;
  left_cmd_data.cmd = 0xA1; // Set velocity command
  left_cmd_data.front_velocity = (int16_t)(hw_commands_velocities_[0] * DATA_SCALE_FACTOR);
  left_cmd_data.rear_velocity = (int16_t)(hw_commands_velocities_[1] * DATA_SCALE_FACTOR);

  // --- Prepare Right Command ---
  ControlData right_cmd_data;
  right_cmd_data.cmd = 0xA1;
  right_cmd_data.front_velocity = (int16_t)(hw_commands_velocities_[2] * DATA_SCALE_FACTOR);
  right_cmd_data.rear_velocity = (int16_t)(hw_commands_velocities_[3] * DATA_SCALE_FACTOR);

  // --- Send Left Command ---
  if (::write(i2c_left_handle_, &left_cmd_data, sizeof(ControlData)) != sizeof(ControlData)) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000, "Failed to write to left Pico");
      return hardware_interface::return_type::ERROR;
  }

  // --- Send Right Command ---
  if (::write(i2c_right_handle_, &right_cmd_data, sizeof(ControlData)) != sizeof(ControlData)) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000, "Failed to write to right Pico");
      return hardware_interface::return_type::ERROR;
  }

  return hardware_interface::return_type::OK;
}

hardware_interface::CallbackReturn SaganControlHardwareInterface::on_shutdown(
    const rclcpp_lifecycle::State & previous_state)
{
    RCLCPP_INFO(get_logger(), "Shutting down hardware...");
    if (i2c_left_handle_ > 0) {
        close(i2c_left_handle_);
    }
    if (i2c_right_handle_ > 0) {
        close(i2c_right_handle_);
    }
    return hardware_interface::CallbackReturn::SUCCESS;
}

}  // namespace sagan_control_hardware_interface

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(
  sagan_control_hardware_interface::SaganControlHardwareInterface, hardware_interface::SystemInterface)

