#include "sagan_control_hardware_interface/sagan_control_hardware_interface.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <vector>
#include <thread>

#include "hardware_interface/lexical_casts.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/rclcpp.hpp"

namespace sagan_control_hardware_interface
{

void SaganControlHardwareInterface::commands_callback(const sagan_interfaces::msg::SaganCmd::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(commands_mutex_);
  latest_commands_ = *msg;
  new_commands_available_ = true;
  
  // Extract wheel velocity commands (convert from rad/s to your hardware units if needed)
  for (int i = 0; i < 4; i++) {
    wheel_velocity_commands_[i] = msg->wheel_cmd[i].angular_velocity;
  }
  
  RCLCPP_DEBUG(get_logger(), "Received new commands from topic");
}

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

  // Create ROS2 node for topic communication
  node_ = std::make_shared<rclcpp::Node>("sagan_hardware_interface_node");

  // Initialize command storage
  wheel_velocity_commands_.resize(4, 0.0);
  steering_position_commands_.resize(4, 0.0);
  new_commands_available_ = false;

  // --- Get Parameters from URDF ---
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
      close(i2c_left_handle_);
      return hardware_interface::CallbackReturn::ERROR;
  }
  
  // --- Open I2C Bus for Right Pico ---
  i2c_right_handle_ = open(i2c_bus_path_.c_str(), O_RDWR);
  if (i2c_right_handle_ < 0) {
      RCLCPP_ERROR(get_logger(), "Failed to open I2C bus %s for right Pico", i2c_bus_path_.c_str());
      close(i2c_left_handle_);
      return hardware_interface::CallbackReturn::ERROR;
  }
  if (ioctl(i2c_right_handle_, I2C_SLAVE, right_pico_addr_) < 0) {
      RCLCPP_ERROR(get_logger(), "Failed to set I2C slave address 0x%X for right Pico", right_pico_addr_);
      close(i2c_left_handle_);
      close(i2c_right_handle_);
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
  LF_wheel_joint = info_.joints[0].name;
  LR_wheel_joint = info_.joints[1].name;
  RF_wheel_joint = info_.joints[2].name;
  RR_wheel_joint = info_.joints[3].name;

  front_left_sensor_name_ = info_.sensors[0].name;
  rear_left_sensor_name_ = info_.sensors[1].name;
  front_right_sensor_name_ = info_.sensors[2].name;
  rear_right_sensor_name_ = info_.sensors[3].name;

  // --- Validate joint interfaces ---
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
    if (joint.state_interfaces.size() != 2)
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

  RCLCPP_INFO(get_logger(), "Exporting state interfaces...");
  
  // Export 4 joints, each with 2 states (position, velocity)
  state_interfaces.emplace_back(LR_wheel_joint, hardware_interface::HW_IF_POSITION, &hw_states_positions_[0]);
  state_interfaces.emplace_back(LR_wheel_joint, hardware_interface::HW_IF_VELOCITY, &hw_states_velocities_[0]);
  state_interfaces.emplace_back(LF_wheel_joint, hardware_interface::HW_IF_POSITION, &hw_states_positions_[1]);
  state_interfaces.emplace_back(LF_wheel_joint, hardware_interface::HW_IF_VELOCITY, &hw_states_velocities_[1]);
  state_interfaces.emplace_back(RR_wheel_joint, hardware_interface::HW_IF_POSITION, &hw_states_positions_[2]);
  state_interfaces.emplace_back(RR_wheel_joint, hardware_interface::HW_IF_VELOCITY, &hw_states_velocities_[2]);
  state_interfaces.emplace_back(RF_wheel_joint, hardware_interface::HW_IF_POSITION, &hw_states_positions_[3]);
  state_interfaces.emplace_back(RF_wheel_joint, hardware_interface::HW_IF_VELOCITY, &hw_states_velocities_[3]);

  // Export 4 sensors, each with 1 state (current)
  state_interfaces.emplace_back(rear_left_sensor_name_, "current", &hw_states_currents_[0]);
  state_interfaces.emplace_back(front_left_sensor_name_, "current", &hw_states_currents_[1]);
  state_interfaces.emplace_back(rear_right_sensor_name_, "current", &hw_states_currents_[2]);
  state_interfaces.emplace_back(front_right_sensor_name_, "current", &hw_states_currents_[3]);

  RCLCPP_INFO(get_logger(), "Exported %zu state interfaces:", state_interfaces.size());
  for (const auto& interface : state_interfaces) {
    RCLCPP_INFO(get_logger(), "  - %s/%s", interface.get_name().c_str(), interface.get_interface_name().c_str());
  }

  return state_interfaces;
}

std::vector<hardware_interface::CommandInterface> SaganControlHardwareInterface::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> command_interfaces;
  
  RCLCPP_INFO(get_logger(), "Exporting command interfaces...");
  
  // Export 4 joints, each with 1 command (velocity)
  command_interfaces.emplace_back(RR_wheel_joint, hardware_interface::HW_IF_VELOCITY, &hw_commands_velocities_[0]);
  command_interfaces.emplace_back(RF_wheel_joint, hardware_interface::HW_IF_VELOCITY, &hw_commands_velocities_[1]);
  command_interfaces.emplace_back(LR_wheel_joint, hardware_interface::HW_IF_VELOCITY, &hw_commands_velocities_[2]);
  command_interfaces.emplace_back(LF_wheel_joint, hardware_interface::HW_IF_VELOCITY, &hw_commands_velocities_[3]);

  RCLCPP_INFO(get_logger(), "Exported %zu command interfaces:", command_interfaces.size());
  for (const auto& interface : command_interfaces) {
    RCLCPP_INFO(get_logger(), "  - %s/%s", interface.get_name().c_str(), interface.get_interface_name().c_str());
  }
  
  return command_interfaces;
}

hardware_interface::CallbackReturn SaganControlHardwareInterface::on_configure(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  // Create subscriber and publisher
  commands_subscriber_ = node_->create_subscription<sagan_interfaces::msg::SaganCmd>(
    "/SaganCommands", rclcpp::SystemDefaultsQoS(),
    std::bind(&SaganControlHardwareInterface::commands_callback, this, std::placeholders::_1));

  states_publisher_ = node_->create_publisher<sagan_interfaces::msg::SaganStates>(
    "/Sagan/SaganStates", rclcpp::SystemDefaultsQoS());

  // reset values always when configuring hardware
  for (uint i = 0; i < hw_states_positions_.size(); i++) {
    hw_states_positions_[i] = 0.0;
    hw_states_velocities_[i] = 0.0;
    hw_commands_velocities_[i] = 0.0;
    hw_states_currents_[i] = 0.0;
  }

  // Initialize states message
  //states_msg_.wheel_state.resize(4);
  //states_msg_.steering_state.resize(4);

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
  // Process ROS callbacks
  rclcpp::spin_some(node_);

  SensorData left_data, right_data;

  // Read from Left Pico - NO command byte in response
  int left_bytes = i2c_smbus_read_i2c_block_data(i2c_left_handle_, 0xB1, sizeof(SensorData), (uint8_t*)&left_data);
  if (left_bytes != sizeof(SensorData)) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000, 
                           "Failed to read from left Pico. Expected %zu bytes, got %d", 
                           sizeof(SensorData), left_bytes);
      return hardware_interface::return_type::ERROR;
  }

  // Read from Right Pico - NO command byte in response
  int right_bytes = i2c_smbus_read_i2c_block_data(i2c_right_handle_, 0xB1, sizeof(SensorData), (uint8_t*)&right_data);
  if (right_bytes != sizeof(SensorData)) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000, 
                           "Failed to read from right Pico. Expected %zu bytes, got %d", 
                           sizeof(SensorData), right_bytes);
      return hardware_interface::return_type::ERROR;
  }

  // --- Process and Store Left Side Data ---
  // Match Pico's DATA_SCALE_FACTOR (100.0)
  double left_vel_front = (double)left_data.front_velocity / DATA_SCALE_FACTOR;
  double left_vel_rear = (double)left_data.rear_velocity / DATA_SCALE_FACTOR;
  
  hw_states_velocities_[0] = left_vel_front;
  hw_states_velocities_[1] = left_vel_rear;
  hw_states_positions_[0] += left_vel_front * period.seconds();
  hw_states_positions_[1] += left_vel_rear * period.seconds();
  
  // Pico sends current * 1000, so divide by 1000 to get Amps
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

  // Update states message for publishing
  {
    std::lock_guard<std::mutex> lock(states_mutex_);
    for (int i = 0; i < 4; i++) {
      states_msg_.wheel_state[i].angular_velocity = hw_states_velocities_[i];
      // For real hardware, steering positions might come from different sensors
      // For now, you can set them to 0 or read from additional hardware
      states_msg_.steering_state[i].angular_position = 0.0; // Placeholder
    }
  }

  // Publish states
  states_publisher_->publish(states_msg_);

  RCLCPP_DEBUG(get_logger(), "Read data - Left: F%.3f/R%.3f, Right: F%.3f/R%.3f", 
               left_vel_front, left_vel_rear, right_vel_front, right_vel_rear);
  
  return hardware_interface::return_type::OK;
}

hardware_interface::return_type SaganControlHardwareInterface::write(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  // Check if we have new commands from the topic
  if (new_commands_available_) {
    std::lock_guard<std::mutex> lock(commands_mutex_);
    
    // Update hardware command interfaces with values from topic
    for (int i = 0; i < 4; i++) {
      hw_commands_velocities_[i] = wheel_velocity_commands_[i];
    }
    new_commands_available_ = false;
  }

  // --- Prepare Left Command ---
  ControlData left_cmd_data;
  left_cmd_data.cmd = 0xA1; // Set velocity command
  // Match Pico's DATA_SCALE_FACTOR (100.0)
  left_cmd_data.front_velocity = (int16_t)(hw_commands_velocities_[2] * DATA_SCALE_FACTOR);
  left_cmd_data.rear_velocity = (int16_t)(hw_commands_velocities_[0] * DATA_SCALE_FACTOR);

  // --- Prepare Right Command ---
  ControlData right_cmd_data;
  right_cmd_data.cmd = 0xA1;
  right_cmd_data.front_velocity = (int16_t)(hw_commands_velocities_[1] * DATA_SCALE_FACTOR);
  right_cmd_data.rear_velocity = (int16_t)(hw_commands_velocities_[3] * DATA_SCALE_FACTOR);

  // Use block write for better reliability
  if (i2c_smbus_write_i2c_block_data(i2c_left_handle_, 0xA1, sizeof(ControlData), (uint8_t*)&left_cmd_data) < 0) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000, "Failed to write to left Pico");
      return hardware_interface::return_type::ERROR;
  }

  if (i2c_smbus_write_i2c_block_data(i2c_right_handle_, 0xA1, sizeof(ControlData), (uint8_t*)&right_cmd_data) < 0) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000, "Failed to write to right Pico");
      return hardware_interface::return_type::ERROR;
  }

  RCLCPP_DEBUG(get_logger(), "Write commands - Left: F%.3f/R%.3f, Right: F%.3f/R%.3f", 
               hw_commands_velocities_[0], hw_commands_velocities_[1],
               hw_commands_velocities_[2], hw_commands_velocities_[3]);
  
  return hardware_interface::return_type::OK;
}

hardware_interface::CallbackReturn SaganControlHardwareInterface::on_shutdown(
    const rclcpp_lifecycle::State & /*previous_state*/)
{
    RCLCPP_INFO(get_logger(), "Shutting down hardware...");
    if (i2c_left_handle_ > 0) {
        close(i2c_left_handle_);
        i2c_left_handle_ = -1;
    }
    if (i2c_right_handle_ > 0) {
        close(i2c_right_handle_);
        i2c_right_handle_ = -1;
    }
    RCLCPP_INFO(get_logger(), "Hardware shutdown complete.");
    return hardware_interface::CallbackReturn::SUCCESS;
}

}  // namespace sagan_control_hardware_interface

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(
  sagan_control_hardware_interface::SaganControlHardwareInterface, hardware_interface::SystemInterface)