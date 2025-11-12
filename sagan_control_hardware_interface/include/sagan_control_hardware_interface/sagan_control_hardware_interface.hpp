#ifndef SAGAN_CONTROL_HARDWARE_INTERFACE__SAGAN_CONTROL_HARDWARE_INTERFACE_HPP_
#define SAGAN_CONTROL_HARDWARE_INTERFACE__SAGAN_CONTROL_HARDWARE_INTERFACE_HPP_

#include <string>
#include <vector>
#include <mutex>

#include "hardware_interface/handle.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/macros.hpp"
#include "rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp"
#include "rclcpp_lifecycle/state.hpp"

// Include your custom messages
#include "sagan_interfaces/msg/sagan_cmd.hpp"
#include "sagan_interfaces/msg/sagan_states.hpp"

// --- Linux I2C Libraries ---
#include <fcntl.h>       // For open()
#include <sys/ioctl.h>   // For ioctl()
#include <linux/i2c-dev.h> // For I2C_SLAVE
extern "C" {
  #include <i2c/smbus.h> // For i2c_smbus_... functions
}
#include <unistd.h>      // For close()

#pragma pack(push, 1)
// Match Pico structure (NO command byte)
struct SensorData {
    int16_t front_velocity;     // Actual velocity of motor 1
    int16_t front_current;      // Current draw of motor 1
    int16_t rear_velocity;      // Actual velocity of motor 2  
    int16_t rear_current;       // Current draw of motor 2
};

// Control data to send to Pico (WITH command byte)
struct ControlData {
    uint8_t cmd;               // Command (0xA1)
    int16_t front_velocity;    // Target velocity for motor 1
    int16_t rear_velocity;     // Target velocity for motor 2
};
#pragma pack(pop)

namespace sagan_control_hardware_interface
{
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
  // ROS2 Node for topic communication
  std::shared_ptr<rclcpp::Node> node_;
  
  // Subscriber and Publisher
  rclcpp::Subscription<sagan_interfaces::msg::SaganCmd>::SharedPtr commands_subscriber_;
  rclcpp::Publisher<sagan_interfaces::msg::SaganStates>::SharedPtr states_publisher_;
  
  // Message storage
  sagan_interfaces::msg::SaganStates states_msg_;
  sagan_interfaces::msg::SaganCmd latest_commands_;
  
  // Mutex for thread safety
  std::mutex commands_mutex_;
  std::mutex states_mutex_;
  
  // Command storage from topic
  std::vector<double> wheel_velocity_commands_;
  std::vector<double> steering_position_commands_;
  
  // Flag to indicate new commands received
  bool new_commands_available_;

  // Parameters
  std::string i2c_bus_path_;
  int left_pico_addr_;
  int right_pico_addr_;

  // I2C file descriptors
  int i2c_left_handle_{-1};
  int i2c_right_handle_{-1};

  // Storage for hardware interfaces
  std::vector<double> hw_commands_velocities_;
  std::vector<double> hw_states_positions_;
  std::vector<double> hw_states_velocities_;
  std::vector<double> hw_states_currents_;

  // Joint names
  std::string LF_wheel_joint;
  std::string LR_wheel_joint;
  std::string RF_wheel_joint;
  std::string RR_wheel_joint;

  // Sensor names
  std::string front_left_sensor_name_;
  std::string rear_left_sensor_name_;
  std::string front_right_sensor_name_;
  std::string rear_right_sensor_name_;

  // Constants - Match Pico scale factors
  static constexpr double DATA_SCALE_FACTOR = 100.0;     // Match Pico's DATA_SCALE_FACTOR
  static constexpr double CURRENT_SCALE_FACTOR = 1000.0; // Pico sends current * 1000

  // Callback for commands topic
  void commands_callback(const sagan_interfaces::msg::SaganCmd::SharedPtr msg);
};

}  // namespace sagan_control_hardware_interface

#endif  // SAGAN_CONTROL_HARDWARE_INTERFACE__SAGAN_CONTROL_HARDWARE_INTERFACE_HPP_