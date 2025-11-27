#include "mpc.hpp"
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include <nav_msgs/msg/path.hpp>

class ControlNode : public rclcpp::Node
{
public:
  ControlNode()
  : Node("control_node"),
    mpc_iter(0),
    dt(0.1),
    sim_time(15.0),
    N(65),
    z0({0.0, 0.0, 0.0}),
    zg({1.5, 1.5, 0.0}),
    zmin({-20.0, -20.0, -10e10}),
    zmax({ 20.0,  20.0,  10e10}),
    umin({ -0.7,  -1.91}),
    umax({  0.7,   1.91}),
    Q({{2.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 2.0}}),
    R({{0.5, 0.0}, {0.0, 0.5}}),
    mpc(N, dt,
        umin, umax, zmin, zmax,
        Q,
        R)
  {
    nz = zmin.size();
    nu = umin.size();

    path.setTrajectory(5, 10, dt, 5, {}, "infinite");

    mpc = MPC(N, dt,
              umin, umax, zmin, zmax,
              Q, R
    );
    
    mpc.Z_ref = path.getTrajectory();

    // mpc.setup_obstacles({0.74, 0.5, 0.05});
    // mpc.setup_obstacles({0.0, 5.0, 0.05});
    // mpc.setup_obstacles({2.0, 2.0, 0.1});
    // mpc.setup_obstacles({5.0, 5.0, 0.1});

    if (mpc.setup_constant_mpc())
      std::cout << "MPC setup successful!" << std::endl;
    else {
      std::cout << "MPC setup failed!" << std::endl;
      rclcpp::shutdown();
      return;
    }

    Z0 = std::vector<std::vector<double>>(N+1, z0);
    U0 = std::vector<std::vector<double>>(N, std::vector<double>(nu, 0.0));
    max_iter = mpc.Z_ref.size();
    mpc_output = std::vector<std::vector<double>>(max_iter, std::vector<double>(nz+nu, 0.0));

    odom_subscriber_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/odom/filtered", 10,
      std::bind(&ControlNode::OdomCallback, this, std::placeholders::_1));

    twist_stamped_publisher_ = this->create_publisher<geometry_msgs::msg::Twist>(
      "/cmd_vel", 10);

    predict_path_publisher_ = this->create_publisher<nav_msgs::msg::Path>(
      "/local_path", 10);

    global_path_publisher_ = this->create_publisher<nav_msgs::msg::Path>(
      "/global_path", 10);

    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(int(dt*1000)),
      std::bind(&ControlNode::ControlLoop, this));
  }

private:

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscriber_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr twist_stamped_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr predict_path_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr global_path_publisher_;
  rclcpp::TimerBase::SharedPtr timer_;

  geometry_msgs::msg::Twist twist;

  int mpc_iter;
  float dt;
  float sim_time;
  int N, nz, nu;
  float max_iter;

  std::vector<double> z0, zg, zmin, zmax, umin, umax;

  Trajectory path;
  MPC mpc;

  std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> res;
  std::vector<std::vector<double>> U0, Z0, mpc_output, Q, R;

  // **************************************************************************************************

  void OdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    tf2::Quaternion q(
        msg->pose.pose.orientation.x,
        msg->pose.pose.orientation.y,
        msg->pose.pose.orientation.z,
        msg->pose.pose.orientation.w
    );
    double roll, pitch, yaw;
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
    z0 = {msg->pose.pose.position.x, msg->pose.pose.position.y, yaw};

    // std::cout << "Iteration: " << mpc_iter << ", States: " << z0 << std::endl;
  }

  void ControlLoop()
  {

    if (mpc_iter < max_iter) {
      if (mpc.Z_ref[mpc_iter][2] - z0[2] > 3.141592654){
        z0[2] = z0[2] + 2*(3.141592654);
      }
    }

    auto t_start = std::chrono::steady_clock::now();
    if (mpc_iter > 0 && mpc_iter-1 < int(mpc_output.size())) {
      mpc_output[mpc_iter-1][0] = z0[0];
      mpc_output[mpc_iter-1][1] = z0[1];
      mpc_output[mpc_iter-1][2] = z0[2];
      mpc_output[mpc_iter-1][3] = U0[0][0];
      mpc_output[mpc_iter-1][4] = U0[0][1];
    }

    if(mpc_iter >= int(max_iter)){
      CloseControl();
      return;
    }

    // for (auto& z : Z0) z = z0;
    // for (auto& u : U0) u = std::vector<double>(nu, 0.0);
    Z0[0] = z0;
    // mpc.shift(dt, Z0, U0); 

    mpc.setup_variable_mpc(mpc_iter, z0, Z0, U0);
    res = mpc.solve_mpc();
    Z0  = res.first;
    U0  = res.second;

    Z0.erase(Z0.begin());
    Z0.push_back(Z0.back());
    U0.erase(U0.begin());
    U0.push_back(U0.back());

    nav_msgs::msg::Path             path_to_pub;
    geometry_msgs::msg::PoseStamped point_to_push;

    /*
      Publishing the predicted path
    */
    path_to_pub.poses.clear();
    path_to_pub.header.frame_id = "odom";
    path_to_pub.header.stamp    = this->get_clock()->now();

    for (auto z: Z0)
    {
      
      point_to_push.pose.position.x = z[0];
      point_to_push.pose.position.y = z[1];
      point_to_push.pose.position.z = 0.0;

      tf2::Quaternion q;
      q.setRPY(0, 0, z[2]);
      point_to_push.pose.orientation.x = q.x();
      point_to_push.pose.orientation.y = q.y();
      point_to_push.pose.orientation.z = q.z();
      point_to_push.pose.orientation.w = q.w();

      path_to_pub.poses.push_back(point_to_push);
    }
    predict_path_publisher_->publish(path_to_pub);

    /*
      Publishing the reference path
    */
    path_to_pub.poses.clear();
    path_to_pub.header.frame_id = "odom";
    path_to_pub.header.stamp    = this->get_clock()->now();

    for (auto ref: mpc.Z_ref)
    {
      
      point_to_push.pose.position.x = ref[0];
      point_to_push.pose.position.y = ref[1];
      point_to_push.pose.position.z = 0.0;

      tf2::Quaternion q;
      q.setRPY(0, 0, ref[2]);
      point_to_push.pose.orientation.x = q.x();
      point_to_push.pose.orientation.y = q.y();
      point_to_push.pose.orientation.z = q.z();
      point_to_push.pose.orientation.w = q.w();

      path_to_pub.poses.push_back(point_to_push);
    }
    global_path_publisher_->publish(path_to_pub);

    auto t_end = std::chrono::steady_clock::now();
    //std::cout << "Iteracion " << mpc_iter << ", TC: " << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count() << " msecs" << std::endl;
    twist.linear.x  = U0[0][0];
    twist.angular.z = U0[0][1];
    twist_stamped_publisher_->publish(twist);

    mpc_iter++;
  }

  void CloseControl()
  {

    twist.linear.x  = 0.0;
    twist.angular.z = 0.0;
    twist_stamped_publisher_->publish(twist);

    path.saveTrajectory(mpc_output, {}, {}, {}, {}, {}, "results.txt");
    path.saveTrajectory(mpc.Z_ref, {}, {}, {}, {}, {}, "reference.txt");

    std::ofstream outFile("obstacles.txt");
    outFile << "nobs cx cy r\n"; // Header
    for (int i = 0; i < mpc.n_obstacles; ++i)
    {
        outFile << i << " " << mpc.obstacles[i][0] << " " << mpc.obstacles[i][1] << " " << mpc.obstacles[i][2] + mpc.robot_radius + mpc.obstacle_tol << "\n";
    }
    outFile.close();

    rclcpp::shutdown();
  }

};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ControlNode>();
    rclcpp::spin(node);
    return 0;
}