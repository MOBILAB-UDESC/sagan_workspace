#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.parameter import Parameter
from rclpy.parameter_client import AsyncParameterClient
from ros_gz_interfaces.srv import SetEntityPose
from ros_gz_interfaces.msg import Entity
from std_srvs.srv import Trigger
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
import math
import numpy as np
import pandas as pd
import time
import threading

# Import your action
from sagan_interfaces.action import FollowPath


class SaganKalmanFilterOptimizer(Node):
    def __init__(self):
        super().__init__('sagan_kalman_filter_optimizer')

        # --- GA Parameters (Tune these) ---
        self.population_size = 20
        self.num_generations = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.elitism_count = 1  # Number of best individuals to carry over
        self.gene_min_val = 0.001  # Min value for any covariance element
        self.gene_max_val = 1000.0  # Max value for any covariance element

        # --- Dynamic Configuration Variables ---
        self.num_genes = None
        self.q_size = 0
        self.odom_r_size = 0
        self.imu_r_size = 0

        # --- ROS 2 Clients & Publishers ---
        self.param_client = AsyncParameterClient(self, "/sagan_kalman_filter")
        self.reset_gz_pose_client = self.create_client(SetEntityPose, '/world/default/set_pose')
        self.reset_odom_client = self.create_client(Trigger, 'reset_odometry')
        self.reset_ekf_client = self.create_client(Trigger, 'reset_ekf')
        
        # Action client for path following
        self.action_client = ActionClient(self, FollowPath, 'follow_path')
        
        # --- ROS 2 Subscribers for Fitness Evaluation ---
        self.ground_truth_topic = "/odom_gz"
        self.ekf_topic = "/odom/filtered"
        self.noisy_odom_topic = "/odom/with_noise"

        # Data lists now store (timestamp, pose) tuples
        self.ground_truth_data = []
        self.ekf_data = []
        self.noisy_data = []

        self.ground_truth_sub = self.create_subscription(Odometry, self.ground_truth_topic, self.ground_truth_callback, 10)
        self.ekf_sub = self.create_subscription(Odometry, self.ekf_topic, self.ekf_callback, 10)
        self.noisy_sub = self.create_subscription(Odometry, self.noisy_odom_topic, self.noisy_callback, 10)

        # Action execution tracking
        self.action_completed_event = threading.Event()
        self.action_success = False
        self.action_feedback = None
        self.max_action_time = 120.0  # Maximum time to complete path (seconds)

        self.get_logger().info("Kalman Filter Optimizer with Action Server is ready.")

    # --- Subscriber Callbacks ---
    def ground_truth_callback(self, msg):
        time_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.ground_truth_data.append((time_sec, msg.pose.pose))

    def ekf_callback(self, msg):
        time_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.ekf_data.append((time_sec, msg.pose.pose))

    def noisy_callback(self, msg):
        time_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.noisy_data.append((time_sec, msg.pose.pose))

    # --- Action Client Methods ---
    def wait_for_action_server(self, timeout_sec=10.0):
        """Wait for the action server to be available"""
        self.get_logger().info("Waiting for action server...")
        if not self.action_client.wait_for_server(timeout_sec=timeout_sec):
            self.get_logger().error(f"Action server not available after {timeout_sec} seconds")
            return False
        self.get_logger().info("Action server connected!")
        return True

    def execute_path_via_action(self, path_msg):
        """Execute path using the action server - FIXED VERSION"""
        # Reset completion event
        self.action_completed_event.clear()
        self.action_success = False
        self.action_feedback = None

        # Send goal
        goal_msg = FollowPath.Goal()
        goal_msg.path = path_msg

        #self.get_logger().info("Sending path to action server...")
        
        # Send goal with feedback callback
        send_goal_future = self.action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.action_feedback_callback
        )
        
        # Wait for goal to be accepted
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()
        
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected by action server")
            return False

        #self.get_logger().info("Goal accepted, waiting for completion...")
        
        # Get result future and add callback
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.action_result_callback)
        
        # Wait for completion or timeout
        start_time = time.time()
        while rclpy.ok():
            # Check if action completed
            if self.action_completed_event.wait(0.1):  # Wait with timeout
                break
            
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > self.max_action_time:
                self.get_logger().warn(f"Action timed out after {elapsed:.1f}s, cancelling...")
                cancel_future = goal_handle.cancel_goal_async()
                rclpy.spin_until_future_complete(self, cancel_future)
                return False
            
            # Process callbacks
            rclpy.spin_once(self, timeout_sec=0)
        
        return self.action_success

    def action_feedback_callback(self, feedback_msg):
        """Callback for action feedback"""
        self.action_feedback = feedback_msg.feedback
        self.get_logger().debug(f"Action progress: {self.action_feedback.current_waypoint}/{self.action_feedback.total_waypoints}")

    # Replace lines 144-156 (the action_result_callback method)
    def action_result_callback(self, future):
        """Callback for action result - FIXED"""
        try:
            result = future.result()
            self.action_success = result.result.success
            if result.result.success:
                self.get_logger().debug("Action completed successfully!")
            else:
                self.get_logger().warn(f"Action failed")
        except Exception as e:
            self.get_logger().error(f"Exception in action result: {str(e)}")
            self.action_success = False
        finally:
            # Signal completion
            self.action_completed_event.set()

    def setup_optimizer_sync(self):
        self.get_logger().info("Waiting for EKF parameter service...")
        if not self.param_client.wait_for_services(timeout_sec=5.0):
            self.get_logger().error("Parameter service '/sagan_kalman_filter/get_parameters' not available. Exiting.")
            return False

        self.get_logger().info("Fetching EKF configuration to adapt optimizer...")
        params_to_get = ['Q_diag', 'odom_sensor.measurement_map', 'imu_sensor.measurement_map']
        future = self.param_client.get_parameters(params_to_get)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()

        if response is None:
            self.get_logger().error("Failed to get EKF parameters. Is the filter node running?")
            return False

        param_dict = {}
        for name, value_msg in zip(params_to_get, response.values):
            if value_msg.type == Parameter.Type.DOUBLE_ARRAY.value:
                param_dict[name] = list(value_msg.double_array_value)
            elif value_msg.type == Parameter.Type.BOOL_ARRAY.value:
                param_dict[name] = list(value_msg.bool_array_value)
        
        for p in params_to_get:
            if p not in param_dict:
                self.get_logger().error(f"Parameter '{p}' not found. Check EKF yaml.")
                return False

        self.q_size = len(param_dict['Q_diag'])
        self.odom_r_size = sum(param_dict['odom_sensor.measurement_map'])
        self.imu_r_size = sum(param_dict['imu_sensor.measurement_map'])
        self.num_genes = self.q_size + self.odom_r_size + self.imu_r_size

        self.get_logger().info("Optimizer configured dynamically:")
        self.get_logger().info(f"  -> Total genes to optimize: {self.num_genes}")
        return True

    def align_and_get_poses(self):
        """
        Improved version with time windowing and interpolation
        """
        if not self.ground_truth_data or not self.ekf_data or not self.noisy_data:
            return None, None, None
        
        # Find common time window
        gt_times = np.array([t for t, p in self.ground_truth_data])
        ekf_times = np.array([t for t, p in self.ekf_data])
        noisy_times = np.array([t for t, p in self.noisy_data])
        
        min_time = max(gt_times.min(), ekf_times.min(), noisy_times.min())
        max_time = min(gt_times.max(), ekf_times.max(), noisy_times.max())
        
        if max_time - min_time < 1.0:  # Less than 1 second of common data
            self.get_logger().warn(f"Insufficient time overlap: {max_time-min_time:.2f}s")
            return None, None, None
        
        # Create uniform time vector for interpolation
        uniform_times = np.linspace(min_time, max_time, num=2000)  # 2000 uniform points
        
        # Interpolate positions
        gt_interp = self.interpolate_poses(uniform_times, self.ground_truth_data)
        ekf_interp = self.interpolate_poses(uniform_times, self.ekf_data)
        noisy_interp = self.interpolate_poses(uniform_times, self.noisy_data)
    
        return gt_interp, ekf_interp, noisy_interp

    def interpolate_poses(self, target_times, data):
        """Interpolate poses to target times"""
        times = np.array([t for t, p in data])
        x_vals = np.array([p.position.x for t, p in data])
        y_vals = np.array([p.position.y for t, p in data])
        
        # Linear interpolation
        from scipy import interpolate
        fx = interpolate.interp1d(times, x_vals, kind='linear', fill_value='extrapolate')
        fy = interpolate.interp1d(times, y_vals, kind='linear', fill_value='extrapolate')
        
        return np.column_stack([fx(target_times), fy(target_times)])

    def calculate_global_error_metrics(self, gt, ekf, noisy):
        """
        Calculate comprehensive error metrics including:
        - Global position error
        - Error correction effectiveness
        - Consistency over time
        """
        if gt is None or len(gt) < 10:
            return 0.0, 0.0, 0.0

        # 1. Global Position Error (RMSE)
        position_errors = np.sqrt(np.sum((gt - ekf)**2, axis=1))
        global_rmse = np.sqrt(np.mean(position_errors**2))
        
        # 2. Error Correction Effectiveness
        noisy_errors = np.sqrt(np.sum((gt - noisy)**2, axis=1))
        ekf_errors = position_errors
        
        # Calculate improvement ratio (how much EKF reduces error)
        improvement_ratio = np.mean(noisy_errors / (ekf_errors + 1e-6))
        
        # 3. Consistency and Smoothness
        # Check if EKF output is smoother than noisy data
        ekf_derivative = np.diff(ekf, axis=0)
        noisy_derivative = np.diff(noisy, axis=0)
        
        ekf_smoothness = np.mean(np.abs(ekf_derivative))
        noisy_smoothness = np.mean(np.abs(noisy_derivative))
        
        smoothness_improvement = noisy_smoothness / (ekf_smoothness + 1e-6)
        
        return global_rmse, improvement_ratio, smoothness_improvement

    # --- Core GA Functions ---
    def run_optimizer(self):
        if not self.setup_optimizer_sync():
            self.get_logger().error("Optimizer setup failed. Shutting down.")
            return

        if not self.wait_for_action_server():
            self.get_logger().error("Action server not available. Shutting down.")
            return

        self.get_logger().info("Starting Genetic Algorithm with Action Server...")
        population = self.initialize_population()
        best_overall_individual = None
        best_overall_fitness = -1

        for generation in range(self.num_generations):
            self.get_logger().info(f"--- Generation {generation + 1}/{self.num_generations} ---")

            fitness_scores = np.array([self.evaluate_fitness(ind) for ind in population])

            best_gen_idx = np.argmax(fitness_scores)
            if fitness_scores[best_gen_idx] > best_overall_fitness:
                best_overall_fitness = fitness_scores[best_gen_idx]
                best_overall_individual = population[best_gen_idx].copy()
                self.get_logger().info(f"!!! New Best Overall Fitness Found: {best_overall_fitness:.4f} !!!")
                self.log_individual(best_overall_individual)
            else:
                self.get_logger().info(f"Best Generation Fitness: {fitness_scores[best_gen_idx]:.4f}")
                self.log_individual(population[best_gen_idx])

            parents = self.selection(population, fitness_scores)
            
            next_population = []
            elite_indices = np.argsort(fitness_scores)[-self.elitism_count:]
            for i in elite_indices:
                next_population.append(population[i].copy())

            while len(next_population) < self.population_size:
                parent1, parent2 = parents[np.random.randint(0, len(parents))], parents[np.random.randint(0, len(parents))]
                offspring = self.crossover(parent1, parent2) if np.random.rand() < self.crossover_rate else parent1.copy()
                next_population.append(self.mutate(offspring))

            population = np.array(next_population)

        self.get_logger().info("--- Genetic Algorithm Finished ---")
        self.get_logger().info("Optimal Covariances Found:")
        self.log_individual(best_overall_individual)

    def evaluate_fitness(self, individual):
        self.set_kalman_params(individual)
        time.sleep(0.2)

        self.reset_simulation_state()
        self.ground_truth_data.clear()
        self.ekf_data.clear()
        self.noisy_data.clear()
        time.sleep(0.2)

        # Create and execute path via action
        path_msg = self.create_test_path()
        success = self.execute_path_via_action(path_msg)

        # self.get_logger().info("Waiting for final data points...")
        # time.sleep(3.0)  # Wait 3 seconds

        # # Diagnostic logging
        # self.get_logger().info(f"Data points collected:")
        # self.get_logger().info(f"  Ground Truth: {len(self.ground_truth_data)}")
        # self.get_logger().info(f"  EKF: {len(self.ekf_data)}")
        # self.get_logger().info(f"  Noisy: {len(self.noisy_data)}")

        # df = pd.DataFrame({
        #     'gazebo': self.ground_truth_data,
        #     'ekf': self.ekf_data,
        #     'odom_noisy': self.noisy_data
        # })

        # df.to_csv('data.csv', index=False)

        if not success:
            self.get_logger().warn("Path execution failed, returning low fitness")
            return 0.1  # Low but non-zero fitness for failed executions

        gt, ekf, noisy = self.align_and_get_poses()

        if gt is None or len(gt) < 10:
            self.get_logger().error("Insufficient data for evaluation, returning fitness of 0.")
            return 0.0
        
        # Calculate comprehensive error metrics
        global_rmse, improvement_ratio, smoothness_improvement = self.calculate_global_error_metrics(gt, ekf, noisy)
        
        # Enhanced fitness function focusing on global error and error correction
        if global_rmse < 1e-6:
            global_rmse = 1e-6
            
        # Primary component: inverse of global error (minimize global error)
        accuracy_component = 1.0 / (1.0 + global_rmse)
        
        # Secondary component: error correction effectiveness
        correction_component = min(improvement_ratio, 10.0)  # Cap at 10x improvement
        
        # Tertiary component: smoothness improvement
        smoothness_component = min(smoothness_improvement, 5.0)  # Cap at 5x smoother
        
        # Combined fitness with weights
        fitness = (0.6 * accuracy_component + 
                  0.3 * correction_component + 
                  0.1 * smoothness_component)

        self.get_logger().info(
            f"Points: {len(gt)}, Global RMSE: {global_rmse:.4f}, "
            f"Improvement: {improvement_ratio:.2f}x, Smoothness: {smoothness_improvement:.2f}x, "
            f"Fitness: {fitness:.4f}"
        )
        return fitness

    def create_test_path(self):
        """Create a lemniscate (figure-eight) path rotated by 135° and starting at origin"""
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "odom"
        
        # Lemniscate parameters
        a = 5.0  # Size parameter
        num_points = 2000  # Number of waypoints
        
        # Start at t = π/2 to ensure we begin at origin
        # For Bernoulli's lemniscate: x = a*cos(t)/(1+sin²(t)), y = a*sin(t)*cos(t)/(1+sin²(t))
        # At t = π/2: cos(π/2) = 0, sin(π/2) = 1 → x = 0, y = 0
        t_start = math.pi / 2  # Start at π/2 to begin at origin
        t_end = math.pi / 2 + 2.0 * math.pi  # One full cycle (2π) from starting point
        
        # Rotation angle in radians (135°)
        rotation_angle = 135.0 * math.pi / 180.0
        cos_theta = math.cos(rotation_angle)
        sin_theta = math.sin(rotation_angle)
        
        # Generate lemniscate waypoints
        for i in range(num_points + 1):
            # Parameter t from t_start to t_end
            t = t_start + (t_end - t_start) * i / num_points
            
            # Parametric equations for lemniscate (Bernoulli's lemniscate)
            denominator = 1.0 + math.sin(t)**2
            x_original = a * math.cos(t) / denominator
            y_original = a * math.sin(t) * math.cos(t) / denominator
            
            # Apply 135° rotation
            x = x_original * cos_theta - y_original * sin_theta
            y = x_original * sin_theta + y_original * cos_theta
            
            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = self.get_clock().now().to_msg()
            pose_stamped.header.frame_id = "odom"
            pose_stamped.pose.position.x = x
            pose_stamped.pose.position.y = y
            pose_stamped.pose.position.z = 0.0
            
            # Calculate orientation (tangent to the path)
            # Use next point for calculating direction
            if i < num_points:
                next_t = t_start + (t_end - t_start) * (i + 1) / num_points
            else:
                # For the last point, use first point (closed loop)
                next_t = t_start
            
            # Calculate next point
            next_denominator = 1.0 + math.sin(next_t)**2
            next_x_original = a * math.cos(next_t) / next_denominator
            next_y_original = a * math.sin(next_t) * math.cos(next_t) / next_denominator
            
            # Apply same rotation to next point
            next_x = next_x_original * cos_theta - next_y_original * sin_theta
            next_y = next_x_original * sin_theta + next_y_original * cos_theta
            
            # Calculate yaw angle (tangent to the rotated curve)
            yaw = math.atan2(next_y - y, next_x - x)
            
            # Convert yaw to quaternion
            pose_stamped.pose.orientation.x = 0.0
            pose_stamped.pose.orientation.y = 0.0
            pose_stamped.pose.orientation.z = math.sin(yaw / 2.0)
            pose_stamped.pose.orientation.w = math.cos(yaw / 2.0)
            
            path_msg.poses.append(pose_stamped)
    
        # Force the first point to be exactly at origin
        if len(path_msg.poses) > 0:
            path_msg.poses[0].pose.position.x = 0.0
            path_msg.poses[0].pose.position.y = 0.0
    
        #self.get_logger().info(f"Created 135° rotated lemniscate path with {len(path_msg.poses)} waypoints")
        return path_msg

    # --- GA Operators ---
    def initialize_population(self):
        return np.random.uniform(low=self.gene_min_val, high=self.gene_max_val, size=(self.population_size, self.num_genes))

    def selection(self, population, fitness_scores):
        selected = []
        for _ in range(self.population_size):
            i1, i2 = np.random.choice(len(population), 2, replace=False)
            winner = i1 if fitness_scores[i1] > fitness_scores[i2] else i2
            selected.append(population[winner])
        return selected

    def crossover(self, parent1, parent2):
        if self.num_genes < 2: return parent1.copy()
        point = np.random.randint(1, self.num_genes)
        return np.concatenate([parent1[:point], parent2[point:]])

    def mutate(self, individual):
        for i in range(self.num_genes):
            if np.random.rand() < self.mutation_rate:
                change = np.random.uniform(-0.5, 0.5) * individual[i] 
                individual[i] = np.clip(individual[i] + change, self.gene_min_val, self.gene_max_val)
        return individual

    # --- Helper Functions ---
    def set_kalman_params(self, individual):
        end_q = self.q_size
        end_odom_r = end_q + self.odom_r_size
        
        params_to_set = [
            Parameter('Q_diag', Parameter.Type.DOUBLE_ARRAY, individual[0:end_q].tolist()),
            Parameter('odom_sensor.R_diag', Parameter.Type.DOUBLE_ARRAY, individual[end_q:end_odom_r].tolist()),
            Parameter('imu_sensor.R_diag', Parameter.Type.DOUBLE_ARRAY, individual[end_odom_r:].tolist())
        ]
        future = self.param_client.set_parameters(params_to_set)
        rclpy.spin_until_future_complete(self, future)

    def reset_simulation_state(self):
        """Resets simulation state and waits for services to complete."""
        gz_req = SetEntityPose.Request()
        gz_req.entity.name = "Sagan"; gz_req.entity.type = Entity.MODEL
        gz_req.pose.position = Point(x=0.0, y=0.0, z=0.2)
        gz_req.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        future_gz = self.reset_gz_pose_client.call_async(gz_req)
        rclpy.spin_until_future_complete(self, future_gz)

        reset_req = Trigger.Request()
        future_odom = self.reset_odom_client.call_async(reset_req)
        rclpy.spin_until_future_complete(self, future_odom)
        future_ekf = self.reset_ekf_client.call_async(reset_req)
        rclpy.spin_until_future_complete(self, future_ekf)

    def log_individual(self, individual):
        end_q = self.q_size
        end_odom_r = end_q + self.odom_r_size
        q = np.round(individual[0:end_q], 4).tolist()
        r_o = np.round(individual[end_q:end_odom_r], 4).tolist()
        r_i = np.round(individual[end_odom_r:], 4).tolist()
        self.get_logger().info(f"    Q_diag: {q}")
        self.get_logger().info(f"    odom_sensor.R_diag: {r_o}")
        self.get_logger().info(f"    imu_sensor.R_diag: {r_i}")

def main(args=None):
    rclpy.init(args=args)
    optimizer_node = SaganKalmanFilterOptimizer()
    optimizer_node.run_optimizer()
    optimizer_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()