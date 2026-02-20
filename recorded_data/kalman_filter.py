import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# 1. Load the Data
imu_df = pd.read_csv('imu_data_resampled.csv')
odom_df = pd.read_csv('odom_noise_data_resampled.csv')
gt_df = pd.read_csv('odom_gt_data_resampled.csv')

# 2. Helper Functions & Pre-processing
dt = 0.01  

def get_yaw(df):
    """Extracts Yaw from quaternion columns."""
    quats = df[['orientation_x', 'orientation_y', 'orientation_z', 'orientation_w']].to_numpy()
    return R.from_quat(quats).as_euler('xyz')[:, 2]

imu_yaw = get_yaw(imu_df)
gt_yaw = get_yaw(gt_df)

# Prepare Measurement Vectors
# Odom AGORA: [vx, vy, omega] -> Removido o Yaw (theta)
Z_odom = np.column_stack([
    odom_df['linear_velocity_x'].values,
    odom_df['linear_velocity_y'].values,
    odom_df['angular_velocity_z'].values
])

# IMU continua: [ax, ay, theta, omega]
Z_imu = np.column_stack([
    imu_df['linear_acceleration_x'].values,
    imu_df['linear_acceleration_y'].values,
    imu_yaw,
    imu_df['angular_velocity_z'].values
])

GT_pos = gt_df[['position_x', 'position_y']].values

# 3. EKF Class Definition
class EKF_8State:
    def __init__(self, Q_diag, R_odom_diag, R_imu_diag, dt=0.01):
        self.dt = dt
        # State: [x, y, vx, vy, ax, ay, theta, omega]
        self.x = np.zeros(8)
        self.P = np.eye(8) * 0.1
        
        self.Q = np.diag(Q_diag)
        self.R_odom = np.diag(R_odom_diag) # Agora espera 3 valores
        self.R_imu = np.diag(R_imu_diag)   # Espera 4 valores

        # H_odom: [vx, vy, omega] -> Estados [2, 3, 7]
        self.H_odom = np.zeros((3, 8))
        self.H_odom[0, 2] = 1; self.H_odom[1, 3] = 1; self.H_odom[2, 7] = 1
        
        # H_imu: [ax, ay, theta, omega] -> Estados [4, 5, 6, 7]
        self.H_imu = np.zeros((4, 8))
        self.H_imu[0, 4] = 1; self.H_imu[1, 5] = 1; self.H_imu[2, 6] = 1; self.H_imu[3, 7] = 1

    def predict(self):
        x, y, vx, vy, ax, ay, theta, omega = self.x
        dt = self.dt
        c, s = np.cos(theta), np.sin(theta)
        
        F = np.eye(8)
        F[0, 2] = c * dt; F[0, 3] = -s * dt
        F[1, 2] = s * dt; F[1, 3] = c * dt
        F[2, 4] = dt; F[3, 5] = dt
        F[6, 7] = dt
        F[0, 6] = (-vx*s - vy*c)*dt
        F[1, 6] = (vx*c - vy*s)*dt

        nx = x + (vx*c - vy*s)*dt 
        ny = y + (vx*s + vy*c)*dt
        nvx = vx + ax*dt
        nvy = vy + ay*dt
        ntheta = theta + omega*dt
        
        self.x = np.array([nx, ny, nvx, nvy, ax, ay, ntheta, omega])
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z, H, R, is_imu=False):
        """Measurement Update with selective angle wrapping"""
        y = z - H @ self.x
        
        # Angle Wrapping apenas para medição de ângulo (IMU index 2)
        if is_imu:
            y[2] = (y[2] + np.pi) % (2 * np.pi) - np.pi
        
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ H) @ self.P

# 4. Filter Execution
# --- PARÂMETROS AJUSTADOS ---
Q_diag = [350.09762308907625,27.586075296707087,22.59600966831785,59.51548030795145,53.83230189591617,22.21231194450657,0.622841947995474,75.91357155223278]
# R_odom agora tem apenas 3 valores (vx, vy, omega)
R_odom_diag = [53.27434573744027,19.927294634072158,1.5949841871476442] 
# R_imu continua com 4 (ax, ay, theta, omega)
R_imu_diag = [39.37035896700368,67.59456602814052,448.4812553444012,92.13887459979527]

ekf = EKF_8State(Q_diag, R_odom_diag, R_imu_diag, dt)

ekf.x[0:2] = GT_pos[0]
ekf.x[6] = gt_yaw[0]

est_path = []

for i in range(len(Z_odom)):
    ekf.predict()
    # Note que agora passamos se é IMU ou não para o angle wrapping
    ekf.update(Z_odom[i], ekf.H_odom, ekf.R_odom, is_imu=False)
    ekf.update(Z_imu[i], ekf.H_imu, ekf.R_imu, is_imu=True)
    est_path.append(ekf.x[0:2])

est_path = np.array(est_path)

# 5. Evaluation
rmse = np.sqrt(np.mean(np.sum((est_path - GT_pos)**2, axis=1)))
odom_pos = odom_df[['position_x', 'position_y']].values
odom_rmse = np.sqrt(np.mean(np.sum((odom_pos - GT_pos)**2, axis=1)))

print(f"Noisy Odom RMSE: {odom_rmse:.4f} meters")
print(f"EKF RMSE:        {rmse:.4f} meters")

# 6. Plotting
plt.figure(figsize=(10, 8))
plt.plot(GT_pos[:,0], GT_pos[:,1], 'k-', label='Ground Truth', linewidth=2)
plt.plot(odom_df['position_x'], odom_df['position_y'], 'r:', 
         label=f'Noisy Odom (RMSE={odom_rmse:.4f}m)', alpha=0.5)
plt.plot(est_path[:,0], est_path[:,1], 'b--', 
         label=f'EKF (RMSE={rmse:.4f}m)', linewidth=2)

plt.title("EKF Trajectory vs Noisy Odometry (Sem Yaw na Odom)")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()