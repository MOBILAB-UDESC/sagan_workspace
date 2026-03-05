import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import random
import csv

# --- 1. Load Data ---
# Certifique-se de que os arquivos existam no diretório
imu_df = pd.read_csv('imu_data_resampled.csv')
odom_df = pd.read_csv('odom_noise_data_resampled.csv') 
gt_df = pd.read_csv('odom_gt_data_resampled.csv')      

# --- 2. Pre-process Data ---
dt = 0.01  

def get_yaw(df):
    quats = df[['orientation_x', 'orientation_y', 'orientation_z', 'orientation_w']].to_numpy()
    return R.from_quat(quats).as_euler('xyz')[:, 2]

imu_yaw = get_yaw(imu_df)
gt_yaw = get_yaw(gt_df)

# Construct Measurement Arrays (Z)
# ODOM AGORA MEDE APENAS: [vx, vy, omega] (Removido o Yaw)
Z_odom = np.column_stack([
    odom_df['linear_velocity_x'].values,
    odom_df['linear_velocity_y'].values,
    odom_df['angular_velocity_z'].values
])

# IMU continua medindo: [ax, ay, theta, omega]
Z_imu = np.column_stack([
    imu_df['linear_acceleration_x'].values,
    imu_df['linear_acceleration_y'].values,
    imu_yaw,
    imu_df['angular_velocity_z'].values
])

GT_pos = gt_df[['position_x', 'position_y']].values

# --- 3. EKF Class ---
class EKF_8State:
    def __init__(self, Q_diag, R_odom_diag, R_imu_diag, dt=0.01):
        self.dt = dt
        # State: [x, y, vx, vy, ax, ay, theta, omega]
        self.x = np.zeros(8)
        self.P = np.eye(8) * 0.1
        
        self.Q = np.diag(Q_diag)
        self.R_odom = np.diag(R_odom_diag)
        self.R_imu = np.diag(R_imu_diag)

        # H_odom: [vx, vy, omega] -> Estados [2, 3, 7]
        self.H_odom = np.zeros((3, 8))
        self.H_odom[0, 2] = 1  # vx
        self.H_odom[1, 3] = 1  # vy
        self.H_odom[2, 7] = 1  # omega
        
        # H_imu: [ax, ay, theta, omega] -> Estados [4, 5, 6, 7]
        self.H_imu = np.zeros((4, 8))
        self.H_imu[0, 4] = 1  # ax
        self.H_imu[1, 5] = 1  # ay
        self.H_imu[2, 6] = 1  # theta (Yaw)
        self.H_imu[3, 7] = 1  # omega

    def predict(self):
        x, y, vx, vy, ax, ay, theta, omega = self.x
        dt = self.dt
        c, s = np.cos(theta), np.sin(theta)
        
        F = np.eye(8)
        F[0, 2] = c * dt; F[0, 3] = -s * dt
        F[1, 2] = s * dt; F[1, 3] = c * dt
        F[2, 4] = dt; F[3, 5] = dt
        F[6, 7] = dt
        
        nx = x + (vx*c - vy*s)*dt 
        ny = y + (vx*s + vy*c)*dt
        nvx = vx + ax*dt
        nvy = vy + ay*dt
        ntheta = theta + omega*dt
        
        self.x = np.array([nx, ny, nvx, nvy, ax, ay, ntheta, omega])
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z, H, R, is_imu=False):
        y = z - H @ self.x
        
        # Angle Wrapping apenas se for a medição da IMU (onde o theta está no índice 2)
        if is_imu:
            y[2] = (y[2] + np.pi) % (2 * np.pi) - np.pi
        
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ H) @ self.P

# --- 4. Simulation Runner ---
def run_simulation(params):
    params = np.abs(params)
    Q = params[0:8]
    R_odom = params[8:11] # 3 parâmetros agora
    R_imu = params[11:15]  # 4 parâmetros
    
    ekf = EKF_8State(Q, R_odom, R_imu, dt)
    ekf.x[0:2] = GT_pos[0]
    ekf.x[6] = gt_yaw[0]
    
    est_path = []
    for i in range(len(Z_odom)):
        ekf.predict()
        ekf.update(Z_odom[i], ekf.H_odom, ekf.R_odom, is_imu=False)
        ekf.update(Z_imu[i], ekf.H_imu, ekf.R_imu, is_imu=True)
        est_path.append(ekf.x[0:2])
        
    est_path = np.array(est_path)
    rmse = np.sqrt(np.mean(np.sum((est_path - GT_pos)**2, axis=1)))
    return rmse

# --- 5. Genetic Algorithm ---
POP_SIZE = 20
GENERATIONS = 150
GENE_COUNT = 15 
GENE_MAX_VALUE = 1000.0
GENE_MIN_VALUE = 0.001

population = np.random.uniform(0.1, 100, (POP_SIZE, GENE_COUNT))
history = []

for gen in range(GENERATIONS):
    fitness_scores = []
    
    for ind in population:
        loss = run_simulation(ind)
        fitness_scores.append(loss)
        history.append([gen, loss] + ind.tolist())
    
    # --- RE-ADDED STATS ---
    fitness_scores = np.array(fitness_scores)
    mean_score = np.mean(fitness_scores)
    median_score = np.median(fitness_scores)
    best_idx = np.argmin(fitness_scores)
    
    # Selection & Reproduction logic stays the same...
    new_pop = [population[best_idx]]
    while len(new_pop) < POP_SIZE:
        p1 = population[np.random.randint(POP_SIZE)]
        p2 = population[np.random.randint(POP_SIZE)]
        mask = np.random.rand(GENE_COUNT) > 0.5
        child = np.where(mask, p1, p2)
        
        if np.random.rand() < 0.3:
            child *= (1 + np.random.normal(0, 0.1, GENE_COUNT))
            child = np.clip(child, GENE_MIN_VALUE, GENE_MAX_VALUE)
        new_pop.append(child)
        
    population = np.array(new_pop)
    
    # Updated print statement
    print(f"Generation {gen} | Best RMSE: {fitness_scores[best_idx]:.4f}, Mean: {mean_score:.4f}, Median: {median_score:.4f}")

# --- 6. Save ---
cols = ['generation', 'rmse'] + [f'Q_{i}' for i in range(8)] + \
       [f'R_odom_{i}' for i in range(3)] + [f'R_imu_{i}' for i in range(4)]

df_res = pd.DataFrame(history, columns=cols)
df_res.to_csv('ga_tuning_results_no_odom_yaw.csv', index=False)