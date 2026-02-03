import numpy as np
from typing import Tuple

class StateProcessor:
    def __init__(self, n_lidar_bins: int = 48):
        self.n_lidar_bins = n_lidar_bins
        # In Stage 'cave' world, walls can be far. 
        # 3.5m is a standard "perception horizon" for the AI.
        self.max_lidar_range = 3.5  

    def process_lidar(self, scan_data: list) -> np.ndarray:
        scan_array = np.array(scan_data)

        # Handle infinite or NaN values from simulation
        scan_array[np.isinf(scan_array)] = self.max_lidar_range
        scan_array[np.isnan(scan_array)] = self.max_lidar_range

        # Clip and Normalize
        scan_array = np.clip(scan_array, 0, self.max_lidar_range)

        # Discretize 360 scan into 'n' bins (taking the min distance per bin)
        points_per_bin = len(scan_array) // self.n_lidar_bins
        binned_scan = []

        for i in range(self.n_lidar_bins):
            start_idx = i * points_per_bin
            end_idx = (i + 1) * points_per_bin if i < self.n_lidar_bins - 1 else len(scan_array)
            
            # Use safe min finding (handle empty slices if necessary)
            if end_idx > start_idx:
                bin_min = np.min(scan_array[start_idx:end_idx])
            else:
                bin_min = self.max_lidar_range
            binned_scan.append(bin_min)

        return np.array(binned_scan) / self.max_lidar_range

    def compute_goal_info(self, 
                          current_pos: Tuple[float, float], 
                          goal_pos: Tuple[float, float], 
                          current_yaw: float) -> np.ndarray:
        
        dx = goal_pos[0] - current_pos[0]
        dy = goal_pos[1] - current_pos[1]

        distance = np.sqrt(dx**2 + dy**2)

        # Calculate angle to goal relative to robot's current heading
        angle_to_goal = np.arctan2(dy, dx)
        relative_angle = angle_to_goal - current_yaw

        # Normalize angle to [-pi, pi]
        relative_angle = np.arctan2(np.sin(relative_angle), np.cos(relative_angle))

        # Normalize features for Neural Network [0, 1] and [-1, 1]
        distance_norm = np.clip(distance / 25.0, 0, 1) 
        angle_norm = relative_angle / np.pi 

        return np.array([distance_norm, angle_norm])

    def get_state(self, scan_data: list, current_pos: Tuple, goal_pos: Tuple, current_yaw: float) -> np.ndarray:
        if scan_data is None:
            return np.zeros(self.n_lidar_bins + 2)
            
        lidar_state = self.process_lidar(scan_data)
        goal_state = self.compute_goal_info(current_pos, goal_pos, current_yaw)

        return np.concatenate([lidar_state, goal_state])