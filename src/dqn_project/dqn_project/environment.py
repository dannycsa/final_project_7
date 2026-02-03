import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
import numpy as np
from typing import Tuple
import math
import time
import cv2
import os


class TurtleBot3Env(Node):
    """ROS2 Environment wrapper adapted for Stage Simulator + Odom Wrapper"""

    def __init__(self):
        super().__init__('turtlebot3_env')
        self.map_path = '/home/danny/project_rmov/src/stage_ros2/world/bitmaps/solid_cave.png'
        
        # Load as grayscale (0=Black/Obstacle, 255=White/Free)
        self.map_img = cv2.imread(self.map_path, cv2.IMREAD_GRAYSCALE)
        
        if self.map_img is None:
            self.get_logger().error(f"COULD NOT LOAD MAP FROM: {self.map_path}")
            # Fallback size if map fails to load (800x800 is standard for Stage)
            self.img_height, self.img_width = 800, 800 
        else:
            self.map_img = np.flipud(self.map_img)

            self.img_height, self.img_width = self.map_img.shape
            self.get_logger().info(f"Map loaded successfully: {self.img_width}x{self.img_height}")

        # Map physical size in meters (from your .world file)
        self.map_size_meters = 16.0 
        # Resolution (Pixels per Meter)
        self.resolution = self.img_width / self.map_size_meters

        self.spawn_x = -7.0
        self.spawn_y = -7.0
        

        # --- Publishers ---
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 100)
        
        # --- Subscribers ---
        # 1. Use /base_scan for Stage
        self.scan_sub = self.create_subscription(LaserScan, '/base_scan',
                                                  self.scan_callback, 100)
        
        # 2. CRITICAL CHANGE: Listen to the WRAPPER'S topic, not the simulator directly
        self.odom_sub = self.create_subscription(Odometry, '/odom/sim',
                                                  self.odom_callback, 100)

        # --- Services ---
        # 3. CRITICAL CHANGE: Call the WRAPPER'S service to reset everything
        self.reset_stage_client = self.create_client(Empty, '/reset_sim')

        # State variables
        self.scan_data = None
        self.position = (0.0, 0.0)
        self.yaw = 0.0
        self.last_position = (0.0, 0.0)

        # Goal position (will be randomized)
        self.goal_position = (4.0, 4.0)

        # Action space: 5 discrete actions
        self.actions = {
            0: (0.15, 0.0),    # Forward
            1: (0.0, 0.25),     # Rotate left
            2: (0.0, -0.25),    # Rotate right
            3: (0.08, 0.03),    # Forward + left
            4: (0.08, -0.03),   # Forward + right
        }

        self.collision_threshold = 0.2
        self.goal_threshold = 0.3

        self.goal_reached_flag = False

    def scan_callback(self, msg: LaserScan):
        """Store latest LiDAR scan"""
        self.scan_data = list(msg.ranges)

    def odom_callback(self, msg: Odometry):
        """Store latest odometry data (Already corrected by wrapper)"""
        self.position = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        )

        orientation_q = msg.pose.pose.orientation
        siny_cosp = 2 * (orientation_q.w * orientation_q.z +
                        orientation_q.x * orientation_q.y)
        cosy_cosp = 1 - 2 * (orientation_q.y * orientation_q.y +
                            orientation_q.z * orientation_q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

        if self.distance_to_goal() < self.goal_threshold:
            self.goal_reached_flag = True

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        linear_vel, angular_vel = self.actions[action]
        self.send_velocity(linear_vel, angular_vel)
        
        # Process messages
        start_time = self.get_clock().now()
        timeout = rclpy.duration.Duration(seconds=0.1)
        while (self.get_clock().now() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.01)
        
        done = False
        reward = 0.0
        
        if self.is_collision():
            reward = -100.0
            done = True
            self.get_logger().info("Collision detected!")
        # CRITICAL: Check both flag AND direct distance measurement
        elif self.goal_reached_flag or self.distance_to_goal() < self.goal_threshold:
            reward = 200.0
            done = True
            self.send_velocity(0.0, 0.0)
            self.goal_reached_flag = True  # Set it if distance check triggered
            self.get_logger().info("Goal reached!")
        else:
            reward = self.compute_reward(action)
        
        return self.get_state(), reward, done


    def compute_reward(self, action: int) -> float:
        current_dist = self.distance_to_goal()

        if hasattr(self, 'last_distance'):
            progress = self.last_distance - current_dist
            progress_reward = progress * 20.0
        else:
            progress_reward = 0.0

        self.last_distance = current_dist

        if self.scan_data:
            real_obstacles = [x for x in self.scan_data if x < 4.9]
            if real_obstacles:
                min_obstacle_dist = np.min(real_obstacles)
            else:
                min_obstacle_dist = 3.5
        else:
            min_obstacle_dist = 3.5

        if min_obstacle_dist < 0.5:
            obstacle_penalty = -5.0 * (0.5 - min_obstacle_dist)
        else:
            obstacle_penalty = 0.0

        action_penalty = -0.01 if action in [1, 2] else 0.0
        
        time_penalty = -0.25

        return progress_reward + obstacle_penalty + time_penalty + action_penalty

    def is_collision(self) -> bool:
        if self.scan_data is None:
            return False
        
        close_objects = [r for r in self.scan_data if r < 4.9]
        if not close_objects:
            return False
            
        min_distance = np.min(close_objects)
        return min_distance < self.collision_threshold

    def is_goal_reached(self) -> bool:
        return self.distance_to_goal() < self.goal_threshold

    def distance_to_goal(self) -> float:
        dx = self.goal_position[0] - self.position[0]
        dy = self.goal_position[1] - self.position[1]
        return math.sqrt(dx**2 + dy**2)

    def send_velocity(self, linear: float, angular: float):
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self.cmd_vel_pub.publish(twist)

    def is_valid_point(self, x: float, y: float) -> bool:
        """
        Check if a coordinate (meters) is in free space (White pixel).
        Stage Map Coordinates: Center is (0,0).
        Image Coordinates: Top-Left is (0,0).
        """
        if self.map_img is None:
            return True # Fallback if map didn't load

        # 1. Convert World (Meters) to Image (Pixels)
        # Shift origin from center (0,0) to top-left corner
        # X: -8 to +8  -> 0 to Width
        # Y: -8 to +8  -> Height to 0 (Image Y is inverted relative to World Y)
        
        pixel_x = int((x + self.map_size_meters / 2) * self.resolution)
        pixel_y = int((y + self.map_size_meters / 2) * self.resolution)

        # 2. Check Bounds
        if (pixel_x < 0 or pixel_x >= self.img_width or 
            pixel_y < 0 or pixel_y >= self.img_height):
            return False # Out of map bounds

        # 3. Check Pixel Color
        # We assume Free Space is White (approx 255) and Obstacles are Black (0)
        # Using a threshold of 127 is safe.
        pixel_value = self.map_img[pixel_y, pixel_x]
        
        return pixel_value > 127  # True if White (Safe), False if Black (Obstacle)
    
    def reset(self, random_goal: bool = True) -> np.ndarray:
        self.send_velocity(0.0, 0.0)

        # Reset via Wrapper
        self.reset_world()

        self.goal_reached_flag = False

        if random_goal:
            for _ in range(100):
                candidate_rel_x = np.random.uniform(0.0, 12.0)
                candidate_rel_y = np.random.uniform(0.0, 12.0)

                world_check_x = candidate_rel_x + self.spawn_x
                world_check_y = candidate_rel_y + self.spawn_y

                if self.is_valid_point(world_check_x, world_check_y):
                    self.goal_position = (candidate_rel_x, candidate_rel_y)
                    self.get_logger().info(f"Goal: ({candidate_rel_x:.1f}, {candidate_rel_y:.1f})")
                    break
            else:
                self.get_logger().warn("Goal Failed! Defaulting.")
                self.goal_position = (2.0, 2.0)

        for _ in range(10): 
            rclpy.spin_once(self, timeout_sec=0.02)

        self.last_distance = self.distance_to_goal()
        return self.get_state()

    def reset_world(self):
        # Calls the wrapper service, NOT stage directly
        if not self.reset_stage_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('/reset_sim service not available')
            return

        request = Empty.Request()
        future = self.reset_stage_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

    def get_state(self) -> np.ndarray:
        return None