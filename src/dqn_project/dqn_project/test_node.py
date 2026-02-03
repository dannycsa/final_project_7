#!/usr/bin/env python3

import rclpy
import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from rclpy.node import Node
from dqn_project.dqn_agent import DQNAgent
from dqn_project.environment import TurtleBot3Env
from dqn_project.state_processor import StateProcessor

class DQNTestNode(Node):
    def __init__(self, model_path: str):
        super().__init__('dqn_test_node')

        self.state_size = 50        
        self.action_size = 5
        self.n_lidar_bins = 48      
        
        self.agent = DQNAgent(self.state_size, self.action_size)
        try:
            self.agent.load(model_path)
            self.get_logger().info(f"Successfully loaded model: {model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            sys.exit(1)

        self.agent.epsilon = 0.05

        self.env = TurtleBot3Env()
        self.state_processor = StateProcessor(n_lidar_bins=self.n_lidar_bins)

        self.init_debug_window()

    def init_debug_window(self):
        plt.ion() 
        self.debug_fig, self.debug_ax = plt.subplots(figsize=(6, 6))
        self.debug_fig.canvas.manager.set_window_title("Test Node - Range 8m Check")
        
        map_path = '/home/danny/project_rmov/src/stage_ros2/world/bitmaps/solid_cave.png'
        
        if os.path.exists(map_path):
            img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
            img = np.flipud(img) 
            self.debug_ax.imshow(img, cmap='gray', extent=[-8, 8, -8, 8], alpha=0.5, origin='lower')
        
        self.debug_ax.set_xlim(-8, 8)
        self.debug_ax.set_ylim(-8, 8)
        self.debug_ax.set_xlabel("World X")
        self.debug_ax.set_ylabel("World Y")
        self.debug_ax.grid(True, alpha=0.3)

        self.goal_dot, = self.debug_ax.plot([], [], 'g*', label='Goal', markersize=15)
        self.goal_circle = patches.Circle((0, 0), radius=0.5, color='g', fill=False, alpha=0.5)
        self.debug_ax.add_patch(self.goal_circle)
        self.debug_ax.legend(loc='upper right')

    def update_debug_window(self):
        if self.env.goal_position is None: return
        gx_rel, gy_rel = self.env.goal_position

        off_x = getattr(self.env, 'spawn_x', -7.0) 
        off_y = getattr(self.env, 'spawn_y', -6.5)
        
        gx_world = gx_rel + off_x
        gy_world = gy_rel + off_y

        self.goal_dot.set_data([gx_world], [gy_world])
        self.goal_circle.center = (gx_world, gy_world)
        self.goal_circle.radius = 0.3 

        self.debug_fig.canvas.draw()
        self.debug_fig.canvas.flush_events()

    def get_processed_state(self):
        return self.state_processor.get_state(
            self.env.scan_data,
            self.env.position,
            self.env.goal_position,
            self.env.yaw
        )

    def generate_constrained_random_goal(self):
        valid = False
        attempts = 0
        
        while not valid and attempts < 100:
            dist = np.random.uniform(2.0, 8.0) 
            angle = np.random.uniform(-np.pi, np.pi)

            cand_rel_x = dist * np.cos(angle)
            cand_rel_y = dist * np.sin(angle)

            world_x = cand_rel_x + self.env.spawn_x
            world_y = cand_rel_y + self.env.spawn_y

            if self.env.is_valid_point(world_x, world_y):
                self.env.goal_position = (cand_rel_x, cand_rel_y)
                valid = True
                self.get_logger().info(f"Goal Generated: ({cand_rel_x:.2f}, {cand_rel_y:.2f}) | Dist: {dist:.2f}m")
            
            attempts += 1
        
        if not valid:
            self.get_logger().warn("Could not find valid goal in range. Defaulting.")
            self.env.goal_position = (2.0, 0.0)

    def test(self, n_episodes: int = 5):
        self.get_logger().info("Waiting for data...")
        while self.env.scan_data is None:
            rclpy.spin_once(self.env, timeout_sec=0.5)
        
        success_count = 0
        total_rewards = []

        for episode in range(n_episodes):
            self.env.reset(random_goal=False) 
            self.generate_constrained_random_goal()
            self.update_debug_window()

            for _ in range(10): 
                rclpy.spin_once(self.env, timeout_sec=0.05)

            state = self.get_processed_state()
            episode_reward = 0
            
            for step in range(1000):
                action = self.agent.act(state, training=False)
                _, reward, done = self.env.step(action)
                state = self.get_processed_state()
                episode_reward += reward

                if step % 50 == 0: self.update_debug_window()

                if done:
                    if self.env.is_goal_reached():
                        success_count += 1
                        self.get_logger().info(f"SUCCESS! Reward: {episode_reward:.2f}")
                    else:
                        self.get_logger().info(f"FAIL/COLLISION. Reward: {episode_reward:.2f}")
                    break

                rclpy.spin_once(self.env, timeout_sec=0.01)

            total_rewards.append(episode_reward)

        plt.close(self.debug_fig)
        self.get_logger().info("="*40)
        self.get_logger().info(f"TEST COMPLETE ({n_episodes} Eps) | Range: 0-8m")
        self.get_logger().info(f"Success Rate: {success_count/n_episodes*100:.1f}%")
        self.get_logger().info("="*40)

def main(args=None):
    rclpy.init(args=args)
    if len(sys.argv) < 2:
        print("Usage: ros2 run dqn_project test_node <model_path>")
        return

    tester = DQNTestNode(sys.argv[1])
    try:
        tester.test(n_episodes=5)
    except KeyboardInterrupt:
        pass
    finally:
        tester.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()