#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import matplotlib.patches as patches
import cv2 

from dqn_project.dqn_agent import DQNAgent
from dqn_project.environment import TurtleBot3Env
from dqn_project.state_processor import StateProcessor

class DQNTrainingNode(Node):
    """Main training node for DQN navigation"""

    def __init__(self):
        super().__init__('dqn_training_node')

        # Training parameters
        self.n_episodes = 500
        self.max_steps_per_episode = 500
        self.state_size = 50  # 10 LiDAR bins + 2 goal info
        self.action_size = 5

        # Initialize components
        self.env = TurtleBot3Env()
        self.state_processor = StateProcessor(n_lidar_bins=48)
        self.agent = DQNAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            learning_rate=0.001,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.99,
            memory_size=50000,
            batch_size=128
        )

        # Logging
        self.episode_rewards = []
        self.episode_steps = []
        self.success_count = 0
        self.collision_count = 0

        # Create results directory in your Home folder so it's easy to find
        self.results_dir = os.path.expanduser(f"~/dqn_results/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.results_dir, exist_ok=True)
        self.get_logger().info(f"Saving results to: {self.results_dir}")

        self.init_debug_window()

    def init_debug_window(self):
        """Sets up the live map visualization"""
        plt.ion() # Interactive mode on
        self.debug_fig, self.debug_ax = plt.subplots(figsize=(6, 6))
        self.debug_fig.canvas.manager.set_window_title("Live Logic Debugger")
        
        # Load the map image for background
        map_path = '/home/danny/project_rmov/src/stage_ros2/world/bitmaps/solid_cave.png'
        if os.path.exists(map_path):
            img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
            img = np.flipud(img)
            self.debug_ax.imshow(img, cmap='gray', extent=[-8, 8, -8, 8], alpha=0.5, origin='lower')
        
        self.debug_ax.set_xlim(-8, 8)
        self.debug_ax.set_ylim(-8, 8)
        self.debug_ax.set_xlabel("World X (Meters)")
        self.debug_ax.set_ylabel("World Y (Meters)")
        self.debug_ax.grid(True, alpha=0.3)

        self.goal_dot, = self.debug_ax.plot([], [], 'g*', label='Goal', markersize=12)  # Green Star
        self.goal_circle = patches.Circle((0, 0), radius=0.5, color='g', fill=False, alpha=0.5)
        self.debug_ax.add_patch(self.goal_circle)
        self.debug_ax.legend(loc='upper right')

    def update_debug_window(self):
        """Updates the robot and goal positions on the plot"""
        # 1. Get positions from Environment (These are Odom/Relative coords)
        gx_odom, gy_odom = self.env.goal_position

        # 2. Convert to World Coordinates for Plotting
        # World = Odom + Spawn_Offset (-7)
        off_x, off_y = self.env.spawn_x, self.env.spawn_y
        
        gx_world = gx_odom + off_x
        gy_world = gy_odom + off_y

        # 3. Update Plots (Only Goal)
        self.goal_dot.set_data([gx_world], [gy_world])
        self.goal_circle.center = (gx_world, gy_world)
        self.goal_circle.radius = self.env.goal_threshold 

        # 4. Refresh
        self.debug_fig.canvas.draw()
        self.debug_fig.canvas.flush_events()


    def get_processed_state(self):
        """Get processed state from environment"""
        return self.state_processor.get_state(
            self.env.scan_data,
            self.env.position,
            self.env.goal_position,
            self.env.yaw
        )

    def train(self):
        """Main training loop"""
        self.get_logger().info("Starting DQN training...")
        
        # Wait for valid data before starting
        while self.env.scan_data is None:
             self.get_logger().info("Waiting for Lidar data...", once=True)
             rclpy.spin_once(self.env, timeout_sec=0.5)

        for episode in range(self.n_episodes):
            # Reset environment
            self.env.reset(random_goal=True)
            # Spin a few times to ensure OdomWrapper has updated the reset
            for _ in range(5):
                rclpy.spin_once(self.env, timeout_sec=0.1)

            state = self.get_processed_state()
            episode_reward = 0

            for step in range(self.max_steps_per_episode):
                # Select and execute action
                action = self.agent.act(state, training=True)
                next_state_raw, reward, done = self.env.step(action)
                next_state = self.get_processed_state()

                # Store experience
                self.agent.remember(state, action, reward, next_state, done)

                # Train agent
                if len(self.agent.memory) > self.agent.batch_size:
                    loss = self.agent.replay()

                episode_reward += reward
                state = next_state

                if step % 50 == 0: 
                    self.update_debug_window()

                if done:
                    break

                # Prevent blocking
                rclpy.spin_once(self.env, timeout_sec=0.01)

            # Episode statistics
            self.episode_rewards.append(episode_reward)
            self.episode_steps.append(step + 1)

            if self.env.is_goal_reached():
                self.success_count += 1
            if self.env.is_collision():
                self.collision_count += 1

            # Logging
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                success_rate = self.success_count / (episode + 1) * 100

                self.get_logger().info(
                    f"Episode: {episode}/{self.n_episodes} | "
                    f"Steps: {step+1} | "
                    f"Reward: {episode_reward:.2f} | "
                    f"Avg Reward (10): {avg_reward:.2f} | "
                    f"Epsilon: {self.agent.epsilon:.3f} | "
                    f"Success Rate: {success_rate:.1f}%"
                )

            # Save model periodically
            if episode % 50 == 0 and episode > 0:
                model_path = os.path.join(self.results_dir, f"model_episode_{episode}.pkl")
                self.agent.save(model_path)

        # Final save
        self.agent.save(os.path.join(self.results_dir, "model_final.pkl"))

        plt.close(self.debug_fig)

        self.plot_results()

    def plot_results(self):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Episode rewards
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True)

        # Moving average reward
        window = 20
        if len(self.episode_rewards) >= window:
            moving_avg = np.convolve(self.episode_rewards,
                                    np.ones(window)/window, mode='valid')
            axes[0, 1].plot(moving_avg)
            axes[0, 1].set_title(f'Moving Average Reward (window={window})')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Avg Reward')
            axes[0, 1].grid(True)

        # Episode steps
        axes[1, 0].plot(self.episode_steps)
        axes[1, 0].set_title('Steps per Episode')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].grid(True)

        # Success rate
        window = 50
        success_history = []
        for i in range(len(self.episode_rewards)):
            if i < window:
                success_history.append(self.success_count / (i + 1))
            else:
                # Count successes in last 'window' episodes
                recent_successes = sum([1 for j in range(i-window+1, i+1)
                                       if self.episode_rewards[j] > 100])
                success_history.append(recent_successes / window)

        axes[1, 1].plot([s * 100 for s in success_history])
        axes[1, 1].set_title(f'Success Rate (window={window})')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Success Rate (%)')
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'training_results.png'))
        self.get_logger().info(f"Results saved to {self.results_dir}")

def main(args=None):
    rclpy.init(args=args)
    trainer = DQNTrainingNode()

    try:
        trainer.train()
    except KeyboardInterrupt:
        trainer.get_logger().info("Training interrupted by user")
    finally:
        trainer.env.send_velocity(0.0, 0.0)
        trainer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()