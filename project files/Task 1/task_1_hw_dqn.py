import gym
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from robobo_interface import IRobobo, SimulationRobobo
from robobo_interface import (
    Emotion,
    LedColor,
    LedId,
    Acceleration,
    Position,
    Orientation,
    WheelPosition,
    SoundEmotion,
)

class RoboboEnv(gym.Env):
    """
    Custom Gym environment for the iRobobo robot.
    """
    def __init__(self, rob: SimulationRobobo):
        super(RoboboEnv, self).__init__()
        self.rob = rob
        self.action_space = gym.spaces.Discrete(3)  # [0: Forward, 1: Left, 2: Right]
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(5,), dtype=np.float32  # IR sensor values: FrontLL, FrontL, FrontC, FrontR, FrontRR
        )

        # Statistics tracking
        self.episode_rewards = []  # Current episode rewards
        self.total_rewards = []    # Total rewards for all episodes
        self.episode_steps = 0     # Steps in the current episode
        self.consecutive_turns = 0 # Track consecutive turns
        self.previous_obs = None  # Previous observation for exploration penalty

    def reset(self):
        """Resets the environment and robot state."""

        self.rob.reset_wheels()
        ir_values = self.rob.read_irs()
        self.previous_obs = np.array(
            [ir_values[3], ir_values[2], ir_values[4], ir_values[6], ir_values[5]], 
            dtype=np.float32
        )  # FrontLL, FrontL, FrontC, FrontR, FrontRR
        return self.previous_obs

    def step(self, action):
        """Takes a step in the environment based on the action."""
        if action == 0:  # Forward
            self.rob.move_blocking(50, 50, 100)
        elif action == 1:  # Left
            self.rob.move_blocking(-50, 50, 100)
        elif action == 2:  # Right
            self.rob.move_blocking(50, -50, 100)

        # Read IR sensor values
        ir_values = self.rob.read_irs()
        obs = np.array([ir_values[3], ir_values[2], ir_values[4], ir_values[6], ir_values[5]], dtype=np.float32)  # FrontLL, FrontL, FrontC, FrontR, FrontRR

        reward = 0
        done = False

        # Define proximity levels based on IR values
        for val in obs:
            if val >= 100:  # Collision detected
                reward = -10
                done = True
                break
            elif 50 <= val < 100:  # "Near" detected
                reward += 0.5  # Slight reward for being near an obstacle but not colliding
            else:  # Far away from obstacles
                reward += 1  # Safe movement or clear area

        # Additional exploration incentives
        if action == 0:  # Forward
            if obs[2] < 50:  # Clear front
                reward += 2  # Higher reward for safe forward movement
        elif action in [1, 2]:  # Turning
            reward -= 0.1  # Slight penalty for turning excessively

        # Penalize lack of movement (repeated observations)
        if self.episode_steps > 1 and np.array_equal(obs, self.previous_obs):
            reward -= 1

        # Penalize consecutive turning
        if action in [1, 2]:
            self.consecutive_turns += 1
        else:
            self.consecutive_turns = 0

        if self.consecutive_turns > 3:
            reward -= 2  # Penalize excessive turning

        done = any(val >= 100 for val in obs)  # End episode on collision

        # Track statistics
        self.episode_rewards.append(reward)
        self.episode_steps += 1
        self.previous_obs = obs

        return obs, reward, done, {}

    def render(self, mode="human"):
        pass

    def close(self):
        self.rob.stop_simulation()

class TrainingMonitor(BaseCallback):
    """
    Custom callback for monitoring training metrics.
    """
    def __init__(self, verbose=0, stop_threshold=200, rolling_window=10):
        super(TrainingMonitor, self).__init__(verbose)
        self.episode_rewards = []
        self.rolling_window = rolling_window
        self.stop_threshold = stop_threshold

    def _on_step(self) -> bool:
        if self.locals["dones"][0]:  # Check if an episode ended
            info = self.locals["infos"][0]
            if "episode" in info:  # Safeguard against missing 'episode' key
                episode_reward = info["episode"]["r"]
                self.episode_rewards.append(episode_reward)
                
                if self.verbose > 0:
                    print(f"Episode ended: Reward={episode_reward}")

                # Check for convergence
                if len(self.episode_rewards) >= self.rolling_window:
                    avg_reward = np.mean(self.episode_rewards[-self.rolling_window:])
                    if avg_reward >= self.stop_threshold:
                        print(f"Training stopped: Converged with avg reward {avg_reward:.2f}")
                        return False  # Signal to stop training
        return True

    
def configure_logger(model):
    """
    Configures TensorBoard for real-time logging of the DQN model.
    """
    new_logger = configure("tensorboard_logs", ["stdout", "tensorboard"])
    model.set_logger(new_logger)

# Main DQN training script
def train_robot_with_dqn(rob: SimulationRobobo):
    # Initialize the environment
    env = DummyVecEnv([lambda: Monitor(RoboboEnv(rob))])

    # Define the DQN model
    policy_kwargs = dict(
        net_arch=[256, 256],  # Two hidden layers with 256 neurons each
        activation_fn=torch.nn.ReLU
    )

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=0.001,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,
        target_update_interval=500,
        train_freq=4,
        verbose=1,
        exploration_initial_eps=1.0,   # Start with high exploration
        exploration_final_eps=0.1,    # Maintain some exploration
        exploration_fraction=0.5,     # Take longer to decay epsilon
        tensorboard_log="./tensorboard_logs/",
        policy_kwargs=policy_kwargs
    )

        # Statistics storage
    episode_rewards = []
    episode_lengths = []

    # Configure logger
    configure_logger(model)

    # Train the model
    model.learn(total_timesteps=5000, callback=TrainingMonitor(verbose=1, stop_threshold=200, rolling_window=10))

    # Save statistics
    np.save("/root/results/episode_rewards.npy", episode_rewards)

    # Save the model
    try:
        model.save('/root/results/dqn_robobo_model')
    except Exception as e:
        print(f"Failed to save the model: {e}")

    env.close()

# Run with final model
def run_sequence_with_dqn(rob: SimulationRobobo):
    """Runs the sequence with a trained DQN model."""
    robobo_env = RoboboEnv(rob)
    model = DQN.load("/root/results/dqn_robobo_model")

    obs = robobo_env.reset()
    done = False

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = robobo_env.step(action)
        print(f"Action: {action}, Reward: {reward}, Done: {done}")

    robobo_env.close()
