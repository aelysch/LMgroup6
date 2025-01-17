import numpy as np
from stable_baselines3 import DQN
from robobo_interface import HardwareRobobo, IRobobo


class HardwareRoboboEnv:
    """
    Custom environment wrapper for running the trained model on HardwareRobobo.
    Mirrors the simulation environment's parameters, rewards, and proximity levels.
    """
    def __init__(self, rob: HardwareRobobo):
        self.rob = rob
        # Action space: [0: Forward, 1: Left, 2: Right]
        self.action_space = 3
        # Observation space: IR sensor values [FrontLL, FrontL, FrontC, FrontR, FrontRR]
        self.observation_space_shape = (5,)
        self.previous_obs = None
        self.consecutive_turns = 0  # To track consecutive turns

    def reset(self):
        """
        Resets the hardware state and initializes observation.
        Hardware cannot reset position, so we only reinitialize wheels and sensors.
        """
        self.rob.reset_wheels()
        ir_values = self.rob.read_irs()
        self.previous_obs = np.array(
            [ir_values[3], ir_values[2], ir_values[4], ir_values[6], ir_values[5]],
            dtype=np.float32,
        )  # FrontLL, FrontL, FrontC, FrontR, FrontRR
        self.consecutive_turns = 0
        return self.previous_obs

    def step(self, action):
        """
        Executes the action on the hardware and retrieves the next observation.
        Uses the same action mappings as in the simulation.
        """
        if action == 0:  # Forward
            self.rob.move_blocking(100, 100, 100)
        elif action == 1:  # Left
            self.rob.move_blocking(-50, 50, 100)
        elif action == 2:  # Right
            self.rob.move_blocking(50, -50, 100)

        # Read IR sensor values
        ir_values = self.rob.read_irs()
        obs = np.array(
            [ir_values[3], ir_values[2], ir_values[4], ir_values[6], ir_values[5]],
            dtype=np.float32,
        )  # FrontLL, FrontL, FrontC, FrontR, FrontRR

        # reward = 0
        done = False

        # # Reward calculation logic (mirrors simulation)
        # for val in obs:
        #     if val >= 10000:  # Collision detected
        #         reward = -10
        #         done = True
        #         break
        #     elif 5000 <= val < 10000:  # "Near" detected
        #         reward += 1  # Slight reward for being near an obstacle but not colliding
        #     else:  # Far away from obstacles
        #         reward += 5  # Safe movement or clear area

        # # Additional exploration incentives
        # if action == 0:  # Forward
        #     if obs[2] < 5000:  # Clear front
        #         reward += 2  # Higher reward for safe forward movement
        # elif action in [1, 2]:  # Turning
        #     reward -= 2  # Slight penalty for turning excessively

        # # Penalize lack of movement (repeated observations)
        # if self.previous_obs is not None and np.array_equal(obs, self.previous_obs):
        #     reward -= 1

        # # Penalize consecutive turning
        # if action in [1, 2]:
        #     self.consecutive_turns += 1
        # else:
        #     self.consecutive_turns = 0

        # if self.consecutive_turns > 2:
        #     reward -= 10  # Penalize excessive turning

                # Reward components
        base_reward = 0
        collision_penalty = -1.0
        near_obstacle_reward = 0.1
        clear_path_reward = 0.2
        exploration_penalty = -0.1
        turn_penalty = -0.05
        excessive_turn_penalty = -0.6
        forward_bonus = 0.5

        done = False

        # Calculate rewards
        for val in obs:
            if val >= 10000:  # Collision detected
                base_reward += collision_penalty
                done = True
                break
            elif 500 <= val < 10000:  # Near obstacle
                base_reward += near_obstacle_reward
            else:  # Far from obstacles
                base_reward += clear_path_reward

        # Additional incentives
        if action == 0:  # Forward movement
            if obs[2] < 500:  # Clear front
                base_reward += forward_bonus
        elif action in [1, 2]:  # Turning actions
            base_reward += turn_penalty

        # Penalize lack of movement (repeated observations)
        if self.episode_steps > 1 and np.array_equal(obs, self.previous_obs):
            base_reward += exploration_penalty

        # Penalize consecutive turning
        if action in [1, 2]:
            self.consecutive_turns += 1
        else:
            self.consecutive_turns = 0

        if self.consecutive_turns > 3:
            base_reward += excessive_turn_penalty

        # Normalize the final reward
        normalized_reward = np.clip(base_reward, -1.0, 1.0)  # Ensure reward stays in range [-1, 1]

        self.previous_obs = obs
        return obs, normalized_reward, done, {}

    def close(self):
        self.rob.reset_wheels()

# Script to load and run the trained model
def run_trained_model_on_hardware(rob: HardwareRobobo):
    """
    Runs the pre-trained DQN model on HardwareRobobo with matching parameters, rewards, and proximity levels.
    """
    robobo_env = HardwareRoboboEnv(rob)
    model = DQN.load("/root/results/dqn_robobo_model")

    obs = robobo_env.reset()
    done = False

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = robobo_env.step(action)
        print(f"Action: {action}, Reward: {reward}, Done: {done}")

    robobo_env.close()

