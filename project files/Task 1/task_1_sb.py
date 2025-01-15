import gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

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
            low=0, high=255, shape=(1,), dtype=np.float32  # IR sensor value
        )
    
    def set_position(self, position: Position, orientation: Orientation) -> None:
        """Set the position of the Robobo in the simulation
        More information at: https://manual.coppeliarobotics.com/en/positionOrientationTransformation.htm
        """
        self._sim.setObjectPosition(self._robobo, [position.x, position.y, position.z])
        self._sim.setObjectOrientation(
            self._robobo, [orientation.yaw, orientation.pitch, orientation.roll]
        )

    def get_base_position(self) -> Position:
        """Get the position of the base to deliver food at.

        This only works in the simulation.
        Trivially doesn't work when the simulation does not have a base.
        """
        if self._base is None:
            raise AttributeError("Scene does not have a base")

        pos = self._sim.getObjectPosition(self._base, self._sim.handle_world)
        return Position(*pos)

    def reset(self, position):
        """Resets the environment and robot state."""
        self.rob.reset_wheels()
        self.rob.set_position(position)
        self.rob.play_simulation()
        return np.array([self.rob.read_irs()[4]])  # Initial IR value

    def step(self, action):
        """Takes a step in the environment based on the action."""
        if action == 0:  # Forward
            self.rob.move_blocking(50, 50, 100)
        elif action == 1:  # Left
            self.rob.move_blocking(-50, 50, 1000)
        elif action == 2:  # Right
            self.rob.move_blocking(50, -50, 1000)

        # Read IR sensor values
        ir_value = self.rob.read_irs()[4]
        obs = np.array([ir_value], dtype=np.float32)

        # Define rewards
        reward = 1 if ir_value < 100 else -10  # Positive reward for avoiding obstacles
        done = ir_value >= 100  # Episode ends when an obstacle is hit

        return obs, reward, done, {}

    def render(self, mode="human"):
        pass

    def close(self):
        self.rob.stop_simulation()

# Main DQN training script
def train_robot_with_dqn(rob: SimulationRobobo):
    # Initialize the environment
    env = DummyVecEnv([lambda: RoboboEnv(rob)])

    # Define the DQN model
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=0.01,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,
        target_update_interval=500,
        train_freq=4,
        verbose=1,
    )

    # Set up checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=100, save_path="./learning_machines/", name_prefix="dqn_robobo"
    )

    # Train the model
    model.learn(total_timesteps=500, callback=checkpoint_callback)

    # Save the model
    model.save("dqn_robobo_model")

    env.close()

# Example usage
def run_sequence_with_dqn(rob: SimulationRobobo):
    """Runs the sequence with a trained DQN model."""
    robobo_env = RoboboEnv(rob)
    model = DQN.load("dqn_robobo_model")

    obs = robobo_env.reset()
    done = False

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = robobo_env.step(action)
        print(f"Action: {action}, Reward: {reward}, Done: {done}")

    robobo_env.close()

def run_task_1_2(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    train_robot_with_dqn(rob)
    # run_sequence_with_dqn(rob)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
