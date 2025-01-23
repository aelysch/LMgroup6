import os
import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
import time
import cv2
import random


from robobo_interface import IRobobo, SimulationRobobo


class SimpleRoboboEnv:
    """
    Simplified environment for the Robobo robot.
    """
    def __init__(self, rob: SimulationRobobo):
        self.rob = rob
        self.action_space = [0, 1, 2,3]  
        self.observation_space_size = 5  # Number of IR sensors
        self.position = self.rob.get_position()
        self.green_threshold = 5000
        self.orientation = self.rob.get_orientation()
        self.reset()
        

    def get_state(self):
        """Combine IR sensor readings and image features to create the state."""
        ir_values = self.rob.read_irs()
        image = self.rob.read_image_front()

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
        # Calculate green centroid
        moments = cv2.moments(green_mask)
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            green_coords = (cx, cy)
        else:
            green_coords = None

        green_detected = np.sum(green_mask)
        green_nearby = int(green_detected > self.green_threshold)
        green_front = int(green_detected > 7500)


        state = np.array(
            [ir_values[3], ir_values[2], ir_values[4], ir_values[6], green_detected],  # 5 elements
            dtype=np.float32,) #[BackL, BackR, FrontL, FrontR, FrontC, FrontRR, BackC, FrontLL]
        return  state, green_front
    
    def reset(self):
        """Resets the environment and the robot."""
        self.rob.reset_wheels()
        self.rob.set_position(self.position, self.orientation)
        self.rob.play_simulation()
        self.state, _ = self.get_state()
        self.previous_distance_to_green = float('inf')   # Initialize self.state using get_state()
        return self.state

    def step(self, action):
        if action == 0:  # Forward
            self.rob.move_blocking(50, 50, 100)
        elif action == 1:  # Left
            self.rob.move_blocking(-50, 50, 100)
        elif action == 2:  # Right
            self.rob.move_blocking(50, -50, 100)
        elif action == 3:  # Backward
            self.rob.move_blocking(-50, -50, 100)

        # Get new state
        ir_values = self.rob.read_irs()
        self.state, green_front = self.get_state()

        # Calculate reward
        reward = 0
        food_collected = 0

        done = False
        ir_values = self.state[:4]
        green_detected = self.state[4]
       # print(max(ir_values))
       # print(green_nearby)
        if green_detected and max(ir_values[0:1]) > 1000:  # Green detected and close
            reward += 100  # Strong reward for successfully eating the package
            food_collected += 1
          #  done = True  # End the episode
        elif action == 0 and green_front:  # Green package detected
            reward += 1
        elif max(ir_values) > 200 and action ==3:
            reward+=1
        elif max(ir_values) > 200 and green_detected > 7500:  # Collision detected
            reward -=10
        elif action == 3:  # Backward motion (if added)
            reward -= 2
           # done = True
# not know what to do turn 

        # Example: Extracting x, y, z from Position
        prev_position = self.rob.get_position()
        new_position = self.rob.get_position()

        # Ensure positions are comparable
        prev_coords = np.array([prev_position.x, prev_position.y])  # Use only x and y
        new_coords = np.array([new_position.x, new_position.y])     # Use only x and y
        movement_distance = np.linalg.norm(new_coords - prev_coords)

        # Calculate Euclidean distance
    #    if movement_distance < 10:  
    #        reward -= 10  # Stronger penalty for staying in one place
    #    elif 10 <= movement_distance <= 20:  
    #        reward += 2  # Small reward for moderate movement
    #    else:
    #        reward += 5 # Penalize lack of movement

       # if green_coords is not None:
      #      distance_to_green = np.linalg.norm(np.array(green_coords) - np.array(new_coords))
      #      if distance_to_green < self.previous_distance_to_green:
       #         reward += 2  # Higher reward for reducing distance to green
      #          self.previous_distance_to_green = distance_to_green
       #     elif distance_to_green > self.previous_distance_to_green:
      #          reward -= 5  # Penalize moving further from green food

        return self.state, reward, done, food_collected

    def close(self):
        self.rob.stop_simulation()


def train_robot_with_q_learning(env, episodes=50, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995, num_bins=5):
    # Initialize Q-table
    q_table = np.zeros((num_bins, num_bins, num_bins, num_bins, 2, len(env.action_space)))
    rewards = []
    steps_per_episode = []  # Track steps for each episode

    def discretize_state(state):
        bins = np.linspace(0, 255, num_bins)
        ir_bins = tuple(np.digitize(s, bins) - 1 for s in state[:4])  # Discretize IR values
        green_bin = int(state[4])  # 0 or 1 for green detected
        return ir_bins + (green_bin,)

    for episode in range(episodes):
        state = env.reset()
        state = discretize_state(state)
        total_reward = 0
        max_steps = 100 
        done = False
        steps = 0  # Reset steps for the episode
        food_collected = 0  # Reset food collected for the episode

        while not done and steps < max_steps:
            # Exploration vs. Exploitation
            if np.random.rand() < exploration_rate:
                action = np.random.choice(env.action_space)
            else:
                action = np.argmax(q_table[state])

            # Take action
            next_state, reward, done, food = env.step(action)
            next_state = discretize_state(next_state)
            total_reward += reward
            food_collected += food

            # Update Q-table
            best_next_action = np.argmax(q_table[next_state])
            q_table[state][action] = q_table[state][action] + learning_rate * (
                reward + discount_factor * q_table[next_state][best_next_action] - q_table[state][action]
            )

            state = next_state
            steps += 1  # Increment step count

        # Track steps for the episode
        steps_per_episode.append(steps)

        # Decay exploration rate
        exploration_rate *= exploration_decay
        rewards.append(total_reward)
        print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}, Steps: {steps}, Food Collected: {food_collected}")

    # Compute and print average steps after training
    avg_steps = sum(steps_per_episode) / len(steps_per_episode)
    print(f"Average Steps Before Collision: {avg_steps}")

    # Save Q-table
    np.save("q_table.npy", q_table)
    print("Q-table saved successfully.")

    return q_table, rewards




def plot_rewards(rewards, output_path="rewards_plot.png"):
    """Plot rewards over episodes."""
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title("Rewards Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid()
    plt.savefig(output_path)
    plt.show()


# Main function
def run_task(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    env = SimpleRoboboEnv(rob)
    q_table, rewards = train_robot_with_q_learning(env, episodes=50)
    plot_rewards(rewards)
    env.close()

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()


def run_task(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):#
        rob.play_simulation()
    rob.set_phone_tilt(105,50)
    env = SimpleRoboboEnv(rob)

    # Train and save the model
    q_table, rewards = train_robot_with_q_learning(env, episodes=50)
    file_path = '/root/results/qlearning_model_3.npy'
    np.save(file_path, q_table)
    print(f"Q-table saved successfully at {file_path}.")
    #plot_rewards(rewards)

    # Validate the model
    print("\nStarting validation...")
   # validate_robot(env, q_table, episodes=10)

    env.close()
    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
