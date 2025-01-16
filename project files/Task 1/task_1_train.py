import cv2
import json
import matplotlib.pyplot as plt

import os

from data_files import FIGURES_DIR
import random
from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
    HardwareRobobo,
)

# Initialize parameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 0.1  # Exploration rate
Q = {}  # Q-table: Q[state][action]


#action space 
actions = ["forward", "left", "right", "backward"]

def get_discrete_state(rob):
    def state_ir_value(ir_value):
        if ir_value < 50:
            return "far"
        elif 50 <= ir_value < 100:
            return "medium"
        elif 100 < ir_value <= 10000:
            return "close"
        elif ir_value > 10000:
            return "collision"
    
    front_ir = rob.read_irs()[4]
    left_ir = (rob.read_irs()[2] + rob.read_irs()[7]) / 2
    right_ir = (rob.read_irs()[3] + rob.read_irs()[5]) / 2

    front_state = state_ir_value(front_ir)
    left_state = state_ir_value(left_ir)
    right_state = state_ir_value(right_ir)

    # Combine states into a tuple
    return (front_state, left_state, right_state)


def select_action(state):
    """Select an action using ε-greedy policy."""
    if state not in Q:
        Q[state] = {action: 0 for action in actions}  # Initialize Q-values for new state

    if random.random() < epsilon:
        return random.choice(actions)  # Exploration
        #return random.choices(actions, weights=[0.5, 0.2, 0.2, 0.1])[0]
    else:
        return max(Q[state], key=Q[state].get)  # Exploitation
    
# def get_reward(state):
    reward_mapping = {
        "far": 1,         
        "medium":0.5,       
        "close": -5,       
        "collision": -10  
    }
    return reward_mapping[state]

#weighted sum of all front sensor states 
# def get_reward(state): 
    front, left, right = state

    reward_front = 1 if front == "far" else -10 if front == "collision" else -5 if front == "close" else 0.5
    reward_left = 0.5 if left == "far" else -10 if left == "collision" else -3 if left == "close" else 0.2
    reward_right = 0.5 if right == "far" else -10 if right == "collision" else -3 if right == "close" else 0.2

    total_reward = 0.6 * reward_front + 0.2 * reward_left + 0.2 * reward_right # Weighted sum
    return total_reward

#reward function prioritising front sensor
def get_reward(state):
    front, left, right = state

    if front == "collision" or left == "collision" or right == "collision":
        return -10  
    elif front == "close":
        return -5  
    elif front == "medium":
        return 1
    else:  
        return 5
    
    
def run_episode(rob, max_steps=100):
    state = get_discrete_state(rob)
    action = select_action(state)
    total_reward = 0
    ep_reward_history=[]
    episode_data =[] 

    for _ in range(max_steps):
        if action == "forward":
            rob.move_blocking(50, 50, 100)
        elif action == "left":
            rob.move_blocking(-50, -50, 100)
            rob.move_blocking(-10, 10, 1000)
        elif action == "right":
            rob.move_blocking(-50, -50, 100)
            rob.move_blocking(10, -10, 1000)
        elif action == "backward":
            rob.move_blocking(-50, -50, 100)

        next_state = get_discrete_state(rob)
        reward = get_reward(next_state)            
        next_action = select_action(next_state) 

        # Update Q-value
        Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])

        episode_data.append((state, action, reward))

        state, action = next_state, next_action  # Update state and action
        total_reward += reward
        ep_reward_history.append(total_reward)

        # if reward == -10:  # Stop if collision occurs
        #     break

    return total_reward, ep_reward_history, episode_data


def extract_policy(q_table):

    policy = {}
    for state, actions in q_table.items():
        best_action = max(actions, key=actions.get)  # Action with highest Q-value
        policy[state] = best_action
    return policy

def extract_final_policy(q_table):
    policy = {}
    for state, actions in q_table.items():
        best_action = max(actions, key=actions.get)  # Action with highest Q-value
        policy[state] = best_action
    return policy


def plot_cumulative_reward(reward_history):
    """Plot cumulative reward over episodes."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(reward_history) + 1), reward_history, marker='o')
    plt.xlabel("Step")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Reward Over Episodes")
    plt.grid(True)
    plt.show()

def run_sequence(rob, num_episodes=100, convergence_threshold=1e-4):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    cumulative_reward = 0
    global epsilon
    previous_q_table = {}  
    reward_history = []  

    for episode in range(num_episodes):
        total_reward, ep_reward_history, _ = run_episode(rob)
        cumulative_reward += total_reward

        epsilon = max(epsilon * 0.99, 0.1)  
        reward_history.append(ep_reward_history)
        # Check for convergence
        max_delta = 0
        for state in Q:
            for action in Q[state]:
                delta = abs(Q[state][action] - previous_q_table.get(state, {}).get(action, 0))
                max_delta = max(max_delta, delta)
        
        if max_delta < convergence_threshold:
            print(f"\nConverged after Episode {episode + 1}")
            break

        previous_q_table = Q.copy()  

        print(f"Episode {episode + 1}: Total Reward = {total_reward}, Cumulative Reward = {cumulative_reward}")
        print("------------------------------------------------------")

    # Print the final converged Q-table
    print("\nFinal Converged Q-table:")
    print(Q)
    # for state, actions in Q.items():
    #     print(f"State: {state}, Actions: {actions}")
    # final_policy = extract_final_policy(Q)
    # save_policy_to_file(final_policy)
    print(reward_history)
    plot_cumulative_reward(reward_history)


    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()

