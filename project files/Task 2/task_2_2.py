import cv2
import numpy as np
import random
import time
from robobo_interface import SimulationRobobo

# Initialize parameters
alpha = 0.2  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 0.5  # Exploration rate
Q = {}  # Q-table: Q[state][action]

# Action space
actions = ["forward", "left", "right"]


def preprocess_camera_data(rob):
    """
    Detect green areas in the camera feed and divide the screen into left, center, and right sections.
    Returns the percentage of green in each section.
    """
    # Read camera image
    image = rob.read_image_front()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV range for green
    lower_green = np.array([40, 50, 50], dtype=np.uint8)
    upper_green = np.array([80, 255, 255], dtype=np.uint8)

    # Create a mask for green pixels
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Get the image dimensions
    height, width = mask.shape

    # Divide the screen into three sections: left, center, and right
    left_section = mask[:, :width // 3]
    center_section = mask[:, width // 3:2 * width // 3]
    right_section = mask[:, 2 * width // 3:]

    # Calculate the total pixels and green pixel counts for each section
    total_pixels = mask.size
    green_left = left_section.sum() // 255
    green_center = center_section.sum() // 255
    green_right = right_section.sum() // 255

    # Calculate the percentage of green pixels in each section
    left_percentage = (green_left / total_pixels) * 100
    center_percentage = (green_center / total_pixels) * 100
    right_percentage = (green_right / total_pixels) * 100

    print(f"[Camera Data] Green percentage - Left: {left_percentage:.2f}%, Center: {center_percentage:.2f}%, Right: {right_percentage:.2f}%")

    return left_percentage, center_percentage, right_percentage


def get_discrete_state(rob):
    """
    Get the discrete state based on the percentage of green pixels in left, center, and right sections.
    """
    # Get green percentages from the camera
    left_percentage, center_percentage, right_percentage = preprocess_camera_data(rob)

    # Define thresholds to classify green presence
    def classify_green(percentage):
        if percentage > 10:  # Example threshold for "high" green presence
            return "high"
        elif percentage > 5:  # Example threshold for "medium" green presence
            return "medium"
        elif percentage > 0:  # Far green presence
            return "far"
        else:
            return "none"

    left_state = classify_green(left_percentage)
    center_state = classify_green(center_percentage)
    right_state = classify_green(right_percentage)

    collision_detected = rob.read_irs()[4] > 10000
    collision_state = "collision" if collision_detected else "no_collision"

    return (left_state, center_state, right_state, collision_state)



def select_action(state):
    """Select an action using ε-greedy policy."""
    if state not in Q:
        Q[state] = {action: 0 for action in actions}  # Initialize Q-values for new state

    if random.random() < epsilon:  # Exploration
        return random.choice(actions)
    else:  # Exploitation
        return max(Q[state], key=Q[state].get)


def get_reward(state, collected):
    left, center, right, collision = state

    if collision == "collision":
        if center == "high":
            return 100  
        else:
            return -20  

    if center == "high":
        return 50  
    elif center == "medium":
        return 10  
    elif center == "far":
        return 5  
    elif left == "high" or right == "high":
        return 10
    elif left == "medium" or right == "medium":
        return 3  
    elif left == "far" or right == "far":
        return 1  
    else:
        return -1  


def run_episode(rob, max_duration=60):
    """Run a single episode to guide the robot."""
    state = get_discrete_state(rob)
    action = select_action(state)
    total_reward = 0
    collected_blocks = 0
    start_time = time.time()

    while True:
        # Execute action
        if action == "forward":
            rob.move_blocking(100, 100, 500)
        elif action == "left":
            rob.move_blocking(-50, -50, 100)
            rob.move_blocking(-10, 10, 1000)
        elif action == "right":
            rob.move_blocking(-50, -50, 100)
            rob.move_blocking(10, -10, 1000)

        # Get the next state
        next_state = get_discrete_state(rob)
        collected = next_state[3] == "collision" and next_state[1] == "high"  # Collision with green in center

        # Reward logic
        reward = get_reward(next_state, collected)
        print("Reward:", reward)

        # Q-value update
        next_action = select_action(next_state)
        Q[state][action] += alpha * (
            reward + gamma * Q[next_state][next_action] - Q[state][action]
        )

        state, action = next_state, next_action
        total_reward += reward

        # Terminate after time limit
        if time.time() - start_time >= max_duration:
            print("[Episode] Time limit reached. Ending episode.")
            break

    duration = time.time() - start_time
    print(f"[Episode] Total Reward: {total_reward}")
    return total_reward, duration, collected_blocks



def run_sequence(rob, num_episodes=3):
    """Train the robot for box collection using Q-learning."""
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    rob.set_phone_tilt(103,50)

    previous_q_table = {}
    results = []

    for episode in range(num_episodes):
        print(f"Starting Episode {episode + 1}")
        total_reward, duration, collected_blocks = run_episode(rob)
        results.append({"episode": episode + 1, "duration": duration, "collected_blocks": collected_blocks})
        print(f"Episode {episode + 1}: Duration = {duration:.2f}s, Collected Blocks = {collected_blocks}")

        # max_change = 0
        # for state, actions in Q.items():
        #     if state not in previous_q_table:
        #         previous_q_table[state] = {a: 0 for a in actions}
        #     for action, value in actions.items():
        #         change = abs(value - previous_q_table[state].get(action, 0))
        #         max_change = max(max_change, change)

        # if max_change < convergence_threshold:
        #     print(f"\nConvergence detected after {episode + 1} episodes. Stopping training.")
        #     break

        # previous_q_table = {state: actions.copy() for state, actions in Q.items()}


    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()

    print("\nFinal Q-Table:")
    print(Q)
    print("\nResults:")
    for result in results:
        print(f"Episode {result['episode']}: Duration = {result['duration']:.2f}s, Collected Blocks = {result['collected_blocks']}")
