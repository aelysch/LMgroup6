import cv2
import numpy as np
import random
import time
from robobo_interface import SimulationRobobo

# Initialize parameters
alpha = 0.2  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.5  # Exploration rate
Q = {}  # Q-table: Q[state][action]

# Expanded Action Space
actions = ["forward", "left", "right", "forward_left", "forward_right", "backward", "sharp_left", "sharp_right"]

def preprocess_camera_data(rob):
    """
    Detect green and red areas in the camera feed and divide the screen into left, center, and right sections.
    Returns the percentage of green and red in each section.
    """
    # Read camera image
    image = rob.read_image_front()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV ranges for green and red
    lower_green = np.array([40, 50, 50], dtype=np.uint8)
    upper_green = np.array([80, 255, 255], dtype=np.uint8)
    lower_red1 = np.array([0, 50, 50], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([170, 50, 50], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    # Create masks for green and red
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Divide the screen into three sections: left, center, right
    height, width = green_mask.shape
    sections = {
        "green": {
            "left": green_mask[:, :width // 3].sum() // 255,
            "center": green_mask[:, width // 3:2 * width // 3].sum() // 255,
            "right": green_mask[:, 2 * width // 3:].sum() // 255,
        },
        "red": {
            "left": red_mask[:, :width // 3].sum() // 255,
            "center": red_mask[:, width // 3:2 * width // 3].sum() // 255,
            "right": red_mask[:, 2 * width // 3:].sum() // 255,
        }
    }

    total_pixels = green_mask.size

    # Convert to percentages
    for color in sections:
        for section in sections[color]:
            sections[color][section] = (sections[color][section] / total_pixels) * 100

    return sections["green"], sections["red"]


def get_discrete_state(rob):
    """
    Get the discrete state for green and red areas in left, center, and right sections.
    """
    green_percentages, red_percentages = preprocess_camera_data(rob)

    def classify_green(percentage):
        """Classifies green intensity based on default thresholds."""
        if percentage > 0.5:
            return "high"
        elif percentage > 0.1:
            return "medium"
        elif percentage > 0.05:
            return "low"
        else:
            return "none"

    def classify_red(percentage):
        """Classifies red intensity based on new thresholds (high >4, medium >2, low >1, else none)."""
        if percentage > 0.9:
            return "high"
        elif percentage > 0.3:
            return "medium"
        elif percentage > 0.0005:
            return "low"
        else:
            return "none"

    green_state = {section: classify_green(green_percentages[section]) for section in green_percentages}
    red_state = {section: classify_red(red_percentages[section]) for section in red_percentages}

    collision_detected = rob.read_irs()[4] > 10000
    collision_state = "collision" if collision_detected else "no_collision"
    #print(green_state["left"], green_state["center"], green_state["right"],
        # if not (red_center == "high" and green_center in ["high", "medium", "low"]):
        # reward += -10

    return (
        green_state["left"], green_state["center"], green_state["right"],
        red_state["left"], red_state["center"], red_state["right"],
        collision_state
    )


def get_reward(state, collected):
    """
    Reward function prioritizing red block in center and navigating toward green.
    """
    green_left, green_center, green_right, red_left, red_center, red_right, collision = state

    # Initialize reward
    reward = 0

    # **Prioritize Red in the Center** (Robot should interact with red block)
    if red_center == "high":
        reward += 70  # High reward for interacting with red in center
    elif red_center == "medium":
        reward += 50  # Medium reward
    elif red_center == "low":
        reward += 20  # Low reward

    if red_left == "high" or red_right == "high":
        reward += 20
    elif red_left == "medium" or red_right == "medium":
        reward += 10
    elif red_left == "low" or red_right == "low":
        reward += 5

    # **Collision Handling**
    if collision == "collision":
        if red_center == "high":
            reward += 150  # Task completion reward
        else:
            reward -= 50  # General collision penalty

    # **Navigation Toward Green**
    if red_center == "high" and green_center == "high":
        reward += 150
    elif red_center == "high" and green_center == "medium":
        reward += 120
    elif red_center == "high" and green_center == "low":
        reward += 80
    elif red_center =="high" and (green_left =="high" or green_right =="high"):
        reward +=15
    elif red_center =="high" and (green_left== "medium" or green_right =="medium"):
        reward +=10
    elif red_center =="high" and (green_left== "low" or green_right =="low"):
        reward +=5
   
    if not (red_center == "high" and green_center in ["high", "medium", "low"]):
        reward += -10

    # **Encourage Red-to-Green Transition**
    if collected:
        reward += 200  # Huge bonus for successfully moving red into green

    return reward


def execute_action(rob, action):
    """
    Execute the given action on the robot.
    """
    if action == "forward":
        rob.move_blocking(100, 100, 500)
    elif action == "left":
        rob.move_blocking(-50, -50, 100)
        rob.move_blocking(-10, 10, 1000)
    elif action == "right":
        rob.move_blocking(-50, -50, 100)
        rob.move_blocking(10, -10, 1000)
    elif action == "forward_left":
        rob.move_blocking(80, 100, 500)
    elif action == "forward_right":
        rob.move_blocking(100, 80, 500)
    elif action == "backward":
        rob.move_blocking(-100, -100, 500)
    elif action == "sharp_left":
        rob.move_blocking(-10, 10, 2000)
    elif action == "sharp_right":
        rob.move_blocking(10, -10, 2000)
    else:
        print(f"Unknown action: {action}")


def select_action(state):
    """
    Select an action using ε-greedy policy.
    """
    if state not in Q:
        Q[state] = {action: 0 for action in actions}  # Initialize Q-values for new state

    if random.random() < epsilon:  # Exploration
        return random.choice(actions)
    else:  # Exploitation
        return max(Q[state], key=Q[state].get)


def run_episode(rob, max_duration=60):
    """
    Runs a single episode where the robot learns.
    """
    state = get_discrete_state(rob)  
    action = select_action(state)
    total_reward = 0
    start_time = time.time()
    reward_history = []

    while True:
        execute_action(rob, action)  

        next_state = get_discrete_state(rob)  

        # Extract relevant variables from the state
        green_left, green_center, green_right, red_left, red_center, red_right, collision = next_state

        # Get reward based on the new state
        reward = get_reward(next_state, collected=False)
        #print(reward)

        # Q-learning update rule
        next_action = select_action(next_state)
        Q[state][action] += alpha * (
            reward + gamma * Q[next_state][next_action] - Q[state][action]
        )

        state, action = next_state, next_action
        total_reward += reward
        reward_history.append(total_reward)

        if red_center == "high" and green_center == "high":
            print("[Episode] Red block successfully moved into green area!")
            break

        # Terminate after time limit
        if time.time() - start_time >= max_duration:
            print("[Episode] Time limit reached. Ending episode.")
            break

    duration = time.time() - start_time
    return total_reward, duration, reward_history



def run_sequence(rob, num_episodes=3):
    """
    Train the robot to push the red block into the green area using Q-learning.
    """
    results = []

    for episode in range(num_episodes):
        if isinstance(rob, SimulationRobobo):
            rob.play_simulation()

        rob.set_phone_tilt(106, 50)
        print(f"Starting Episode {episode + 1}")
        total_reward, duration, reward_history = run_episode(rob)
        results.append({
            "episode": episode + 1,
            "duration": duration,
            "cumulative_reward": total_reward,
            "reward_history": reward_history
        })
        print(f"Episode {episode + 1}: Duration = {duration:.2f}s, Cumulative Reward = {total_reward}")

        if isinstance(rob, SimulationRobobo):
            rob.stop_simulation()

    print("\nFinal Q-Table:")
    print(Q)
    return results

