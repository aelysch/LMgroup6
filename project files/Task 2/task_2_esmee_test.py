import cv2
import numpy as np
import random
import time
from robobo_interface import SimulationRobobo

# Initialize parameters
alpha = 0.2  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 0.5  # Exploration rate

# Final Q-table provided
Q = {
    ('far', 'medium', 'far', 'no_collision'): {'forward': 17.501672, 'left': 5, 'right': 5},
    ('far', 'high', 'far', 'no_collision'): {'forward': 18.0, 'left': 5, 'right': 5},
    ('high', 'high', 'far', 'no_collision'): {'forward': 2.0, 'left': 1.0, 'right': 1.0},
    ('far', 'far', 'medium', 'no_collision'): {'forward': 1.9980000000000002, 'left': -1.910794008668277, 'right': 2.076408},
    ('far', 'far', 'high', 'no_collision'): {'forward': 0, 'left': 1.8, 'right': 1.9},
    ('high', 'medium', 'far', 'no_collision'): {'forward': 1.0, 'left': 2.2372039999999997, 'right': 1.0},
    ('far', 'far', 'none', 'no_collision'): {'forward': 5.618559993365655, 'left': 2.8792783745251525, 'right': 3},
    ('medium', 'none', 'none', 'no_collision'): {'forward': 3, 'left': 3.9073193479798078, 'right': 2},
    ('none', 'far', 'medium', 'no_collision'): {'forward': 8.048891648120536, 'left': 1.723981507190484, 'right': 2},
    ('none', 'none', 'high', 'no_collision'): {'forward': -0.2, 'left': 0, 'right': 0.798},
    ('none', 'none', 'none', 'no_collision'): {'forward': -18.738686119993197, 'left': -17.601584749203877, 'right': -17.633137581211827},
    ('far', 'none', 'none', 'no_collision'): {'forward': -0.42037589131297537, 'left': 4.985138054568201, 'right': -2.846686450786729},
    ('far', 'far', 'far', 'no_collision'): {'forward': 4.9584000000000001, 'left': 4.852946034693856, 'right': 2},
    ('none', 'far', 'far', 'no_collision'): {'forward': 2.2945498928062538, 'left': -0.4392564754992545, 'right': 4.0818126746189},
    ('medium', 'far', 'none', 'no_collision'): {'forward': 1.0158819010963736, 'left': 1., 'right': 1},
    ('high', 'none', 'none', 'no_collision'): {'forward': 0.004335999999999985, 'left': 2.3292426377643785, 'right': 1},
    ('medium', 'far', 'far', 'no_collision'): {'forward': 1.0, 'left': 0.5, 'right': 0.5},
    ('high', 'far', 'far', 'no_collision'): {'forward': 1, 'left': 2.1168000000000005, 'right': 1},
    ('none', 'none', 'far', 'no_collision'): {'forward': -2.9132239000333904, 'left': 0.0014341036031102483, 'right': 0.959824321352356},
    ('none', 'none', 'none', 'collision'): {'forward': -7.2, 'left': -2.9359814971311016, 'right': -5.737960648298321},
    ('none', 'medium', 'medium', 'no_collision'): {'forward': 2.0, 'left': -1.0455953583320399, 'right': 1},
    ('none', 'medium', 'high', 'no_collision'): {'forward': -2.5310113065325943, 'left': 0.5, 'right': 0.5},
    ('high', 'high', 'none', 'no_collision'): {'forward': 1.9, 'left': 1.08, 'right': 0.5},
    ('none', 'none', 'medium', 'no_collision'): {'forward': -1.6121991456126947, 'left': -1.2837207628628349, 'right': 1.0},
    ('none', 'far', 'high', 'no_collision'): {'forward': 0, 'left': 0.4355520060882456, 'right': 0.5},
    ('none', 'medium', 'far', 'no_collision'): {'forward': 5, 'left': 0.5, 'right': 4.6512},
    ('far', 'none', 'far', 'no_collision'): {'forward': 0.2, 'left': 2.0, 'right': 1},
    ('medium', 'medium', 'far', 'no_collision'): {'forward': 10.0, 'left': 1, 'right': 1},
    ('far', 'medium', 'none', 'no_collision'): {'forward': 10.0, 'left': 1, 'right': 1},
    ('medium', 'high', 'none', 'no_collision'): {'forward': 2, 'left': 0.2396, 'right': 1},
    ('far', 'high', 'high', 'no_collision'): {'forward': 10.1188, 'left': 1, 'right': 1},
    ('high', 'far', 'none', 'no_collision'): {'forward': 2.0, 'left': 2.3, 'right': 1},
    ('none', 'far', 'none', 'no_collision'): {'forward': 4.620127986355092, 'left': -1.200222029473661, 'right': 1},
}





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


def execute_final_policy(rob, max_duration=180):
    """Run a demonstration using the final Q-table to show off the learned model."""
    rob.set_phone_tilt(103,50)
    state = get_discrete_state(rob)
    total_reward = 0
    collected_blocks = 0
    start_time = time.time()


    print("\n[Demo] Executing final policy using the learned Q-table.")
    while time.time() - start_time < max_duration:
        # Choose the best action for the current state
        if state in Q:
            action = max(Q[state], key=Q[state].get)  # Greedy action
        else:
            action = random.choice(["left", "right"])  # Randomly choose left or right  # Default action if the state is unknown

        # else:
        #     action = "forward"  # Default action if the state is unknown

        print(f"[Action] Executing action: {action}")
        
        # Execute the action
        if action == "forward":
            rob.move_blocking(100, 100, 500)
        elif action == "left":
            rob.move_blocking(-50, -50, 100)
            rob.move_blocking(-10, 10, 1000)
        elif action == "right":
            rob.move_blocking(-50, -50, 100)
            rob.move_blocking(10, -10, 1000)

        # Collision detection
        collision_detected = rob.read_irs()[4] > 10000

        # Get the next state
        next_state = get_discrete_state(rob)
        collected = collision_detected and next_state[3] == "high"

        # Track collected blocks
        if collected:
            collected_blocks += 1
            print(f"[Demo] Collected green box! Total collected: {collected_blocks}")

        state = next_state

    print(f"[Demo] Total Reward: {total_reward}, Blocks Collected: {collected_blocks}")
    return total_reward, collected_blocks


def run_demo(rob):
    """Set up the robot and run the demonstration."""
    rob = SimulationRobobo()
    rob.play_simulation()
    
    try:
        total_reward, collected_blocks = execute_final_policy(rob, max_duration=180)
        print(f"Demo Finished: Total Reward = {total_reward}, Blocks Collected = {collected_blocks}")
    finally:
        rob.stop_simulation()

