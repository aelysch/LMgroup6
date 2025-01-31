import cv2
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from filterpy.kalman import KalmanFilter
from robobo_interface import IRobobo, SimulationRobobo

# Initialize parameters
alpha = 0.001  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32
memory_size = 10000
target_update = 10

# Action space
actions = ["forward", "left", "right"]

# Neural Network for DQN
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Initialize networks
input_dim = 7  # green_left, green_center, green_right, red_left, red_center, red_right
output_dim = len(actions)
policy_net = DQN(input_dim, output_dim)
target_net = DQN(input_dim, output_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=alpha)
criterion = nn.MSELoss()

# Replay Memory
memory = deque(maxlen=memory_size)

def initialize_kalman():
    kf = KalmanFilter(dim_x=1, dim_z=1)  # 1D state (percentage of color)
    kf.x = np.array([[0]])  # Initial state estimate
    kf.F = np.array([[1]])  # State transition matrix
    kf.H = np.array([[1]])  # Measurement function
    kf.P *= 1000  # Initial uncertainty
    kf.R = 10  # Measurement noise covariance
    kf.Q = 1  # Process noise covariance
    return kf

green_kf = [initialize_kalman() for _ in range(3)]  # Left, Center, Right
red_kf = [initialize_kalman() for _ in range(3)]  # Left, Center, Right

def apply_kalman_filter(kf_list, measurements):
    filtered_values = []
    for i, kf in enumerate(kf_list):
        kf.predict()
        kf.update(measurements[i])
        filtered_values.append(kf.x[0, 0])
    return filtered_values

def preprocess_camera_data(rob):
    """
    Detect green areas in the camera feed and divide the screen into left, center, and right sections.
    Returns the percentage of green in each section.
    """
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
    green_left_section = mask[:, :width // 3]
    green_center_section = mask[:, width // 3:2 * width // 3]
    green_right_section = mask[:, 2 * width // 3:]

    # Calculate the total pixels and green pixel counts for each section
    green_left = green_left_section.sum() // 255
    green_center = green_center_section.sum() // 255
    green_right = green_right_section.sum() // 255

    # Calculate the percentage of green pixels in each section
    section_pixels = (width // 3) * height
    green_measurements = [(green_left / section_pixels) * 100, (green_center / section_pixels) * 100, (green_right / section_pixels) * 100]

    green_filtered_values = apply_kalman_filter(green_kf, green_measurements)
    print(f"[Camera Data] Green percentage - Left: {green_filtered_values[0]:.2f}%, Center: {green_filtered_values[1]:.2f}%, Right: {green_filtered_values[2]:.2f}%")

    return {
        "green_left": green_filtered_values[0],
        "green_center": green_filtered_values[1],
        "green_right": green_filtered_values[2]
    }

def detect_red_block(rob):
    """
    Detect the red block in the camera feed.
    Returns the percentage of red in each section (left, center, right).
    """
    image = rob.read_image_front()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV range for red
    lower_red1 = np.array([0, 30, 30], dtype=np.uint8)
    upper_red1 = np.array([15, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([165, 30, 30], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    # Create masks for red pixels
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Get the image dimensions
    height, width = mask.shape

    # Divide the screen into three sections: left, center, and right
    red_left_section = mask[:, :width // 3]
    red_center_section = mask[:, width // 3:2 * width // 3]
    red_right_section = mask[:, 2 * width // 3:]

    # Calculate the total pixels and red pixel counts for each section
    red_left = red_left_section.sum() // 255
    red_center = red_center_section.sum() // 255
    red_right = red_right_section.sum() // 255

    # Calculate the percentage of red pixels in each section
    section_pixels = (width // 3) * height
    red_measurements = [(red_left / section_pixels) * 100, (red_center / section_pixels) * 100, (red_right / section_pixels) * 100]

    red_filtered_values = apply_kalman_filter(red_kf, red_measurements)
    print(f"[Red Block] Red percentage - Left: {red_filtered_values[0]:.2f}%, Center: {red_filtered_values[1]:.2f}%, Right: {red_filtered_values[2]:.2f}%")

    return {
        "red_left": red_filtered_values[0],
        "red_center": red_filtered_values[1],
        "red_right": red_filtered_values[2]
    }

def get_discrete_state(rob):
    """
    Get the discrete state based on the percentage of green and red pixels in left, center, and right sections.
    """

    # Detect green area
    green_data = preprocess_camera_data(rob)
    # green_left_percentage, green_center_percentage, green_right_percentage = preprocess_camera_data(rob)

    # Detect red block
    red_data = detect_red_block(rob)
    # red_left_percentage, red_center_percentage, red_right_percentage = detect_red_block(rob)

    # Classify green and red presence
    def classify_presence(percentage):
        if percentage >= 10:  # High presence
            return "high"
        elif percentage >= 5:  # Medium presence
            return "medium"
        elif percentage >= 1:  # Far presence
            return "far"
        else:
            return "none"
    
    green_left_state = classify_presence(green_data["green_left"])
    green_center_state = classify_presence(green_data["green_center"])
    green_right_state = classify_presence(green_data["green_right"])

    red_left_state = classify_presence(red_data["red_left"])
    red_center_state = classify_presence(red_data["red_center"])
    red_right_state = classify_presence(red_data["red_right"])

    # green_left_state = classify_presence(green_left_percentage)
    # green_center_state = classify_presence(green_center_percentage)
    # green_right_state = classify_presence(green_right_percentage)

    # red_left_state = classify_presence(red_left_percentage)
    # red_center_state = classify_presence(red_center_percentage)
    # red_right_state = classify_presence(red_right_percentage)

    # Collision detection
    ir_values = rob.read_irs()
    collision_detected = any(ir_values[i] > 10000 for i in [2, 3, 5, 7])  # Check if any of the front side sensors detect a collision
    collision_state = "collision" if collision_detected else "no_collision"

    return (green_left_state, green_center_state, green_right_state,
            red_left_state, red_center_state, red_right_state, collision_state)

def state_to_tensor(state):
    """
    Convert state to a tensor for the neural network.
    """
    state_mapping = {"none": 0, "far": 1, "medium": 2, "high": 3
                     , "collision": 1, "no_collision": 0
                     }
    return torch.tensor([state_mapping[s] for s in state], dtype=torch.float32)

def select_action(state, epsilon):
    """
    Select an action.
    """
    if random.random() < epsilon:
        return random.choice(actions)
    else:
        with torch.no_grad():
            state_tensor = state_to_tensor(state)
            q_values = policy_net(state_tensor)
            return actions[torch.argmax(q_values).item()]

def get_reward(state):
    """
    Calculate the reward based on the state.
    """
    green_left, green_center, green_right, red_left, red_center, red_right, collision = state

    # Reward for reaching the green area with the red block
    if green_center == "high" and red_center == "high":
        return 20  # Success: red block is in the green area

    # Penalty for colliding with walls
    if collision == "collision":
        return -5

    # Reward for pushing the red block toward the green area
    elif red_center == "high" and green_center == "medium":
        return 15
    elif red_center == "high" and green_center == "far":
        return 10

    # Reward for moving toward the red block
    if red_center == "high":
        return 5
    elif red_center == "medium":
        return 3
    elif red_center == "far":
        return 1

    if red_left == "high" or red_right == "high":
        return 3
    elif red_left == "medium" or red_right == "medium":
        return 2
    elif red_left == "far" or red_right == "far":
        return 1
    
    # Reward for moving toward the green block with the red block
    if green_left == "high" and red_center == "high":
        return 8
    elif green_left == "medium" and red_center == "high":
        return 6
    elif green_right == "high" and red_center == "high":
        return 8
    elif green_right == "medium" and red_center == "high":
        return 6
    
    if (green_left == "high" or green_center == "high" or green_right == "high") and (red_left != "high" and red_center != "high" and red_right != "high"):
        return -3
    elif (green_left == "medium" or green_center == "medium" or green_right == "medium") and (red_left != "high" and red_center != "high" and red_right != "high"):
        return -2
    elif (green_left == "far" or green_center == "far" or green_right == "far") and (red_left != "high" and red_center != "high" and red_right != "high"):
        return -1

    # Small penalty for no progress
    return -5

def run_episode(rob, max_duration=120):
    """Run a single episode to guide the robot."""
    state = get_discrete_state(rob)
    total_reward = 0
    start_time = time.time()

    while True:
        action = select_action(state, epsilon)
        if action == "forward":
            rob.move_blocking(50, 50, 500)
        elif action == "left":
            rob.move_blocking(-30, -30, 100)
            rob.move_blocking(-5, 5, 1000)
        elif action == "right":
            rob.move_blocking(-30, -30, 100)
            rob.move_blocking(5, -5, 1000)

        next_state = get_discrete_state(rob)
        reward = get_reward(next_state)
        done = time.time() - start_time >= max_duration

        state = next_state
        total_reward += reward

        if done:
            print("[Episode] Time limit reached. Ending episode.")
            break

    duration = time.time() - start_time
    print(f"[Episode] Total Reward: {total_reward}")
    return total_reward, duration

def run_training(rob, num_episodes=30):
    """Train the robot for pushing using DQN."""
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    rob.set_phone_tilt(95, 50)

    global epsilon
    results = []

    for episode in range(num_episodes):
        print(f"Starting Episode {episode + 1}")
        total_reward, duration = run_episode(rob)
        results.append({"episode": episode + 1, "duration": duration, "total_reward": total_reward})
        print(f"Episode {episode + 1}: Duration = {duration:.2f}s, Total Reward = {total_reward}")

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if (episode + 1) % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
    
    torch.save(policy_net.state_dict(), "/root/results/dqn_model.pth")
    print("Model saved as dqn_model.pth")

    print("\nResults:")
    for result in results:
        print(f"Episode {result['episode']}: Duration = {result['duration']:.2f}s, Total Reward = {result['total_reward']}")

# Load the trained model
def load_model(model_path, input_dim, output_dim):
    model = DQN(input_dim, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    return model

def run_test_episode(rob, max_duration=180):
    """Run a test episode using the trained model."""
    rob.set_phone_tilt(97, 50)

    model_path = "/root/results/dqn_model.pth"
    input_dim = 7  # Adjust if state encoding changes
    output_dim = 3  # Number of actions (forward, left, right)
    model = load_model(model_path, input_dim, output_dim)
    print("Model loaded successfully!")

    state = get_discrete_state(rob)
    total_reward = 0
    start_time = time.time()

    while True:
        with torch.no_grad():
            state_tensor = state_to_tensor(state)
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()
        
        # Map action index to command
        if action == 0:
            rob.move_blocking(50, 50, 500)
        elif action == 1:
            rob.move_blocking(-50, 50, 100)
        elif action == 2:
            rob.move_blocking(50, -50, 100)
        
        next_state = get_discrete_state(rob)
        state = next_state
        duration = time.time() - start_time

        if duration >= max_duration:
            print("[Test Run] Time limit reached. Ending episode.")
            break
    
    print("[Test Run] Completed.")

def run_task_3(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    # run_training(rob)
    run_test_episode(rob)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()