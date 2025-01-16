import cv2
import json
import matplotlib.pyplot as plt
import os
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

# Pre-trained Q-table (final version)
Q = {
    ('far', 'far', 'far'): {'forward': 50.00619347840659, 'left': 34.8805043027467, 'right': 29.47236442484463, 'backward': 30.535540949380177},
    ('medium', 'medium', 'far'): {'forward': -0.5, 'left': 9.62742981915827, 'right': 0, 'backward': 0},
    ('close', 'close', 'far'): {'forward': -0.95, 'left': -0.5, 'right': 0, 'backward': 0.1},
    ('close', 'far', 'close'): {'forward': -0.95, 'left': 21.29096896453334, 'right': 0, 'backward': -0.40595000000000003},
    ('close', 'medium', 'close'): {'forward': -1.0, 'left': 2.7600472400080496, 'right': 0, 'backward': 0},
    ('collision', 'close', 'collision'): {'forward': -1.9, 'left': 2.34264464993883, 'right': 0, 'backward': 0},
    ('collision', 'close', 'close'): {'forward': -1.9, 'left': 0.5, 'right': 0, 'backward': 0},
    ('far', 'far', 'close'): {'forward': 22.48180890640738, 'left': 9.22787244528828, 'right': -0.5, 'backward': 4.992718135269587},
    ('close', 'far', 'far'): {'forward': -0.5, 'left': 0.64394158955, 'right': 0, 'backward': 0},
    ('close', 'medium', 'far'): {'forward': 0, 'left': 0, 'right': 0.5, 'backward': 0},
    ('far', 'medium', 'far'): {'forward': 28.940932615138042, 'left': 0.23928912530166835, 'right': 1.5944051871877076, 'backward': 6.0475192097805435},
    ('far', 'close', 'far'): {'forward': 1.860806691704092, 'left': 0, 'right': 15.925713427330471, 'backward': 0},
    ('close', 'collision', 'far'): {'forward': -1.9, 'left': 0, 'right': 0, 'backward': 0},
    ('close', 'collision', 'medium'): {'forward': -1.0, 'left': 0, 'right': 0, 'backward': 0},
    ('collision', 'collision', 'close'): {'forward': -1.0, 'left': -0.5, 'right': 0, 'backward': -0.5},
    ('close', 'close', 'medium'): {'forward': -0.5, 'left': 3.5090931248982047, 'right': 0, 'backward': 0},
    ('close', 'close', 'close'): {'forward': -1.0, 'left': 0.4499694893990902, 'right': 0, 'backward': 0},
    ('far', 'far', 'medium'): {'forward': 41.294345318318605, 'left': 21.60129908053735, 'right': 18.65074505331191, 'backward': 28.822537085717325},
    ('collision', 'collision', 'far'): {'forward': -1.0, 'left': -0.4505, 'right': 0, 'backward': 0},
    ('collision', 'close', 'far'): {'forward': -1.0, 'left': 0, 'right': 0, 'backward': 0},
    ('medium', 'far', 'far'): {'forward': -0.5, 'left': 33.09307203528387, 'right': 0, 'backward': 4.387160077547462},
    ('far', 'far', 'collision'): {'forward': -1.904431299051454, 'left': 16.988613588386336, 'right': 0, 'backward': 0},
    ('close', 'far', 'collision'): {'forward': -1.9, 'left': 4.599524240463147, 'right': -0.26014847259844154, 'backward': 0},
    ('medium', 'close', 'far'): {'forward': -0.5, 'left': 5.3275557510971, 'right': 0, 'backward': 0},
    ('medium', 'far', 'medium'): {'forward': 0.01744416281042427, 'left': 32.607483098027906, 'right': 0, 'backward': 5.359163720116641},
    ('close', 'medium', 'medium'): {'forward': -0.5, 'left': 8.659192048204318, 'right': 0, 'backward': 0},
    ('medium', 'far', 'close'): {'forward': -0.33510554307573187, 'left': 34.58461566779879, 'right': 0, 'backward': 1.82915346015186},
    ('medium', 'far', 'collision'): {'forward': -0.9901, 'left': 4.5875049413329645, 'right': 0, 'backward': 0},
    ('far', 'medium', 'medium'): {'forward': 3.355170942553941, 'left': 0, 'right': 0, 'backward': 0},
    ('medium', 'close', 'medium'): {'forward': -0.3955033561661001, 'left': 0, 'right': 0, 'backward': 0},
    ('close', 'far', 'medium'): {'forward': -0.24426094290781558, 'left': 0, 'right': 0, 'backward': 0},
    ('medium', 'medium', 'close'): {'forward': 0, 'left': 2.6970993937011145, 'right': 0, 'backward': 17.31128379952983},
    ('far', 'medium', 'close'): {'forward': 7.0197860196886905, 'left': 0, 'right': 0, 'backward': 0},
    ('medium', 'medium', 'medium'): {'forward': 0.31979181831906545, 'left': 0, 'right': 0, 'backward': 0}
}

# Action space
actions = ["forward", "left", "right", "backward"]

def get_discrete_state(rob):
    """Get the discrete state from the robot's IR sensor readings."""
    def state_ir_value(ir_value):
        if ir_value < 50:
            return "far"
        elif 50 <= ir_value < 100:
            return "medium"
        elif 100 <= ir_value <= 10000:
            return "close"
        elif ir_value > 10000:
            return "collision"
    
    front_ir = rob.read_irs()[4]
    #left_ir = max(rob.read_irs()[2], rob.read_irs()[7]) 
    #right_ir = max(rob.read_irs()[3], rob.read_irs()[5])
    left_ir =  rob.read_irs()[7]
    right_ir =rob.read_irs()[5]

    front_state = state_ir_value(front_ir)
    left_state = state_ir_value(left_ir)
    right_state = state_ir_value(right_ir)

    return (front_state, left_state, right_state)

def select_best_action(state):
    """Select the best action based on the pre-trained Q-table."""
    if state in Q:
        return max(Q[state], key=Q[state].get)  # Choose the action with the highest Q-value
    else:
        return "forward"  # Default action if the state is unknown


def run_final_demo(rob, max_steps=500):
    """Run the robot using the final pre-trained Q-table."""
    state = get_discrete_state(rob)
    total_reward = 0

    for step in range(max_steps):
        action = select_best_action(state)

        # Execute the selected action
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

        # Update state and accumulate reward
        state = get_discrete_state(rob)
        reward = get_reward(state)
        total_reward += reward

        # if reward == -10:  # Stop if collision occurs
        #     print("Collision detected! Ending demo.")
        #     break

    print(f"Total Reward for Final Demo Run: {total_reward}")

def get_reward(state):
    """Reward function prioritizing forward movement and penalizing collisions."""
    front, left, right = state

    if front == "collision" or left == "collision" or right == "collision":
        return -10  
    elif front == "close":
        return -5  
    elif front == "medium":
        return 1
    else:  
        return 5


