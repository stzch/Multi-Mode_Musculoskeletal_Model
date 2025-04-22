

import numpy as np
import matplotlib.pyplot as plt
import time

# Function to parse the hierarchy and offsets
def parse_hierarchy(file_path):
    hierarchy = {}
    with open(file_path, 'r') as f:
        data = f.readlines()
    
    joint_stack = []
    current_joint = None
    
    for line in data:
        if "ROOT" in line or "JOINT" in line:
            joint_name = line.split()[1]
            joint_stack.append(joint_name)
            hierarchy[joint_name] = {'parent': current_joint, 'children': [], 'offset': None}
            if current_joint is not None:
                hierarchy[current_joint]['children'].append(joint_name)
            current_joint = joint_name
        
        elif "End Site" in line:
            joint_stack.pop()  # Move back up the hierarchy
        
        elif "OFFSET" in line:
            offset = [float(x) for x in line.split()[1:]]
            hierarchy[current_joint]['offset'] = offset
        
        elif "{" in line or "}" in line:
            if "}" in line:
                current_joint = joint_stack.pop() if joint_stack else None
        
        if "MOTION" in line:
            break

    return hierarchy

# Function to parse motion data
def parse_motion_data(file_path):
    with open(file_path, 'r') as f:
        data = f.readlines()

    # Extract motion data
    motion_start = data.index('MOTION\n') + 3  # Skipping 'Frames:' and 'Frame Time'
    motion_data = data[motion_start:]
    
    # Convert the motion data into a numpy array for easy manipulation
    motion = []
    for line in motion_data:
        motion.append([float(x) for x in line.strip().split()])

    return np.array(motion)

# Function to calculate joint positions based on hierarchy and motion data
def calculate_joint_positions(hierarchy, motion_frame):
    positions = {}
    
    def recurse_joint(joint_name, parent_pos):
        offset = np.array(hierarchy[joint_name]['offset'])
        position = parent_pos + offset  # Offset from parent
        
        positions[joint_name] = position
        
        # Recurse for children
        for child in hierarchy[joint_name]['children']:
            recurse_joint(child, position)
    
    # Start from ROOT and calculate positions recursively
    root_pos = np.array([motion_frame[0], motion_frame[1], motion_frame[2]])  # Xposition, Yposition, Zposition
    root_joint = list(hierarchy.keys())[0]
    recurse_joint(root_joint, root_pos)
    
    return positions

# Plot the skeleton based on joint positions
def plot_skeleton(positions, ax):
    # Example of connecting joints based on their names
    connections = [
        ('Character1_Hips', 'Character1_Spine'),
        ('Character1_Spine', 'Character1_Spine1'),
        ('Character1_Spine1', 'Character1_LeftShoulder'),
        ('Character1_LeftShoulder', 'Character1_LeftArm'),
        ('Character1_LeftArm', 'Character1_LeftForeArm'),
        ('Character1_LeftForeArm', 'Character1_LeftHand'),
        ('Character1_Spine1', 'Character1_RightShoulder'),
        ('Character1_RightShoulder', 'Character1_RightArm'),
        ('Character1_RightArm', 'Character1_RightForeArm'),
        ('Character1_RightForeArm', 'Character1_RightHand'),
        # Add more connections based on hierarchy...
    ]
    
    for joint1, joint2 in connections:
        pos1 = positions[joint1]
        pos2 = positions[joint2]
        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'ro-')

# Main function to animate the skeleton
def animate_skeleton(file_path):
    hierarchy = parse_hierarchy(file_path)
    motion_data = parse_motion_data(file_path)
    
    fig, ax = plt.subplots()
    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)
    
    for frame in motion_data:
        ax.clear()
        joint_positions = calculate_joint_positions(hierarchy, frame)
        plot_skeleton(joint_positions, ax)
        plt.pause(0.05)  # Pause for animation effect
        time.sleep(0.05)
    
    plt.show()


# Path to your motion file
#file_path = '../data/motion/run.txt'
file_path = '/home/chen/BidirectionalGaitNet/data/motion/run.txt'
# Run the animation
animate_skeleton(file_path)
