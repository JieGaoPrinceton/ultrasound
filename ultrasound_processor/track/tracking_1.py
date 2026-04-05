"""
Tracking module for ultrasound microbubble tracking.

This module provides algorithms for tracking microbubbles across
ultrasound image sequences, including Hungarian algorithm-based
tracking and deep learning approaches.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops
from typing import List, Tuple, Optional


def detect_bubbles_in_frame(
    frame: np.ndarray,
    threshold: float = 0.1,
    sigma: float = 2.0
) -> List[np.ndarray]:
    """
    Detect bubbles in a single frame using thresholding and connected components.
    
    Args:
        frame: Input image frame
        threshold: Intensity threshold for binarization
        sigma: Standard deviation for Gaussian pre-filtering
        
    Returns:
        List of bubble centroid coordinates
    """
    # Apply Gaussian filter
    filtered_frame = gaussian_filter(frame, sigma=sigma)
    
    # Binary thresholding
    binary_frame = filtered_frame > threshold
    binary_frame = binary_frame.astype(np.uint8)
    
    # Connected component analysis
    labeled_frame, num_features = label(binary_frame, return_num=True)
    regions = regionprops(labeled_frame)
    
    # Extract centroids
    detected_positions = [region.centroid for region in regions]
    
    return detected_positions


def compute_cost_matrix(
    tracked_positions: List[np.ndarray],
    detected_positions: List[np.ndarray]
) -> np.ndarray:
    """
    Compute cost matrix based on Euclidean distances.
    
    Args:
        tracked_positions: List of previously tracked positions
        detected_positions: List of newly detected positions
        
    Returns:
        Cost matrix where element (i,j) is the distance between
        tracked position i and detected position j
    """
    cost_matrix = np.zeros((len(tracked_positions), len(detected_positions)))
    
    for i, track in enumerate(tracked_positions):
        for j, detect in enumerate(detected_positions):
            cost_matrix[i, j] = np.linalg.norm(np.array(track) - np.array(detect))
    
    return cost_matrix


def track_bubbles_hungarian(
    frames: List[np.ndarray],
    initial_threshold: float = 0.1,
    sigma: float = 2.0
) -> List[List[np.ndarray]]:
    """
    Track bubbles across frames using the Hungarian algorithm.
    
    Args:
        frames: List of image frames
        initial_threshold: Threshold for bubble detection
        sigma: Standard deviation for Gaussian filtering
        
    Returns:
        List of tracked positions for each frame
    """
    all_tracked_positions = []
    tracked_positions = []
    
    for frame_idx, frame in enumerate(frames):
        # Detect bubbles in current frame
        detected_positions = detect_bubbles_in_frame(frame, initial_threshold, sigma)
        
        if frame_idx == 0:
            # Initialize with first frame detections
            tracked_positions = detected_positions
        else:
            if len(tracked_positions) > 0 and len(detected_positions) > 0:
                # Compute cost matrix
                cost_matrix = compute_cost_matrix(tracked_positions, detected_positions)
                
                # Hungarian algorithm for optimal assignment
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                # Update tracked positions with matched detections
                updated_positions = []
                for i, j in zip(row_ind, col_ind):
                    updated_positions.append(detected_positions[j])
                tracked_positions = updated_positions
        
        all_tracked_positions.append(tracked_positions.copy())
    
    return all_tracked_positions


def generate_synthetic_sequence(
    num_frames: int = 20,
    image_size: Tuple[int, int] = (200, 200),
    num_bubbles: int = 10,
    max_displacement: int = 3,
    noise_level: float = 0.02,
    seed: Optional[int] = None
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Generate a synthetic ultrasound sequence with moving bubbles.
    
    Args:
        num_frames: Number of frames to generate
        image_size: Size of each frame (height, width)
        num_bubbles: Number of bubbles to simulate
        max_displacement: Maximum pixel displacement per frame
        noise_level: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (frames_with_noise, true_trajectories)
        where true_trajectories has shape (num_bubbles, num_frames, 2)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize bubble trajectories
    bubble_trajectories = [
        np.array([np.random.randint(0, min(image_size), size=2)])
        for _ in range(num_bubbles)
    ]
    
    frames = []
    for frame_idx in range(num_frames):
        frame = np.zeros(image_size)
        new_positions = []
        
        for idx, bubble in enumerate(bubble_trajectories):
            # Simulate random movement
            new_position = bubble[-1] + np.random.randint(-max_displacement, max_displacement + 1, size=2)
            new_position = np.clip(new_position, 0, min(image_size) - 1)
            new_positions.append(new_position)
            bubble_trajectories[idx] = np.vstack([bubble, new_position])
            
            # Add bubble signal to frame
            x, y = new_position
            frame[x, y] = 1
        
        frames.append(frame)
    
    # Apply Gaussian filter and add noise
    frames = [gaussian_filter(frame, sigma=2) + noise_level * np.random.rand(*image_size) for frame in frames]
    
    # Stack trajectories into array
    true_trajectories = np.stack([np.stack(traj) for traj in bubble_trajectories], axis=0)
    
    return frames, true_trajectories
