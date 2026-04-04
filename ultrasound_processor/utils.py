"""
Utility functions for ultrasound bubble processing.
"""

import numpy as np
from typing import Tuple, List, Optional


def generate_test_signal(
    sampling_rate: int = 1000,
    duration: float = 1.0,
    frequencies: List[float] = None,
    noise_level: float = 0.3,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a test ultrasound signal with specified frequencies and noise.
    
    Args:
        sampling_rate: Sampling rate in Hz
        duration: Signal duration in seconds
        frequencies: List of frequencies to include in the signal
        noise_level: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (time_array, signal_array)
    """
    if seed is not None:
        np.random.seed(seed)
    
    if frequencies is None:
        frequencies = [50, 200]
    
    t = np.linspace(0, duration, int(sampling_rate * duration))
    signal = np.zeros_like(t)
    
    for freq in frequencies:
        signal += np.sin(2 * np.pi * freq * t)
    
    signal += noise_level * np.random.randn(len(t))
    
    return t, signal


def generate_test_image(
    image_size: Tuple[int, int] = (200, 200),
    num_bubbles: int = 10,
    noise_level: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a test ultrasound image with simulated bubbles.
    
    Args:
        image_size: Size of the image (height, width)
        num_bubbles: Number of bubbles to simulate
        noise_level: Level of Gaussian noise to add
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (image_array, bubble_positions)
    """
    if seed is not None:
        np.random.seed(seed)
    
    image = np.zeros(image_size)
    min_dim = min(image_size)
    bubble_positions = np.random.randint(0, min_dim, size=(num_bubbles, 2))
    
    for pos in bubble_positions:
        image[pos[0], pos[1]] = 1
    
    image += noise_level * np.random.rand(*image_size)
    
    return image, bubble_positions


def calculate_distance(
    point1: np.ndarray,
    point2: np.ndarray
) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: First point coordinates
        point2: Second point coordinates
        
    Returns:
        Euclidean distance
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))


def create_cost_matrix(
    tracked_positions: List[np.ndarray],
    detected_positions: List[np.ndarray]
) -> np.ndarray:
    """
    Create a cost matrix based on Euclidean distances.
    
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
            cost_matrix[i, j] = calculate_distance(track, detect)
    
    return cost_matrix


def validate_coordinates(
    coordinates: np.ndarray,
    image_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Validate that coordinates are within image bounds.
    
    Args:
        coordinates: Array of (row, col) coordinates
        image_shape: Shape of the image (height, width)
        
    Returns:
        Filtered array of valid coordinates
    """
    if len(coordinates) == 0:
        return coordinates
    
    valid_mask = (
        (coordinates[:, 0] >= 0) & 
        (coordinates[:, 0] < image_shape[0]) &
        (coordinates[:, 1] >= 0) & 
        (coordinates[:, 1] < image_shape[1])
    )
    
    return coordinates[valid_mask]
