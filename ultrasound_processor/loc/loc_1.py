"""
Localization module for ultrasound bubble detection.

This module provides various algorithms for detecting and localizing
microbubbles in ultrasound images, including Gaussian filtering,
peak detection, and morphological operations.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, median_filter
from skimage.feature import peak_local_max
from typing import Tuple, List, Optional


def detect_bubbles_gaussian(
    image: np.ndarray,
    sigma: float = 2.0,
    min_distance: int = 10,
    threshold_abs: float = 0.2
) -> np.ndarray:
    """
    Detect bubbles using Gaussian filtering and peak detection.
    
    Args:
        image: Input ultrasound image
        sigma: Standard deviation for Gaussian filter
        min_distance: Minimum distance between peaks
        threshold_abs: Absolute intensity threshold
        
    Returns:
        Array of (row, col) coordinates of detected bubbles
    """
    # Apply Gaussian filter to simulate PSF
    filtered_image = gaussian_filter(image, sigma=sigma)
    
    # Detect local maxima
    coordinates = peak_local_max(
        filtered_image,
        min_distance=min_distance,
        threshold_abs=threshold_abs
    )
    
    return coordinates


def detect_bubbles_maximum_filter(
    image: np.ndarray,
    sigma: float = 2.0,
    filter_size: int = 20,
    threshold: float = 0.2
) -> np.ndarray:
    """
    Detect bubbles using maximum filter for local maxima detection.
    
    Args:
        image: Input ultrasound image
        sigma: Standard deviation for Gaussian pre-filtering
        filter_size: Size of the maximum filter window
        threshold: Intensity threshold
        
    Returns:
        Array of (row, col) coordinates of detected bubbles
    """
    # Apply Gaussian pre-filtering
    filtered_image = gaussian_filter(image, sigma=sigma)
    
    # Detect local maxima using maximum filter
    local_max = (filtered_image == maximum_filter(filtered_image, size=filter_size)) & \
                (filtered_image > threshold)
    coordinates = np.argwhere(local_max)
    
    return coordinates


def detect_bubbles_median(
    image: np.ndarray,
    filter_size: int = 3,
    max_filter_size: int = 20,
    threshold: float = 0.2
) -> np.ndarray:
    """
    Detect bubbles using median filtering for noise removal.
    
    Args:
        image: Input ultrasound image
        filter_size: Size of median filter window
        max_filter_size: Size of maximum filter window
        threshold: Intensity threshold
        
    Returns:
        Array of (row, col) coordinates of detected bubbles
    """
    # Apply median filter to remove noise
    filtered_image = median_filter(image, size=filter_size)
    
    # Detect local maxima
    local_max = (filtered_image == maximum_filter(filtered_image, size=max_filter_size)) & \
                (filtered_image > threshold)
    coordinates = np.argwhere(local_max)
    
    return coordinates


def detect_bubbles_adaptive(
    image: np.ndarray,
    median_size: int = 3,
    gaussian_sigma: float = 1.0,
    std_multiplier: float = 1.5,
    min_area: int = 10,
    max_area: int = 100
) -> np.ndarray:
    """
    Detect bubbles using adaptive thresholding and connected component analysis.
    
    Args:
        image: Input ultrasound image
        median_size: Size of median filter window
        gaussian_sigma: Standard deviation for Gaussian smoothing
        std_multiplier: Multiplier for standard deviation in threshold calculation
        min_area: Minimum area for valid regions
        max_area: Maximum area for valid regions
        
    Returns:
        Array of (row, col) coordinates of detected bubble centroids
    """
    from scipy.ndimage import label
    from skimage.measure import regionprops
    
    # Apply median filter
    filtered_image = median_filter(image, size=median_size)
    
    # Apply Gaussian smoothing
    smoothed_image = gaussian_filter(filtered_image, sigma=gaussian_sigma)
    
    # Adaptive thresholding
    threshold = np.mean(smoothed_image) + std_multiplier * np.std(smoothed_image)
    binary_image = smoothed_image > threshold
    
    # Connected component analysis
    labeled_image, num_features = label(binary_image)
    regions = regionprops(labeled_image)
    
    # Filter by area and extract centroids
    coordinates = np.array([
        region.centroid for region in regions
        if min_area < region.area < max_area
    ])
    
    return coordinates


def detect_bubbles_morphological(
    image: np.ndarray,
    canny_sigma: float = 1.0,
    otsu_threshold: bool = True,
    min_size: int = 20
) -> np.ndarray:
    """
    Detect bubbles using morphological operations and edge detection.
    
    Args:
        image: Input ultrasound image
        canny_sigma: Sigma for Canny edge detection
        otsu_threshold: Whether to use Otsu thresholding
        min_size: Minimum object size to keep
        
    Returns:
        Array of (row, col) coordinates of detected bubble centroids
    """
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects, disk, closing
    from skimage.measure import regionprops, label
    from skimage.feature import canny
    from scipy.ndimage import binary_fill_holes
    
    # Canny edge detection
    edges = canny(image, sigma=canny_sigma)
    
    # Fill edges
    filled_edges = binary_fill_holes(edges)
    
    # Otsu thresholding
    if otsu_threshold:
        thresh_val = threshold_otsu(image)
        binary_image = image > thresh_val
    else:
        binary_image = image > np.mean(image)
    
    # Morphological closing
    closed_image = closing(binary_image, disk(3))
    cleaned_image = remove_small_objects(closed_image, min_size=min_size)
    
    # Connected component analysis
    labeled_image, num_features = label(cleaned_image, return_num=True)
    regions = regionprops(labeled_image)
    
    # Extract centroids with area filtering
    coordinates = np.array([
        region.centroid for region in regions
        if 10 < region.area < 100
    ])
    
    return coordinates