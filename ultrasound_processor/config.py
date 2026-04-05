"""Configuration settings for ultrasound bubble processing."""

class Config:
    """Default configuration parameters."""
    
    # Signal processing
    DEFAULT_SAMPLING_RATE = 1000  # Hz
    
    # Filter settings
    BUTTERWORTH_ORDER = 4
    DEFAULT_LOW_CUTOFF = 100  # Hz
    DEFAULT_HIGH_CUTOFF = 150  # Hz
    
    # Image processing
    IMAGE_SIZE = (200, 200)
    GAUSSIAN_SIGMA = 2
    MIN_DISTANCE_PEAKS = 10
    THRESHOLD_ABS = 0.2
    
    # Tracking
    MAX_BUBBLE_DISPLACEMENT = 3  # pixels
    NUM_FRAMES = 20
    NUM_BUBBLES = 10
    
    # Visualization
    FIGSIZE = (12, 6)
    DPI = 100
