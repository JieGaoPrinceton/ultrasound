# Ultrasound Bubble Processing

A Python package for processing ultrasound microbubble signals, including filtering, localization, and tracking modules.

## Project Structure

```
ultrasound_processor/
├── __init__.py          # Package initialization
├── config.py            # Configuration settings
├── utils.py             # Utility functions
├── filter/              # Signal filtering module
│   ├── __init__.py
│   ├── fil_1.py         # Butterworth, Chebyshev, FIR, Kalman filters
│   └── fil_2.py         # Advanced filtering with spectral analysis
├── loc/                 # Bubble localization module
│   ├── __init__.py
│   └── loc_1.py         # Multiple detection algorithms
└── track/               # Bubble tracking module
    ├── __init__.py
    ├── tracking_1.py    # Hungarian algorithm tracking
    ├── tracking_dl.py   # Deep learning tracker (ConvLSTM)
    └── gui_demo.py      # Streamlit GUI demo
```

## Features

### Filter Module
- Butterworth low-pass and high-pass filters
- Chebyshev Type I filters with configurable ripple
- FIR filters using window method
- 1D Kalman filter for signal smoothing
- Power spectral density analysis using Welch's method

### Localization Module
- Gaussian filtering with peak detection
- Maximum filter-based local maxima detection
- Median filtering for noise removal
- Adaptive thresholding with connected component analysis
- Morphological operations with Canny edge detection
- Otsu thresholding

### Tracking Module
- Hungarian algorithm for optimal bubble matching
- Connected component analysis for bubble detection
- Synthetic sequence generation for testing
- Deep learning-based tracking using ConvLSTM (tracking_dl.py)
- Interactive GUI demo using Streamlit (gui_demo.py)

## Installation

1. Ensure Python 3.8 or higher is installed.

2. Install required dependencies:

```bash
pip install numpy matplotlib scipy scikit-image
```

Optional dependencies for advanced features:
```bash
pip install torch streamlit pykalman
```

## Usage

### Basic Import

```python
from ultrasound_processor import Config
from ultrasound_processor.filter import fil_1
from ultrasound_processor.loc import loc_1
from ultrasound_processor.track import tracking_1
from ultrasound_processor import utils
```

### Filtering Example

```python
from ultrasound_processor.filter.fil_1 import (
    apply_butterworth_filter,
    create_butterworth_filter,
    KalmanFilter1D
)
import numpy as np

# Generate test signal
t = np.linspace(0, 1, 1000)
signal = np.sin(2 * np.pi * 50 * t) + 0.3 * np.random.randn(1000)

# Apply Butterworth low-pass filter
filtered = apply_butterworth_filter(
    signal,
    order=4,
    cutoff_frequency=100,
    sampling_rate=1000,
    btype='low'
)

# Use Kalman filter
kf = KalmanFilter1D(initial_state=0.0)
kalman_filtered = kf.filter_sequence(signal)
```

### Localization Example

```python
from ultrasound_processor.loc.loc_1 import (
    detect_bubbles_gaussian,
    detect_bubbles_adaptive,
    detect_bubbles_morphological
)
import numpy as np

# Create test image
image = np.zeros((200, 200))
image[50, 50] = 1
image[100, 100] = 1
image += 0.1 * np.random.rand(200, 200)

# Detect bubbles using different methods
positions_gaussian = detect_bubbles_gaussian(image, sigma=2.0)
positions_adaptive = detect_bubbles_adaptive(image)
positions_morph = detect_bubbles_morphological(image)
```

### Tracking Example

```python
from ultrasound_processor.track.tracking_1 import (
    track_bubbles_hungarian,
    generate_synthetic_sequence
)

# Generate synthetic sequence
frames, true_trajectories = generate_synthetic_sequence(
    num_frames=20,
    image_size=(200, 200),
    num_bubbles=10,
    seed=42
)

# Track bubbles
tracked_positions = track_bubbles_hungarian(frames)
```

### Using Utilities

```python
from ultrasound_processor import utils

# Generate test data
t, signal = utils.generate_test_signal(
    sampling_rate=1000,
    frequencies=[50, 200],
    seed=42
)

image, positions = utils.generate_test_image(
    image_size=(200, 200),
    num_bubbles=10,
    seed=42
)
```

### Running the GUI Demo

```bash
cd ultrasound_processor/track
streamlit run gui_demo.py
```

## Configuration

Default configuration parameters are available in `config.py`:

```python
from ultrasound_processor import Config

print(f"Sampling rate: {Config.DEFAULT_SAMPLING_RATE} Hz")
print(f"Image size: {Config.IMAGE_SIZE}")
print(f"Gaussian sigma: {Config.GAUSSIAN_SIGMA}")
```

## API Reference

### Filter Module (`ultrasound_processor.filter.fil_1`)

- `create_butterworth_filter(order, cutoff_frequency, sampling_rate, btype)`
- `apply_butterworth_filter(signal, order, cutoff_frequency, sampling_rate, btype)`
- `create_chebyshev_filter(order, cutoff_frequency, sampling_rate, ripple, btype)`
- `apply_chebyshev_filter(signal, order, cutoff_frequency, sampling_rate, ripple, btype)`
- `create_fir_filter(numtaps, cutoff_frequency, sampling_rate, pass_zero)`
- `apply_fir_filter(signal, numtaps, cutoff_frequency, sampling_rate, pass_zero)`
- `KalmanFilter1D` - Class for 1D Kalman filtering

### Localization Module (`ultrasound_processor.loc.loc_1`)

- `detect_bubbles_gaussian(image, sigma, min_distance, threshold_abs)`
- `detect_bubbles_maximum_filter(image, sigma, filter_size, threshold)`
- `detect_bubbles_median(image, filter_size, max_filter_size, threshold)`
- `detect_bubbles_adaptive(image, median_size, gaussian_sigma, std_multiplier, min_area, max_area)`
- `detect_bubbles_morphological(image, canny_sigma, otsu_threshold, min_size)`

### Tracking Module (`ultrasound_processor.track.tracking_1`)

- `detect_bubbles_in_frame(frame, threshold, sigma)`
- `compute_cost_matrix(tracked_positions, detected_positions)`
- `track_bubbles_hungarian(frames, initial_threshold, sigma)`
- `generate_synthetic_sequence(num_frames, image_size, num_bubbles, max_displacement, noise_level, seed)`

### Utilities (`ultrasound_processor.utils`)

- `generate_test_signal(sampling_rate, duration, frequencies, noise_level, seed)`
- `generate_test_image(image_size, num_bubbles, noise_level, seed)`
- `calculate_distance(point1, point2)`
- `create_cost_matrix(tracked_positions, detected_positions)`
- `validate_coordinates(coordinates, image_shape)`

## License

This project is licensed under the MIT License.
