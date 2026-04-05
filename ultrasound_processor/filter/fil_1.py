"""
Filter module for ultrasound signal processing.

This module provides various filtering techniques for processing
ultrasound microbubble signals, including Butterworth, Chebyshev,
FIR, and Kalman filters.
"""

import numpy as np
from scipy.signal import butter, filtfilt, cheby1, firwin
from typing import Tuple, Optional


def create_butterworth_filter(
    order: int,
    cutoff_frequency: float,
    sampling_rate: int,
    btype: str = 'low'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a Butterworth filter with specified parameters.
    
    Args:
        order: Filter order
        cutoff_frequency: Cutoff frequency in Hz
        sampling_rate: Sampling rate in Hz
        btype: Filter type ('low', 'high', 'bandpass', 'bandstop')
        
    Returns:
        Tuple of (numerator, denominator) coefficients
    """
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    return b, a


def apply_butterworth_filter(
    signal: np.ndarray,
    order: int,
    cutoff_frequency: float,
    sampling_rate: int,
    btype: str = 'low'
) -> np.ndarray:
    """
    Apply a Butterworth filter to a signal.
    
    Args:
        signal: Input signal array
        order: Filter order
        cutoff_frequency: Cutoff frequency in Hz
        sampling_rate: Sampling rate in Hz
        btype: Filter type ('low', 'high', 'bandpass', 'bandstop')
        
    Returns:
        Filtered signal array
    """
    b, a = create_butterworth_filter(order, cutoff_frequency, sampling_rate, btype)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


def create_chebyshev_filter(
    order: int,
    cutoff_frequency: float,
    sampling_rate: int,
    ripple: float = 1.0,
    btype: str = 'low'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a Chebyshev Type I filter with specified parameters.
    
    Args:
        order: Filter order
        cutoff_frequency: Cutoff frequency in Hz
        sampling_rate: Sampling rate in Hz
        ripple: Maximum ripple allowed in passband (dB)
        btype: Filter type ('low', 'high', 'bandpass', 'bandstop')
        
    Returns:
        Tuple of (numerator, denominator) coefficients
    """
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_frequency / nyquist
    b, a = cheby1(order, ripple, normal_cutoff, btype=btype, analog=False)
    return b, a


def apply_chebyshev_filter(
    signal: np.ndarray,
    order: int,
    cutoff_frequency: float,
    sampling_rate: int,
    ripple: float = 1.0,
    btype: str = 'low'
) -> np.ndarray:
    """
    Apply a Chebyshev Type I filter to a signal.
    
    Args:
        signal: Input signal array
        order: Filter order
        cutoff_frequency: Cutoff frequency in Hz
        sampling_rate: Sampling rate in Hz
        ripple: Maximum ripple allowed in passband (dB)
        btype: Filter type ('low', 'high', 'bandpass', 'bandstop')
        
    Returns:
        Filtered signal array
    """
    b, a = create_chebyshev_filter(order, cutoff_frequency, sampling_rate, ripple, btype)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


def create_fir_filter(
    numtaps: int,
    cutoff_frequency: float,
    sampling_rate: int,
    pass_zero: str = 'lowpass'
) -> np.ndarray:
    """
    Create an FIR filter using the window method.
    
    Args:
        numtaps: Number of taps (filter order + 1)
        cutoff_frequency: Cutoff frequency in Hz
        sampling_rate: Sampling rate in Hz
        pass_zero: 'lowpass' or 'highpass'
        
    Returns:
        FIR filter coefficients
    """
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_frequency / nyquist
    coefficients = firwin(numtaps, normal_cutoff, pass_zero=pass_zero)
    return coefficients


def apply_fir_filter(
    signal: np.ndarray,
    numtaps: int,
    cutoff_frequency: float,
    sampling_rate: int,
    pass_zero: str = 'lowpass'
) -> np.ndarray:
    """
    Apply an FIR filter to a signal.
    
    Args:
        signal: Input signal array
        numtaps: Number of taps (filter order + 1)
        cutoff_frequency: Cutoff frequency in Hz
        sampling_rate: Sampling rate in Hz
        pass_zero: 'lowpass' or 'highpass'
        
    Returns:
        Filtered signal array
    """
    coefficients = create_fir_filter(numtaps, cutoff_frequency, sampling_rate, pass_zero)
    filtered_signal = filtfilt(coefficients, [1.0], signal)
    return filtered_signal


class KalmanFilter1D:
    """
    Simple 1D Kalman filter for signal smoothing.
    
    This is a basic implementation for educational purposes.
    For production use, consider using pykalman or filterpy libraries.
    """
    
    def __init__(
        self,
        initial_state: float = 0.0,
        process_variance: float = 1e-5,
        measurement_variance: float = 0.1
    ):
        """
        Initialize the Kalman filter.
        
        Args:
            initial_state: Initial state estimate
            process_variance: Process noise variance (Q)
            measurement_variance: Measurement noise variance (R)
        """
        self.x = initial_state  # State estimate
        self.P = 1.0  # Error covariance
        self.Q = process_variance  # Process noise variance
        self.R = measurement_variance  # Measurement noise variance
    
    def update(self, measurement: float) -> float:
        """
        Update the filter with a new measurement.
        
        Args:
            measurement: New measurement value
            
        Returns:
            Updated state estimate
        """
        # Prediction step
        x_pred = self.x
        P_pred = self.P + self.Q
        
        # Update step
        K = P_pred / (P_pred + self.R)  # Kalman gain
        self.x = x_pred + K * (measurement - x_pred)
        self.P = (1 - K) * P_pred
        
        return self.x
    
    def filter_sequence(self, measurements: np.ndarray) -> np.ndarray:
        """
        Filter a sequence of measurements.
        
        Args:
            measurements: Array of measurements
            
        Returns:
            Array of filtered state estimates
        """
        filtered = np.zeros_like(measurements)
        for i, measurement in enumerate(measurements):
            filtered[i] = self.update(measurement)
        return filtered
