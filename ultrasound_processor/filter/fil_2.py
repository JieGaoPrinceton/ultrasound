"""
Advanced filter implementations for ultrasound signal processing.

This module provides additional filtering techniques including
Chebyshev filters with spectral analysis and advanced smoothing methods.
"""

import numpy as np
from scipy.signal import cheby1, filtfilt, welch
from typing import Tuple


def apply_chebyshev_filter_with_spectral_analysis(
    signal: np.ndarray,
    sampling_rate: int,
    cutoff_frequency: float,
    order: int = 4,
    ripple: float = 1.0,
    btype: str = 'low',
    nperseg: int = 1024
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Apply Chebyshev Type I filter and compute power spectral density.
    
    Args:
        signal: Input signal array
        sampling_rate: Sampling rate in Hz
        cutoff_frequency: Cutoff frequency in Hz
        order: Filter order
        ripple: Maximum ripple allowed in passband (dB)
        btype: Filter type ('low', 'high', 'bandpass', 'bandstop')
        nperseg: Size of each segment for Welch method
        
    Returns:
        Tuple of (filtered_signal, original_psd, filtered_psd)
        where psd is (frequencies, power_spectral_density)
    """
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_frequency / nyquist
    
    # Design filter
    b, a = cheby1(order, ripple, normal_cutoff, btype=btype, analog=False)
    
    # Apply filter
    filtered_signal = filtfilt(b, a, signal)
    
    # Compute PSD using Welch method
    f_original, Pxx_original = welch(signal, sampling_rate, nperseg=nperseg)
    f_filtered, Pxx_filtered = welch(filtered_signal, sampling_rate, nperseg=nperseg)
    
    return filtered_signal, (f_original, Pxx_original), (f_filtered, Pxx_filtered)


def create_chebyshev_filter_params(
    order: int,
    cutoff_frequency: float,
    sampling_rate: int,
    ripple: float = 1.0,
    btype: str = 'low'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create Chebyshev Type I filter coefficients.
    
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