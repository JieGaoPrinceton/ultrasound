import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, filtfilt, welch
from pykalman import KalmanFilter

# 模拟超声信号
np.random.seed(0)
sampling_rate = 1000  # 采样率，单位为 Hz
t = np.linspace(0, 1.0, sampling_rate)  # 1 秒的时间序列
freq1, freq2 = 50, 200  # 两个不同频率的信号
ultrasound_signal = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t) + 0.3 * np.random.randn(len(t))

# 使用基于 FIR 滤波器的滤波算法
numtaps = 101  # 滤波器的阶数
cutoff_frequency = 100  # 截止频率，单位为 Hz
nyquist = 0.5 * sampling_rate
normal_cutoff = cutoff_frequency / nyquist
fir_coeff = firwin(numtaps, normal_cutoff, pass_zero='lowpass')

# 对超声信号进行 FIR 滤波
filtered_signal_fir = filtfilt(fir_coeff, [1.0], ultrasound_signal)

# 使用卡尔曼滤波对超声信号进行平滑处理
kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
kalman_filtered_signal, _ = kf.filter(ultrasound_signal)

# 绘制原始信号和 FIR 滤波后的信号
plt.figure(figsize=(12, 6))
plt.plot(t, ultrasound_signal, label='Original Signal', color='blue', alpha=0.5)
plt.plot(t, filtered_signal_fir, label='Filtered Signal (FIR Filter)', color='red')
plt.title('Ultrasound Signal Filtering - FIR Filter')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 绘制卡尔曼滤波后的信号
plt.figure(figsize=(12, 6))
plt.plot(t, ultrasound_signal, label='Original Signal', color='blue', alpha=0.5)
plt.plot(t, kalman_filtered_signal, label='Filtered Signal (Kalman Filter)', color='green')
plt.title('Ultrasound Signal Filtering - Kalman Filter')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 使用 Welch 方法计算滤波前后的功率谱密度
f_original, Pxx_original = welch(ultrasound_signal, sampling_rate, nperseg=1024)
f_fir, Pxx_fir = welch(filtered_signal_fir, sampling_rate, nperseg=1024)
f_kalman, Pxx_kalman = welch(kalman_filtered_signal.flatten(), sampling_rate, nperseg=1024)

# 绘制功率谱密度
plt.figure(figsize=(12, 6))
plt.semilogy(f_original, Pxx_original, label='Original Signal PSD', color='blue', alpha=0.5)
plt.semilogy(f_fir, Pxx_fir, label='Filtered Signal PSD (FIR Filter)', color='red')
plt.semilogy(f_kalman, Pxx_kalman, label='Filtered Signal PSD (Kalman Filter)', color='green')
plt.title('Power Spectral Density - Welch Method')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power Spectral Density [V**2/Hz]')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()