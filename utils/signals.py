# 生成时间序列信号

import matplotlib.pyplot as plt
import numpy as np


def generate_sin_signal(amplitude, frequency, phase, duration, sampling_rate, noise=False, noise_amplitude=0.1):
    """Generate a sine wave signal."""
    t = np.arange(0, duration, 1/sampling_rate)
    signal = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    if noise:
        signal += noise_amplitude * np.random.randn(len(signal))
    return t, signal

def generate_cos_signal(amplitude, frequency, phase, duration, sampling_rate, noise=False, noise_amplitude=0.1):
    """Generate a cosine wave signal."""
    t = np.arange(0, duration, 1/sampling_rate)
    signal = amplitude * np.cos(2*np.pi*frequency*t + phase)
    if noise:
        signal += noise_amplitude * np.random.randn(len(signal))
    return t, signal

def generate_pulse_signal(amplitude, duration, sampling_rate, pulse_width, noise=False, noise_amplitude=0.1):
    """Generate a pulse wave signal."""
    t = np.arange(0, duration, 1/sampling_rate)
    signal = np.zeros(len(t))
    signal[0:pulse_width] = amplitude
    if noise:
        signal += noise_amplitude * np.random.randn(len(signal))
    return t, signal


if __name__ == '__main__':

    amplitude = 1
    frequency = 1
    phase = 0
    duration = 1
    sampling_rate = 1000

    noise = False
    noise_amplitude = 0.1

    pulse_width = 100

    # Generate a sine wave signal with noise
    t, signal = generate_sin_signal(amplitude, frequency, phase, duration, sampling_rate, noise=noise, noise_amplitude=noise_amplitude)
    plt.plot(t, signal)
    plt.title("Sine wave signal with noise")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()

    # Generate a cosine wave signal without noise
    t, signal = generate_cos_signal(amplitude, frequency, phase, duration, sampling_rate, noise=noise, noise_amplitude=noise_amplitude)
    plt.plot(t, signal)
    plt.title("Cosine wave signal without noise")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()

    # Generate a pulse wave signal with noise
    t, signal = generate_pulse_signal(amplitude, frequency, sampling_rate, pulse_width, noise=noise, noise_amplitude=noise_amplitude)
    plt.plot(t, signal)
    plt.title("Pulse wave signal without noise")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()