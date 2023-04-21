import matplotlib.pyplot as plt
import numpy as np

from utils.signals import generate_cos_signal, generate_sin_signal


def compare_x(x: np.ndarray, recon_x: np.ndarray, epoch: int) -> None:
    """
    Plot original x and reconstructed x.

    Parameters
    ----------
    x : np.ndarray
        Original x.
    recon_x : np.ndarray
        Reconstructed x.
    epoch : int
        Epoch number.
    """
    
    # Create a figure with a size of 10 by 10 inches
    plt.figure(figsize=(10, 10))
    
    # Create a time array
    t = np.arange(0, 5, 1 / (178 / 5))
    
    # Plot the original x and the reconstructed x
    plt.plot(t, x, 'g', label='original x')
    plt.plot(t, recon_x, 'r', label='reconstructed x')
    plt.title(f'epoch: {epoch}')
    plt.legend(loc=1)
    
    # Save the plot
    plt.savefig(f'recon/{epoch}.png')
    plt.close()


if __name__ == "__main__":
    
    amplitude: float = 1
    frequency: float = 1
    phase: float = 0
    duration: float = 5
    sampling_rate: float = 178 / duration
    
    _, x = generate_sin_signal(amplitude, frequency, phase, duration, sampling_rate)
    
    _, recon_x = generate_sin_signal(amplitude, frequency, phase, duration, sampling_rate)
    
    compare_x(x, recon_x, 1)