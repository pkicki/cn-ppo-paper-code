import numpy as np
import torch as th
import matplotlib.pyplot as plt
from scipy import signal

from cnppo.lp import LowPassFilteredNoiseDist

def test_filtering():
    # --- Configuration ---
    SAMPLING_RATE = 100.0  # Hz (e.g., dt=0.01)
    CUTOFF_FREQ = 5.0      # Hz
    DURATION = 20.0        # Seconds
    N_ENVS = 1
    ACTION_DIM = 1
    SEQ_LEN = 200          # Internal buffer length (smaller than total duration to test refills)
    
    # Calculate steps
    n_steps = int(DURATION * SAMPLING_RATE)
    time_axis = np.linspace(0, DURATION, n_steps)

    # --- Setup Distribution ---
    print(f"Initializing Distribution: Fs={SAMPLING_RATE}Hz, Cutoff={CUTOFF_FREQ}Hz")
    dist = LowPassFilteredNoiseDist(
        cutoff_freq=CUTOFF_FREQ,
        sampling_rate=SAMPLING_RATE,
        seq_len=SEQ_LEN,
        action_dim=ACTION_DIM,
        filter_order=4 # Sharper cutoff for better visualization
    )

    # Mock SB3 behavior: Create internal Normal distribution
    # We set mean=0, log_std=0 (std=1)
    mean = th.zeros(N_ENVS, ACTION_DIM)
    log_std = th.zeros(N_ENVS, ACTION_DIM)
    dist.proba_distribution(mean, log_std)

    # --- Sampling Loop ---
    samples = []
    print(f"Sampling {n_steps} steps...")
    
    for _ in range(n_steps):
        # sample() returns a tensor of shape (n_envs, action_dim)
        action = dist.sample()
        samples.append(action.item())

    samples = np.array(samples)

    # --- Visualization ---
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), constrained_layout=True)

    # 1. Time Domain Plot
    axs[0].plot(time_axis, samples, color='royalblue', lw=1.5, label='Sampled Noise')
    axs[0].set_title(f"Time Domain Signal (Butterworth Low-Pass @ {CUTOFF_FREQ}Hz)")
    axs[0].set_ylabel("Amplitude")
    axs[0].set_xlabel("Time (s)")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    # 2. Spectrogram
    f, t, Sxx = signal.spectrogram(samples, fs=SAMPLING_RATE, nperseg=128, noverlap=64)
    pcm = axs[1].pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='inferno')
    axs[1].set_title("Spectrogram")
    axs[1].set_ylabel("Frequency (Hz)")
    axs[1].set_xlabel("Time (s)")
    axs[1].axhline(y=CUTOFF_FREQ, color='white', linestyle='--', label=f'Cutoff {CUTOFF_FREQ}Hz')
    axs[1].legend(loc='upper right')
    fig.colorbar(pcm, ax=axs[1], label='Power (dB)')

    # 3. Power Spectral Density (Welch's Method)
    freqs, psd = signal.welch(samples, fs=SAMPLING_RATE, nperseg=512)
    axs[2].semilogy(freqs, psd, color='darkorange', lw=2)
    axs[2].set_title("Power Spectral Density (PSD)")
    axs[2].set_ylabel("Power/Frequency (dB/Hz)")
    axs[2].set_xlabel("Frequency (Hz)")
    axs[2].axvline(x=CUTOFF_FREQ, color='red', linestyle='--', label=f'Cutoff {CUTOFF_FREQ}Hz')
    axs[2].grid(True, which='both', alpha=0.3)
    axs[2].legend()

    print("Plot generated. Please check the output window.")
    plt.show()

if __name__ == "__main__":
    test_filtering()