import numpy as np
import torch as th
from scipy import signal
from stable_baselines3.common.distributions import DiagGaussianDistribution


class ButterworthNoiseProcess:
    """
    A process that generates low-pass filtered Gaussian noise (temporally correlated).
    Mimics the API of the pink.cnrl.ColoredNoiseProcess for compatibility.
    """
    def __init__(self, size, cutoff, fs, order=2, rng=None):
        """
        :param size: tuple (n_envs, action_dim, seq_len)
        :param cutoff: Cutoff frequency in Hz
        :param fs: Sampling frequency in Hz
        :param order: Order of the Butterworth filter
        :param rng: Numpy random generator
        """
        self.size = size  # (n_envs, action_dim, seq_len)
        self.n_envs, self.action_dim, self.seq_len = size
        self.cutoff = cutoff
        self.fs = fs
        self.order = order
        self.rng = rng if rng is not None else np.random.default_rng()

        # Design filter
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        self.b, self.a = signal.butter(order, normal_cutoff, btype='low', analog=False)

        # Calculate normalization factor to ensure unit variance output
        # We simulate a dummy signal to find the attenuation factor of the filter
        w, h = signal.freqz(self.b, self.a)
        # Estimate RMS gain via Parseval's theorem or simple simulation
        dummy_noise = self.rng.standard_normal(10000)
        filtered_dummy = signal.lfilter(self.b, self.a, dummy_noise)
        self.scale_factor = 1.0 / np.std(filtered_dummy)

        # Internal state
        self._buffer = None
        self._idx = 0
        # Filter state (zi) must be maintained across buffer refills to prevent discontinuities
        # Shape of zi for lfilter is (max(len(a), len(b)) - 1, n_envs, action_dim)
        self._zi = signal.lfilter_zi(self.b, self.a)
        self._zi = np.tile(self._zi[None, None, :], (self.n_envs, self.action_dim, 1))

    def sample(self):
        """Returns a single step of noise for all envs."""
        if self._buffer is None or self._idx >= self.seq_len:
            self._refill_buffer()
        
        sample = self._buffer[:, :, self._idx]
        self._idx += 1
        return sample

    def _refill_buffer(self):
        """Generates a new sequence of noise and filters it."""
        # 1. Generate White Noise
        white_noise = self.rng.standard_normal(self.size)
        
        # 2. Apply Low Pass Filter
        # We apply along the last axis (time/seq_len)
        # We pass self._zi to maintain correlation with the end of the previous buffer
        filtered_noise, self._zi = signal.lfilter(
            self.b, self.a, white_noise, axis=-1, zi=self._zi
        )
        
        # 3. Normalize and Store
        self._buffer = filtered_noise * self.scale_factor
        self._idx = 0


class LowPassFilteredNoiseDist(DiagGaussianDistribution):
    def __init__(self, cutoff_freq, sampling_rate, seq_len, action_dim=None, rng=None, 
                 action_low=None, action_high=None, filter_order=2):
        """
        Gaussian colored noise distribution using a Butterworth low-pass filter.

        This allows for smooth exploration by correlating actions in time. 
        Similar structure to ColoredNoiseDist but uses Scipy signal processing.

        Parameters
        ----------
        cutoff_freq : float
            Cutoff frequency of the low-pass filter in Hz. Lower values = smoother, 
            more correlated noise.
        sampling_rate : float
            The sampling rate of the environment (e.g., 1 / dt).
        seq_len : int
            Length of internal noise buffer. Should roughly match episode horizon.
        action_dim : int
            Dimensionality of the action space.
        rng : np.random.Generator, optional
            Random number generator.
        action_low : float or array, optional
            Min action bound.
        action_high : float or array, optional
            Max action bound.
        filter_order : int, optional
            Order of the Butterworth filter (default: 2).
        """
        assert action_dim is not None, "`action_dim` must be specified."

        assert (action_low is None and action_high is None) or (
            action_low is not None and action_high is not None
        ), "Set either none of action_low and action_high, or both."
        
        self.action_low = -1.0 if action_low is None else action_low
        self.action_high = 1.0 if action_high is None else action_high
        self.action_halfspan = (self.action_high - self.action_low) / 2.0
        self.action_middle = (self.action_high + self.action_low) / 2.0
        
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.cutoff = cutoff_freq
        self.fs = sampling_rate
        self.order = filter_order
        self.rng = rng
        
        self._gens = {}
        
        # Initialize parent DiagGaussian
        super().__init__(self.action_dim)

    def get_gen(self, n_envs, device="cpu"):
        if n_envs not in self._gens:
            # Create a process generator for this batch size
            gen = ButterworthNoiseProcess(
                size=(n_envs, self.action_dim, self.seq_len), 
                cutoff=self.cutoff, 
                fs=self.fs, 
                order=self.order,
                rng=self.rng
            )
            self._gens[n_envs] = gen
        return self._gens[n_envs]

    def sample(self) -> th.Tensor:
        device = self.distribution.mean.device
        n_envs, action_dim = self.distribution.batch_shape
        
        # Get generator for this batch size
        gen = self.get_gen(n_envs, device=device)
        
        # Sample from the filtered process
        lp_sample = th.tensor(gen.sample(), dtype=th.float32, device=device)
        
        # Apply the distribution parameters (Mean + StdDev * Noise)
        self.gaussian_actions = self.distribution.mean + self.distribution.stddev * lp_sample
        
        # Scale to action space
        return self.gaussian_actions * self.action_halfspan + self.action_middle

    def __repr__(self) -> str:
        return f"LowPassFilteredNoiseDist(cutoff={self.cutoff}, fs={self.fs})"