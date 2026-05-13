"""
Focus widget alternative: computes focus state from EEG band powers.
Provides calibration, smoothing, and hysteresis thresholding.
"""
import time
import numpy as np
from brainflow.board_shim import BoardShim

def compute_bandpowers(channel_data, fs, band_defs):
    """
    Compute power spectral density in frequency bands.
    
    Args:
        channel_data: 1D numpy array of EEG samples
        fs: sampling rate (Hz)
        band_defs: dict of band_name -> (fmin, fmax)
    
    Returns:
        dict of band_name -> power
    """
    n = len(channel_data)
    if n < 2:
        return {b: 0.0 for b in band_defs}
    
    x = channel_data - np.mean(channel_data)
    win = np.hanning(n)
    xw = x * win
    fft = np.fft.rfft(xw)
    psd = (np.abs(fft) ** 2) / (np.sum(win ** 2) * fs)
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    
    band_powers = {}
    for name, (fmin, fmax) in band_defs.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        if np.any(mask):
            band_powers[name] = float(np.trapz(psd[mask], freqs[mask]))
        else:
            band_powers[name] = 0.0
    return band_powers

# this is the focus widget  
def compute_focus_metric_from_bands(bp, eps=1e-12):
    """
    Compute focus metric from band powers.
    
    Args:
        bp: dict with keys 'theta', 'alpha', 'beta'
        eps: small value to avoid division by zero
    
    Returns:
        float: focus metric (beta / (alpha + theta))
    """
    alpha = bp.get('alpha', 0.0)
    theta = bp.get('theta', 0.0)
    beta = bp.get('beta', 0.0)
    return beta / (alpha + theta + eps)


# this is the relaxation widget 
def compute_relaxation_metric_from_bands(bp, eps=1e-12):
    """
    Relaxation metric: (alpha + theta) / (beta + eps).
    Higher => more relaxed. Use this if you want volume to increase with relaxation
    without using invert=True in mapping.
    """
    alpha = bp.get('alpha', 0.0)
    theta = bp.get('theta', 0.0)
    beta = bp.get('beta', 0.0)
    return (alpha + theta) / (beta + eps)


def calibrate_baseline(board, eeg_channels, sampling_rate, window_seconds, duration_seconds, band_defs):
    """
    Collect baseline metrics while user is relaxed.
    
    Args:
        board: BrainFlow BoardShim instance
        eeg_channels: list of EEG channel indices
        sampling_rate: sampling rate (Hz)
        window_seconds: window length for PSD computation
        duration_seconds: how long to calibrate (seconds)
        band_defs: frequency band definitions
    
    Returns:
        tuple: (baseline_mean, baseline_std)
    """
    n_samples = max(4, int(round(window_seconds * sampling_rate)))
    metrics = []
    t_end = time.time() + duration_seconds
    
    while time.time() < t_end:
        raw = board.get_current_board_data(n_samples)
        if raw is None or raw.size == 0:
            time.sleep(0.05)
            continue
        
        vals = []
        for ch_idx in eeg_channels:
            if ch_idx >= raw.shape[0]:
                continue
            ch_data = raw[ch_idx, -n_samples:]
            bp = compute_bandpowers(ch_data, sampling_rate, band_defs)
            vals.append(compute_focus_metric_from_bands(bp))
        
        if vals:
            metrics.append(np.mean(vals))
        time.sleep(0.05)
    
    arr = np.array(metrics)
    mean = float(np.mean(arr)) if arr.size else 0.0
    std = float(np.std(arr)) if arr.size else 1e-6
    if std == 0.0:
        std = 1e-6
    
    return mean, std


class FocusState:
    """
    Tracks focus state with hysteresis and hold time.
    
    Uses z-score normalization, exponential smoothing (EMA),
    and hysteresis thresholds to determine relaxed vs focused state.
    """
    
    def __init__(self, smooth_alpha=0.3, high_z=1.0, low_z=0.5, hold_time=0.5):
        """
        Initialize focus state tracker.
        
        Args:
            smooth_alpha: EMA smoothing factor (0..1), higher = more responsive
            high_z: z-score threshold to enter focused state
            low_z: z-score threshold to exit focused state
            hold_time: seconds metric must exceed high_z before entering focused
        """
        self.smooth_alpha = float(smooth_alpha)
        self.high_z = float(high_z)
        self.low_z = float(low_z)
        self.hold_time = float(hold_time)
        self.smoothed = None
        self.state = False
        self._enter_timer = None
    
    def update(self, metric, baseline_mean, baseline_std, ts=None):
        """
        Update focus state based on new metric.
        
        Args:
            metric: raw focus metric value
            baseline_mean: baseline mean from calibration
            baseline_std: baseline std from calibration
            ts: timestamp (optional, uses current time if not provided)
        
        Returns:
            tuple: (is_focused, smoothed_z_score, raw_z_score)
        """
        # Compute z-score
        z = (metric - baseline_mean) / (baseline_std if baseline_std != 0 else 1e-6)
        
        # Exponential moving average smoothing
        if self.smoothed is None:
            self.smoothed = z
        else:
            self.smoothed = self.smooth_alpha * z + (1 - self.smooth_alpha) * self.smoothed
        
        now = ts if ts is not None else time.time()
        
        # Hysteresis with hold time
        if not self.state:
            # Try to enter focused state
            if self.smoothed >= self.high_z:
                if self._enter_timer is None:
                    self._enter_timer = now
                elif (now - self._enter_timer) >= self.hold_time:
                    self.state = True
                    self._enter_timer = None
            else:
                self._enter_timer = None
        else:
            # Try to exit focused state
            if self.smoothed < self.low_z:
                self.state = False
                self._enter_timer = None
        
        return self.state, self.smoothed, z