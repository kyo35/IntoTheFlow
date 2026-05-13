"""
Helpers to map focus/relaxation metric to a 0..1 volume and apply it.

Usage:
  from volume_control import map_metric_to_volume, set_system_volume_percent, TonePlayer
"""
import subprocess
import threading
import math
import numpy as np

def map_metric_to_volume(metric, baseline_mean=0.0, baseline_std=1.0,
                         invert=False, z_min=-2.0, z_max=2.0,
                         min_v=0.05, max_v=1.0, curve='linear'):
    """
    Map a raw metric (or z-score) -> volume in [min_v, max_v].
    - If baseline_mean/std provided, metric is treated as raw and converted to z.
    - invert=True makes lower metric -> higher volume (for relaxation).
    - curve: 'linear' or 'exp' (exponential curve for more sensitivity).
    """
    # convert to z if baseline provided
    if baseline_std is None or baseline_std == 0:
        z = metric - baseline_mean
    else:
        z = (metric - baseline_mean) / baseline_std

    if invert:
        z = -z

    # clamp and normalize to 0..1
    z_clamped = max(min(z, z_max), z_min)
    norm = (z_clamped - z_min) / (z_max - z_min)

    if curve == 'exp':
        # emphasize extremes (0..1)
        norm = math.pow(norm, 0.5)  # tweak exponent as desired

    vol = min_v + (max_v - min_v) * norm
    return max(0.0, min(1.0, vol))


def set_system_volume_percent(percent):
    """
    Set system (ALSA) Master volume to percent (0..100). Uses amixer (common on Linux).
    If you're using PulseAudio you can use 'pactl' instead.
    """
    try:
        pct = int(max(0, min(100, percent)))
        subprocess.run(['amixer', 'sset', 'Master', f'{pct}%'], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


class TonePlayer(threading.Thread):
    """
    Background sine tone player using sounddevice (optional dependency).
    Call set_amplitude(0..1) to change loudness in real-time.
    """
    def __init__(self, frequency=440.0, samplerate=44100, blocksize=1024):
        super().__init__(daemon=True)
        self.frequency = frequency
        self.samplerate = samplerate
        self.blocksize = blocksize
        self._amp = 0.0
        self._keep_running = True
        self._phase = 0.0
        self._lock = threading.Lock()
        self._has_sounddevice = False
        try:
            import sounddevice as sd
            self._sd = sd
            self._has_sounddevice = True
        except Exception:
            self._has_sounddevice = False

    def set_amplitude(self, amp):
        with self._lock:
            self._amp = max(0.0, min(1.0, float(amp)))

    def stop(self):
        self._keep_running = False

    def run(self):
        if not self._has_sounddevice:
            return
        sd = self._sd
        two_pi_f = 2.0 * math.pi * self.frequency
        def callback(outdata, frames, time_info, status):
            t = (np.arange(frames) + 0) / self.samplerate
            with self._lock:
                amp = self._amp
            # keep phase between 0..2pi
            phase = self._phase
            samples = amp * np.sin(two_pi_f * t + phase)
            self._phase = (phase + two_pi_f * frames / self.samplerate) % (2.0 * math.pi)
            outdata[:,0] = samples
        try:
            import numpy as np
            with sd.OutputStream(channels=1, callback=callback, samplerate=self.samplerate, blocksize=self.blocksize):
                while self._keep_running:
                    sd.sleep(100)
        except Exception as e:
            print(f"TonePlayer error: {e}")
        finally:
            print("TonePlayer stopped")