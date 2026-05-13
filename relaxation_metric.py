"""
relaxation_metric.py

Drop-in module for computing a smoothed relaxation metric from EEG band powers
using BrainFlow. Designed for OpenBCI Cyton / Cyton+Daisy on a Raspberry Pi.

The main() example at the bottom is set up for CYTON_DAISY_BOARD.

This is the main file for this project. 
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import math
 
import numpy as np
import threading
import subprocess
import os
import PyQt5
import re
from SessionStats import Ui_MainWindow as StatsUI


from brainflow.data_filter import DataFilter
# Only needed for the example usage:
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# for the GUI 
from ActualUI import Ui_MainWindow
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QScroller
from PyQt5.QtCore import Qt 


import sys

def get_screen_geometry(app, preferred_index=0):
    """
    Return the geometry for a preferred Qt screen index if available.
    Falls back to the primary screen.
    """
    screens = app.screens()
    if screens and 0 <= preferred_index < len(screens):
        return screens[preferred_index].availableGeometry()
    return app.primaryScreen().availableGeometry()

# global variable to hold the current relaxation metric (for demonstration/logging purposes)
current_metric = 0.0
vol_sys = 0
pos = 0
videos = []
PLAYBACK_MODE = False
playback_index = 0
last_relaxed_state = False
video = None
# for active channels
channel_model = None

shutdown_requested = False # to allow the user or operator to exit session / stop streaming

# globals to store session relaxation data in the GUI
session_time_data = []
session_metric_data = []
session_media_data = []
session_band_data = []
session_volume_data = []

current_bands = [0, 0, 0, 0, 0]   # delta theta alpha beta gamma stored globally for use in gui
last_volume_percent = None
last_active_channels = []
last_debug_print_time = 0.0

@dataclass
class RelaxationConfig:
    sampling_rate: int
    eeg_channels: List[int]

    # Which EEG channels to compute the metric on (defaults to all EEG channels)
    metric_channels: Optional[List[int]] = None

    window_sec: float = 5.0        # window length for band powers
    smoothing_alpha: float = 0.2    # EMA smoothing, 0–1 (higher = less smoothing)
    sigmoid_gain: float = 10.0      # gain for logistic scaling
    sigmoid_center: float = 0.5     # center point for logistic scaling

    # Metric selection:
    #  - "ratio": weighted ratio of "relax" bands vs total
    #  - "log_ab": log(alpha) - log(beta) (more stable than a raw ratio)
    #  - "openbci_focus": OpenBCI-style alpha-high AND beta-low gating (produces 0..1 score)
    metric_mode: str = "ratio"

    # Weights for ratio mode (helps reduce "sleepy theta" blow-ups)
    theta_weight: float = 1.0
    delta_weight: float = 0.25

    # Numerical safety
    eps: float = 1e-9
    denom_floor: float = 1e-6
    band_floor: float = 1e-6
    raw_clip_min: float = -2.5
    raw_clip_max: float = 2.5
    max_raw_step: float = 0.75

    # OpenBCI focus-style thresholds (original article thresholds are in uV amplitude;
    # here we approximate amplitude via sqrt(power))
    alpha_threshold_uV: float = 1.0
    beta_threshold_uV: float = 1.0
    alpha_noise_cap_uV: float = 4.0

    # How soft the OpenBCI gating is (higher = closer to hard thresholding)
    gate_gain: float = 8.0


class RelaxationMetric:
    """
    Stateful relaxation metric calculator.

    Call `update(eeg_data_window)` repeatedly with a 2D numpy array:
        shape = (num_channels, num_samples)
    containing the most recent window of EEG data.

    Returns a float in [0, 1] each time.
    """

    def __init__(self, config: RelaxationConfig):
        self.cfg = config
        self._ema_value: Optional[float] = None  # exponential moving average
        self._latest_bands: Optional[Tuple[float, float, float, float, float]] = None

    # ---------- internal helpers ----------

    # Note: BrainFlow's get_avg_band_powers expects data in shape (num_channels, num_samples)
    # returns avg_bands (delta, theta, alpha, beta, gamma) and band_freqs
    def _compute_band_powers(
        self, data: "np.ndarray"
    ) -> Tuple[float, float, float, float, float]:
        """
        Uses BrainFlow's get_avg_band_powers to get average band powers.
        Returns: (delta, theta, alpha, beta, gamma)
        """
        ch = self.cfg.metric_channels if self.cfg.metric_channels is not None else self.cfg.eeg_channels
        avg_bands, _ = DataFilter.get_avg_band_powers(
            data,
            ch,
            self.cfg.sampling_rate,
            True  # apply_filter
        )
        # avg_bands = [delta, theta, alpha, beta, gamma]
        bands = tuple(avg_bands)  # type: ignore
        self._latest_bands = bands
        return bands

    def _sanitize_bands(
        self, delta: float, theta: float, alpha: float, beta: float, gamma: float
    ) -> Tuple[float, float, float, float, float]:
        """
        Keep band powers finite and above a small floor so one bad window does not
        explode the log-ratio calculation.
        """
        floor = self.cfg.band_floor
        vals = [delta, theta, alpha, beta, gamma]
        cleaned = []
        for v in vals:
            if not math.isfinite(v):
                cleaned.append(floor)
            else:
                cleaned.append(max(float(v), floor))
        return tuple(cleaned)  # type: ignore

    # raw relaxation ratio: (alpha + theta) / (alpha + theta + beta + gamma)
    # determine how much the "relaxation" bands (alpha, theta) dominate over the "arousal" bands (beta, gamma)

    def _compute_raw_relaxation(
        self, delta: float, theta: float, alpha: float, beta: float, gamma: float
    ) -> float:
        """Compute the raw (pre-smoothing) value according to cfg.metric_mode."""

        eps = self.cfg.eps
        delta, theta, alpha, beta, gamma = self._sanitize_bands(delta, theta, alpha, beta, gamma)

        if self.cfg.metric_mode == "log_ab":
            # More stable than a ratio; reduces blow-ups when beta gets very small.
            return math.log(alpha + 0.15*theta + eps) - math.log(beta + eps)

        # Default: weighted ratio.
        # Downweight delta (and optionally theta) to avoid "tired" inflation.
        relax = alpha + self.cfg.theta_weight * theta + self.cfg.delta_weight * delta
        arousal = beta + gamma
        denom = relax + arousal
        denom = max(denom, self.cfg.denom_floor)
        return relax / denom
    def _stabilize_raw(self, raw: float) -> float:
        """
        Clip extreme raw values and limit one-window jumps so brief artifacts do not
        slam the metric to 0 or 1.
        """
        raw = max(self.cfg.raw_clip_min, min(self.cfg.raw_clip_max, raw))
        if self._ema_value is None:
            return raw

        delta = raw - self._ema_value
        max_step = self.cfg.max_raw_step
        if delta > max_step:
            raw = self._ema_value + max_step
        elif delta < -max_step:
            raw = self._ema_value - max_step
        return raw



    # EMA smoothing to reduce noise and make the metric more stable over time.
    # Moving average to reduce noise 
    def _update_ema(self, value: float) -> float:
        """
        Exponential moving average to smooth the metric over time.
        """
        a = self.cfg.smoothing_alpha
        if self._ema_value is None:
            self._ema_value = value
        else:
            self._ema_value = a * value + (1.0 - a) * self._ema_value
        return self._ema_value
    
    
    # Fixes input into a range of [0,1]

    def _sigmoid_scale(self, x: float) -> float:
        """
        Map raw ratio into [0, 1] via a logistic function.
        """
        k = self.cfg.sigmoid_gain
        x0 = self.cfg.sigmoid_center
        z = -k * (x - x0)
        return 1.0 / (1.0 + math.exp(z))

    def _openbci_focus_score(self, alpha: float, beta: float) -> float:
        """OpenBCI-inspired focus score.

        The original OpenBCI article uses averaged FFT amplitudes in uV with thresholds like:
          alpha_avg > 1.0uV, beta_avg < 1.0uV, alpha_avg < 4.0uV

        Here we approximate an amplitude-like value as sqrt(power).
        Returns a smooth score in [0,1] (not a hard boolean).
        """
        eps = self.cfg.eps
        alpha_uV = math.sqrt(max(alpha, 0.0) + eps)
        beta_uV = math.sqrt(max(beta, 0.0) + eps)

        # Soft gates: alpha above threshold, beta below threshold, alpha below noise cap.
        k = self.cfg.gate_gain
        a_hi = 1.0 / (1.0 + math.exp(-k * (alpha_uV - self.cfg.alpha_threshold_uV)))
        b_lo = 1.0 / (1.0 + math.exp(-k * (self.cfg.beta_threshold_uV - beta_uV)))
        a_cap = 1.0 / (1.0 + math.exp(-k * (self.cfg.alpha_noise_cap_uV - alpha_uV)))

        # a_cap is ~1 when alpha is BELOW cap; so use it directly.
        score = a_hi * b_lo * a_cap
        return max(0.0, min(1.0, score))

    # ---------- public API ----------


    # event loop calls this repeatedly with the latest EEG window to get the current relaxation metric

    # this is more of a logger for debugging 
    def update(self, data_window: "np.ndarray") -> float:
        """
        Compute the relaxation metric for the latest EEG window.

        :param data_window: numpy array (num_channels x num_samples)
                            containing the last `window_sec` of EEG data.
        :return: relaxation metric in [0, 1]
        """
        if data_window is None:
            raise ValueError("data_window cannot be None")

        if self._latest_bands is None:
            delta, theta, alpha, beta, gamma = self._compute_band_powers(data_window)
        else:
            delta, theta, alpha, beta, gamma = self._latest_bands

        delta, theta, alpha, beta, gamma = self._sanitize_bands(delta, theta, alpha, beta, gamma)

        if self.cfg.metric_mode == "openbci_focus":
            raw = self._openbci_focus_score(alpha, beta)
            raw = self._stabilize_raw(raw)
            smoothed = self._update_ema(raw)
            metric = smoothed  # already in [0,1]
            metric = max(0.0, min(1.0, metric))
            alpha_uV = math.sqrt(max(alpha, 0.0) + self.cfg.eps)
            beta_uV = math.sqrt(max(beta, 0.0) + self.cfg.eps)
            print(f"Metric={metric:.3f}")
            return metric

        raw = self._compute_raw_relaxation(delta, theta, alpha, beta, gamma)
        raw = self._stabilize_raw(raw)
        smoothed = self._update_ema(raw)
        metric = self._sigmoid_scale(smoothed)
        metric = max(0.0, min(1.0, metric))  # clamp

        print(f"Metric={metric:.3f}")
        return metric

# ---------------------------------------------------------------------------
# Example usage with CYTON + DAISY
# ---------------------------------------------------------------------------

class RelaxationVideo:
    def __init__(self, media_path: str, screen: int = 0):
        self.media_path = media_path
        self.screen = screen
        self.process = None

    def start(self):
        self.stop()
        if not self.media_path:
            print("No media path set for this video.")
            return
        if not os.path.exists(self.media_path):
            print(f"Warning: media file not found: {self.media_path}")
        try:
            self.process = subprocess.Popen(
                [
                    "mpv",
                    "--fs",
                    f"--screen={self.screen}",
                    "--loop-file=inf",
                    "--really-quiet",
                    "--no-terminal",
                    "--profile=sw-fast",
                    "--video-sync=audio",
                    self.media_path,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            print("mpv is not installed or not in PATH.")
            self.process = None

    def stop(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
# ---------------------------------------------------------------------------
# Example usage with CYTON + DAISY
# ---------------------------------------------------------------------------
def set_system_volume_percent(percent: int):
    """
    Set system volume on Linux using 'amixer' command.
    Avoid redundant subprocess calls when the requested volume has not changed.
    """
    global last_volume_percent
    try:
        pct = int(max(0, min(100, percent)))
        if last_volume_percent == pct:
            return pct
        subprocess.run(
            ['amixer', 'sset', 'Master', f'{pct}%'],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        last_volume_percent = pct
        return pct
    except Exception:
        return last_volume_percent if last_volume_percent is not None else 0

# -------preparing the board for streaming------




#USE_SIMULATION = False # <-- for testing the GUI, set to false when not testing




def run_eeg():
    import time
    import numpy as np

    # --- Board setup for Cyton + Daisy ---
    params = BrainFlowInputParams()
    # IMPORTANT: set this to your actual serial port on the Pi, e.g.:
    params.serial_port = "/dev/ttyUSB0"

    board_id = BoardIds.CYTON_DAISY_BOARD.value  # 16-channel Cyton + Daisy
    board = BoardShim(board_id, params)

    BoardShim.enable_dev_board_logger()  # optional but useful

    board.prepare_session()
    board.start_stream()  # you can pass a buffer size or streamer if you want

    sampling_rate = BoardShim.get_sampling_rate(board_id)  # should be 125 Hz
    eeg_channels = BoardShim.get_eeg_channels(board_id)[:8] 

    window_sec = 10.0
    num_samples = int(sampling_rate * window_sec)

    cfg = RelaxationConfig(
        sampling_rate=sampling_rate,
        eeg_channels=eeg_channels,
        metric_channels=eeg_channels,  # use all EEG channels for the band-power metric
        window_sec=5.0,
        # Make the metric less twitchy
        smoothing_alpha=0.08,

        # For log_ab, the natural "neutral" point is near 0 (alpha ~= beta)
        sigmoid_gain=3.0,
        sigmoid_center=0.25,

        # Choose one:
        # metric_mode="ratio",   # weighted ratio of (alpha + w*theta + 0.25*delta) vs total
        metric_mode="log_ab",    # log(alpha) - log(beta): stable, less blow-up
        # metric_mode="openbci_focus",

        # If you switch back to ratio, keep theta downweighted to avoid "tired" inflating the score
        theta_weight=0.35,
        delta_weight=0.25,

        # (OpenBCI gating params kept for later experimentation)
        alpha_threshold_uV=1.0,
        beta_threshold_uV=1.0,
        alpha_noise_cap_uV=4.0,
        gate_gain=8.0,
    )
    relax = RelaxationMetric(cfg)
    # -------preparing the board for streaming------



    # -------- VIDEO PLAYER INIT ----------

    video_paths = [
        "capstone media files/handpan2.mp4",
        "capstone media files/jelly.mp4",
        "capstone media files/lava.mp4",
        "capstone media files/trippy3.mp4",
    ]
    global pos, videos, video

    videos = [RelaxationVideo(path, screen=1) for path in video_paths]
    pos = 0
    video = videos[pos]
    window = 100
    window_metric = 0.0
    i = 0
    flag = True
    set_system_volume_percent(30)
    video.start()
    last_active_check = 0.0

    ui.NowPlayingLabel.setText("Now Playing: Handpan Audio")
    ui.NowShowingLabel.setText("Now Showing: RGB Visual")

    metric_scores = [0.0, 0.0, 0.0, 0.0]

    MIN_VOL = 0.02   # very quiet
    MAX_VOL = 0.9    # safe comfortable cap

    MIN_VOL_SYS = 30
    MAX_VOL_SYS = 60 # edit

    print(f"Sampling rate: {sampling_rate} Hz, using 8 EEG channels: {eeg_channels}")
    print("Streaming… press Ctrl+C to stop.")
    print("Calibrating (~20 sec per tone)")

    try:            
        while  not shutdown_requested:
            data = board.get_current_board_data(num_samples)
            if data.shape[1] < num_samples:
                # not enough data yet
                time.sleep(0.1)
                continue
            

            global current_metric, current_bands, vol_sys # define gloval vars for use in the GUI 
            """
            if USE_SIMULATION:
                current_metric = 0.5 + 0.4*np.sin(time.time()/5)

                current_bands = np.array([
                    20 + 10*np.sin(time.time()/6),   # delta
                    35 + 8*np.sin(time.time()/5),    # theta
                    60 + 20*np.sin(time.time()/4),   # alpha
                    25 + 6*np.sin(time.time()/3),    # beta
                    10 + 4*np.sin(time.time()/2)     # gamma
                ])
                vol_sys = 20 + int(40 * current_metric)

                time.sleep(0.2)
                continue
                """
            # DO NOT DELETE 
            # this code block is for the active channel display in the GUI 

            global channel_model, last_active_channels, last_debug_print_time

            now = time.time()
            if now - last_active_check >= 2.0:
                active_list = []
                for ch in eeg_channels:
                    signal = np.asarray(data[ch], dtype=float)

                    if signal.size < 10:
                        continue

                    std_val = np.std(signal)
                    ptp_val = np.ptp(signal)
                    diff_std = np.std(np.diff(signal))

                    score = 0

                    if std_val > 1.0:
                        score += 1
                    if std_val < 150.0:
                        score += 1
                    if ptp_val < 1000:
                        score += 1
                    if diff_std > 0.2:
                        score += 1

                    if score >= 3:
                        active_list.append(f"Channel {ch}")

                last_active_channels = active_list
                last_active_check = now
                # if channel_model is not None:
                #     channel_model.clear()
                #     for name in active_list:
                #         item = QtGui.QStandardItem(name)
                #         channel_model.appendRow(item)
            # DO NOT DELETE 
            # this code block is for the active channel display in the GUI         

            

            delta, theta, alpha, beta, gamma = relax._compute_band_powers(data)
            current_bands = [delta, theta, alpha, beta, gamma]

            metric = relax.update(data)
            relax._latest_bands = None
            current_metric = metric

            is_relaxed = metric > 0.7  # you can tweak this threshold
            #vol = MIN_VOL + (MAX_VOL - MIN_VOL) * metric
            

            # ensure the system volume is not changed during calibration 
            if(flag == False): 
                
                vol_sys = MIN_VOL_SYS + int((MAX_VOL_SYS - MIN_VOL_SYS) * metric)
                set_system_volume_percent(vol_sys)


            # -------------- PROGRAM START HERE ! ----------------
            else:
                vol_sys = 30
                set_system_volume_percent(vol_sys)
                
            if now - last_debug_print_time >= 1.0:
                print(f"Relaxation metric: {metric:.3f}  |  relaxed={is_relaxed} |  system volume={vol_sys}%  | current video={video.media_path}")
                last_debug_print_time = now

            window_metric += metric
            i += 1

            if(i == window and flag):
                window_metric = window_metric/window
                metric_scores[pos] = window_metric
                window_metric = 0.0
                i = 0
                pos += 1
                if(pos == 4):
                    pos = np.argmax(metric_scores)
                    flag = False
                    print("Video selected, begin volume adjustment")

                video.stop()
                video = videos[pos]
                    # # This is logic to display the current audio / Visual in the GUI
                    # if pos == 0:
                    #     ui.NowPlayingLabel.setText("Now Playing: Flute Audio")
                    #     ui.NowShowingLabel.setText("Now Showing: Flute Visual")
                    #
                    # elif pos == 1:
                    #     ui.NowPlayingLabel.setText("Now Playing: Ocean Audio")
                    #     ui.NowShowingLabel.setText("Now Showing: Jellyfish Visual")
                    #
                    # elif pos == 2:
                    #     ui.NowPlayingLabel.setText("Now Playing: Flute Audio")
                    #     ui.NowShowingLabel.setText("Now Showing: Lava Lamp Visual Distortion")
                    #
                    # elif pos == 3:
                    #     ui.NowPlayingLabel.setText("Now Playing: Chimes Audio")
                    #     ui.NowShowingLabel.setText("Now Showing: Monocolor Pattern Visual")
                video.start()

            time.sleep(0.25)  # slightly lighter update load for smoother playback


    except KeyboardInterrupt:
        print("Stopping…")
    finally:
        board.stop_stream()
        board.release_session()
        video.stop()



if __name__ == "__main__":


 ## GUI CODE EXTENDED !

 

    # Start EEG streaming in background thread
    app = QtWidgets.QApplication(sys.argv)

    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    from PyQt5 import QtGui


    def scale_children(parent, factor):
        for w in parent.findChildren(QtWidgets.QWidget):
            #try x,y,width,height = w.geometry().getRect()
            g = w.geometry()
            w.setGeometry(int(g.x()*factor),int(g.y()*factor),int(g.width()*factor),int(g.height()*factor))
    # For the active channel display, use the number of samples to determine if a channel is active or not

    ui.setupUi(MainWindow) # allows for the setup from ActualUI.py to be used in this file
    # scale_children(MainWindow,0.65)
    old_widget = MainWindow.centralWidget()

    scroll = QtWidgets.QScrollArea()
    scroll.setWidget(old_widget)
    scroll.setWidgetResizable(False)
    scroll.viewport().setAttribute(Qt.WA_AcceptTouchEvents, True)
    scroll.setAttribute(Qt.WA_AcceptTouchEvents, True)
    QScroller.grabGesture(scroll.viewport(), QScroller.TouchGesture)

    scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
    scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
    MainWindow.setCentralWidget(scroll)

    old_widget.adjustSize()

    screen = get_screen_geometry(app, preferred_index=0)
    MainWindow.resize(int(screen.width()*0.9), int(screen.height()*0.9))
    MainWindow.move(screen.x(), screen.y())

    #MainWindow.show()
    audio_names = [
        "Flute Audio",
        "Night Audio",
        "Visual Distortion Audio",
        "Waterfall Audio"
    ]   

    visual_names = [
        "Flute Visual",
        "Night Visual",
        "Visual Distortion",
        "Waterfall Visual"
    ]
    
    channel_model = QtGui.QStandardItemModel()
    ui.ActiveChannelList.setModel(channel_model)

    session_seconds = 0 # the session duration
    
    
    def play_media():
     global video
     if video is not None:
        video.start()

    def stop_media():
        global video
        if video is not None:
            video.stop()

    def cycle_media():
        global video, videos, pos

        if video is not None:
            video.stop()

            pos = (pos + 1) % len(videos)
            video = videos[pos]
            video.start()

            ui.NowPlayingLabel.setText(f"Now Playing: {audio_names[pos]}")
            ui.NowShowingLabel.setText(f"Now Showing: {visual_names[pos]}")

    # connect buttons with playback functions defined above
    ui.PlayButton.pressed.connect(play_media)
    ui.StopButton.pressed.connect(stop_media)
    ui.Cyclebutton.pressed.connect(cycle_media)


    eeg_thread = threading.Thread(target=run_eeg, daemon=True)
    eeg_thread.start()
    
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   

    fig = Figure()
    canvas = FigureCanvas(fig)

    ax = fig.add_subplot(111)

    layout = QtWidgets.QVBoxLayout(ui.PSDBarPlot)
    layout.addWidget(canvas)
    bands = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
    bar_container = ax.bar(bands, current_bands)
    ax.set_ylabel("Power")
    ax.set_title("EEG Band Powers")

    plot_timer = QtCore.QTimer()

    # plot the band powers in the GUI, updated every 0.5 seconds. Uses the global variable current_bands which is updated in the EEG loop.
    def update_plot():
        values = current_bands
        ymax = max(1.0, max(values) * 1.15)

        for bar, value in zip(bar_container, values):
            bar.set_height(value)

        ax.set_ylim(0, ymax)
        canvas.draw_idle()

    plot_timer.timeout.connect(update_plot)
    #plot_timer.start(2000)  # update every 2 seconds and not every 500 ms to increase performance 
    timer = QtCore.QTimer()
    
    quotes = [
    "Relax... Let go of tension",
    "Slow your breathing",
    "Release your shoulders",
    "Stay present in this moment",
    "Calm mind, steady body",
    "Breathe in... breathe out",
    "Ease into the flow",
    "Quiet the noise",
    "Loosen your jaw",
    "Unclench your hands",
    "Drop the tension in your neck",
    "Let your thoughts drift by",
    "You are safe in this moment",
    "Settle into stillness",
    "One breath at a time",
    "Release what you don't need",
    "Let the room grow quiet",
    "Feel your body unwind",
    "Return to center",
    "Allow calm to rise",
    "Nothing to solve right now",
    "Let your shoulders sink",
    "Rest your mind gently",
    "Ease the pressure",
    "Focus on this breath",
    "Inhale peace... exhale stress",
    "Let calm take over",
    "Slow down your thoughts",
    "You are doing well",
    "Be here now",
    "Sink into comfort",
    "Relax deeper with each breath",
    "Let your body feel heavy",
    "Quiet mind, steady breath",
    "The moment is enough",
    "Soften your posture",
    "Peace begins here",
    "Let the noise fade away",
    "Your only task is to breathe",
    "Choose calm right now"
    ]

    

    quote_index = 0
    last_quote_change = QtCore.QTime.currentTime()
    
    # function to allow the session to be exited via button click 
    def exit_session():
        def save_csv():
            import csv
            from datetime import datetime
        
            filename = datetime.now().strftime("session_%Y%m%d_%H%M%S.csv")
        
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
        
                writer.writerow([
                    "Time_s",
                    "Metric",
                    "Media",
                    "Delta",
                    "Theta",
                    "Alpha",
                    "Beta",
                    "Gamma"
                ])
        
                for i in range(len(session_time_data)):
                    bands = session_band_data[i] if i < len(session_band_data) else [None]*5
        
                    writer.writerow([
                        session_time_data[i],
                        session_metric_data[i],
                        session_media_data[i],
                        bands[0],
                        bands[1],
                        bands[2],
                        bands[3],
                        bands[4]
                    ])
        
            print(f"Saved {filename}")
        

        global shutdown_requested

        # close main window
        MainWindow.close()
        
        shutdown_requested = True
        # Open session stats window
        stats_window = QtWidgets.QMainWindow()
        stats_ui = StatsUI()
        stats_ui.setupUi(stats_window)
        # scale_children(stats_window,0.65)

        old_widget_stats = stats_window.centralWidget()

        stats_scroll = QtWidgets.QScrollArea()
        stats_scroll.setWidget(old_widget_stats)
        stats_scroll.setWidgetResizable(False)
        stats_scroll.viewport().setAttribute(Qt.WA_AcceptTouchEvents, True)
        stats_scroll.setAttribute(Qt.WA_AcceptTouchEvents, True)
        QScroller.grabGesture(stats_scroll.viewport(), QScroller.TouchGesture)

        stats_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        stats_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        stats_window.setCentralWidget(stats_scroll)

        old_widget_stats.adjustSize()

        screen = get_screen_geometry(app, preferred_index=0)
        stats_window.resize(int(screen.width()*0.9), int(screen.height()*0.9))
        stats_window.move(screen.x(), screen.y())
            # Calculations
        total_time = session_seconds
        total_minutes = round(total_time / 60)

        relaxed_points = sum(1 for x in session_metric_data if x > 0.7)
        time_relaxed = relaxed_points    # logged every second

        avg_metric = 0
        if len(session_metric_data) > 0:
            avg_metric = sum(session_metric_data) / len(session_metric_data)

        # determine best audio from metric_scores if available
        names = ["Flute", "Night", "Visual Distortion", "Waterfall"]

        from collections import defaultdict 
        audio_scores = defaultdict(list)
        for i in range(len(session_metric_data)):
            audio = session_media_data[i]
            metric = session_metric_data[i]
            audio_scores[audio].append(metric)

        if len(audio_scores) > 0:
            avg_scores = {
                audio: sum(vals) / len(vals) 
                for audio, vals in audio_scores.items() 
            }
            best_audio = max(avg_scores, key=avg_scores.get)
        else:
            best_audio = "N/A"

        # Display labels
        
        stats_ui.SessionDurationLabel.setText(
            f"<html><body><p><span style='font-size:16pt;'>"
            f"Session Duration: {total_minutes:02d}:{total_time%60:02d}"
            f"</span></p></body></html>"
        )

        stats_ui.YourTimeRelaxedLabel.setText(
            f"<html><body><p><span style='font-size:16pt;'>"
            f"Your Time Relaxed: {round(time_relaxed / 60):02d}:{time_relaxed%60:02d}"
            f"</span></p></body></html>"
        )

        stats_ui.AverageRelaxationScoreLabel.setText(
            f"<html><body><p><span style='font-size:16pt;'>"
            f"Average Relaxation Score: {avg_metric*100:.0f}%"
            f"</span></p></body></html>"
        )

        stats_ui.StimulatingAudioLabel.setText(
            f"<html><body><p><span style='font-size:16pt;'>"
            f"Most Engaging Audio: {best_audio}"
            f"</span></p></body></html>"
        )
        
        # Relaxation vs time graph
        fig1 = Figure()
        canvas1 = FigureCanvas(fig1)

        ax1 = fig1.add_subplot(111)
        ax1.plot(session_time_data, session_metric_data)
        ax1.set_title("Relaxation vs Time (s)")
        ax1.set_xlabel("Seconds")
        ax1.set_ylabel("Metric")

        layout1 = QtWidgets.QVBoxLayout(stats_ui.RelaxationVsTime)
        layout1.addWidget(canvas1)

        # Audio vs Time graph
        fig2 = Figure()
        canvas2 = FigureCanvas(fig2)
        
        ax2 = fig2.add_subplot(111)
        ax2.plot(session_time_data,session_media_data,drawstyle='steps-post')
        ax2.set_title("Audio Selection vs Time")
        ax2.set_xlabel("Seconds")
        ax2.set_ylabel("Audio")
        ax2.set_yticks([0,1,2,3])
        ax2.set_yticklabels(["Flute", "Night", "Distortion", "Waterfall"])




        layout2 = QtWidgets.QVBoxLayout(stats_ui.AudioVsTime)
        layout2.addWidget(canvas2)

        # buttons
        stats_ui.ExitStatsButton.pressed.connect(stats_window.close)

        def redo_session():
            stats_window.close()
            QtWidgets.QApplication.quit()
            os.execv(sys.executable, [sys.executable] + sys.argv)
        stats_ui.RedoSession.pressed.connect(redo_session)

        stats_window.show()

        # keep reference alive
        global stats_ref
        stats_ref = stats_window

    # connect the button to the callback function above
    ui.ExitSessionButton.clicked.connect(exit_session)

   
    

    elapsed_seconds = 0
    elapsed_minutes = 0


    def update_time():
     global elapsed_seconds, elapsed_minutes

     elapsed_seconds += 1

     if elapsed_seconds == 60:
         elapsed_seconds = 0
         elapsed_minutes += 1

     ui.Time.display(f"{elapsed_minutes:02d}:{elapsed_seconds:02d}")

    # function to log relaxation every second  
    def log_metric():
     global session_seconds

     session_seconds += 1

     session_time_data.append(session_seconds)
     session_metric_data.append(current_metric)
     session_media_data.append(audio_names[pos])
     session_band_data.append(current_bands.copy())
     session_volume_data.append(vol_sys)


    
    def update_volume_bar():
        ui.VolumeBar.setValue(vol_sys)

    # Progress bar
    def play_next_frame():
        global playback_index, current_metric, current_bands, vol_sys, pos

        current_metric = session_metric_data[playback_index]
        vol_sys = session_volume_data[playback_index]
        pos = session_media_data[playback_index]
        current_bands = session_band_data[playback_index]
        playback_index += 1


    def update_gui():
     global quote_index, last_quote_change, session_seconds, last_relaxed_state
     percent = int(current_metric * 100)
     ui.RelxationBar.setValue(percent)
     relaxed = current_metric > 0.7
     
     #update_volume_bar()
     # RELAXED
     """
     if relaxed != last_relaxed_state:
        if relaxed:

            ui.DynamicRelaxationColor.setStyleSheet(
                "background-color: rgb(87, 227, 137);"
                "border-radius: 60px;"
            )

            ui.DynamicRelaxationLabel.setText(
                "<html><body><p align='center'>"
                "<span style='font-size:16pt;'>Relaxed</span>"
                "</p></body></html>"
            )

            ui.RelaxationQuote.setText(
                "<html><body><p align='center'>"
                "<span style='font-size:24pt;'>Hold Steady</span>"
                "</p></body></html>"
            )

        # NOT RELAXED
        else:

            ui.DynamicRelaxationColor.setStyleSheet(
                "background-color: rgb(192, 28, 40);"
                "border-radius: 60px;"
            )

            ui.DynamicRelaxationLabel.setText(
                "<html><body><p align='center'>"
                "<span style='font-size:16pt;'>Not Relaxed</span>"
                "</p></body></html>"
            )
            
            # Change quote every 5 sec
            if last_quote_change.secsTo(QtCore.QTime.currentTime()) >= 5:
                quote_index = (quote_index + 1) % len(quotes)
                last_quote_change = QtCore.QTime.currentTime()

            ui.RelaxationQuote.setText(
                f"<html><body><p align='center'>"
                f"<span style='font-size:24pt;'>{quotes[quote_index]}</span>"
                f"</p></body></html>"
            )
     last_relaxed_state = relaxed
        """
        
     
    # session timer
    clock_timer = QtCore.QTimer()
    clock_timer.timeout.connect(update_time)
    #clock_timer.start(1000)
    #update gui timer
    timer.timeout.connect(update_gui)
    timer.start(1500) # initially set to 400 ms, setting it to 800 ms to reduce load and improve performance, can be tweaked as needed
    # log timer for the relaxation metric
    log_timer = QtCore.QTimer()
    log_timer.timeout.connect(log_metric)
    log_timer.start(1000)
     
    MainWindow.show()

    sys.exit(app.exec_())


 ## GUI CODE EXTENDED !
 ## GUI CODE EXTENDED !