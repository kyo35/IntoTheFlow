# IntoTheFlow

IntoTheFlow is an EEG-driven relaxation system that uses OpenBCI brainwave data to select and control calming audiovisual media in real time. The system reads EEG data through BrainFlow, computes a relaxation metric from EEG band powers, and adjusts media playback and system volume based on the user’s relaxation state.

The project was designed as an immersive relaxation experience using a geodesic dome, projector visuals, ambient audio, and real-time EEG feedback.

---

# Features

- Real-time EEG streaming using BrainFlow
- OpenBCI Cyton + Daisy support
- EEG band power extraction
- Relaxation metric based on EEG activity
- Exponential smoothing for stable relaxation scoring
- Media playback using MPV
- Dynamic system volume control
- PyQt-based graphical interface
- Calibration phase for determining optimal media
- Multi-video relaxation environment support

---

# Hardware Used

- Raspberry Pi 5
- OpenBCI Cyton + Daisy board
- OpenBCI USB Dongle
- EEG electrodes
- Projector / external display
- Speakers
- Touchscreen display (optional)

---

# Repository Structure

```text
IntoTheFlow/
│
├── relaxation_metric.py      # Main project file
├── volume_control.py         # Volume testing/control
├── tone_test.py              # Earlier audio testing
├── streamer.py               # EEG streaming utilities
├── focus.py                  # Experimental focus logic
├── focus_stream.py           # Alternate stream logic
│
├── ActualUI.py               # GUI file
├── Brainflow.ui              # Qt Designer UI
├── Flow.ui                   # Qt Designer UI
├── SessionStats.py           # Session statistics window
├── SessionStats.ui           # Qt Designer UI
│
└── media/                    # Media files (user supplied)
```

---

# How the System Works

The system continuously streams EEG data from the OpenBCI Cyton + Daisy board through BrainFlow. The EEG signals are separated into standard EEG frequency bands:

- Delta
- Theta
- Alpha
- Beta
- Gamma

The relaxation metric is computed using the relationship between relaxation-associated activity (primarily alpha and theta) and higher-arousal activity (primarily beta/gamma). The metric is smoothed using an exponential moving average to prevent rapid fluctuations caused by noise or artifacts.

During calibration, the system plays several media samples and records the relaxation metric for each one. The media that produces the best relaxation response is selected for continued playback.

The system can then dynamically adjust media volume based on the user's relaxation state.

---

# Recommended Environment

- Python 3.9+
- Raspberry Pi OS / Ubuntu Linux
- OpenBCI Cyton + Daisy
- BrainFlow
- MPV media player

---

# Creating a Virtual Environment

It is strongly recommended to use a Python virtual environment.

## Linux / Raspberry Pi / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## Windows

```bash
python -m venv venv
venv\Scripts\activate
```

After activation, upgrade pip:

```bash
python -m pip install --upgrade pip
```

---

# Installing Dependencies

Install the required Python packages:

```bash
python -m pip install brainflow numpy matplotlib PyQt5
```

On Raspberry Pi / Ubuntu Linux, also install:

```bash
sudo apt update
sudo apt install mpv alsa-utils python3-pyqt5
```

If PyQt5 causes issues through pip on Linux, install it through apt instead:

```bash
sudo apt install python3-pyqt5
```

---

# BrainFlow Setup

This project uses BrainFlow to interface with the OpenBCI Cyton + Daisy board.

Install BrainFlow:

```bash
python -m pip install brainflow
```

BrainFlow documentation:
https://brainflow.readthedocs.io/

---

# OpenBCI Setup

The project is configured for the OpenBCI Cyton + Daisy board.

BrainFlow board configuration:

```python
BoardIds.CYTON_DAISY_BOARD
```

You must also specify the correct serial port.

Example:

```python
params.serial_port = "/dev/ttyUSB0"
```

To locate your serial device on Linux/Raspberry Pi:

```bash
ls /dev/ttyUSB*
ls /dev/ttyACM*
```

---

# Serial Port Permissions

If you encounter permission issues:

```bash
sudo usermod -a -G dialout $USER
```

Then reboot or log out and back in.

---

# Media File Setup

Update the media file paths inside `relaxation_metric.py`.

Example:

```python
video_paths = [
    "media/video1.mp4",
    "media/video2.mp4",
    "media/video3.mp4",
    "media/video4.mp4"
]
```

Replace these with your actual media file locations.

---

# Running the Project

Activate your virtual environment:

```bash
source venv/bin/activate
```

Then run:

```bash
python relaxation_metric.py
```

Before running, make sure:

- The OpenBCI dongle is plugged in
- The Cyton/Cyton Daisy board is powered on
- The correct serial port is configured
- MPV is installed
- Media paths are correct

---

# Relaxation Metric Overview

The relaxation metric is based on EEG band power activity. BrainFlow is used to compute average band powers from the incoming EEG stream.

The system emphasizes relaxation-associated frequency bands such as alpha and theta while suppressing higher-arousal activity such as beta and gamma.

The metric is smoothed using an exponential moving average:

```text
EMA_new = α * current_value + (1 - α) * EMA_previous
```

This prevents sudden spikes or drops caused by EEG artifacts and produces more stable system behavior.

---

# GUI

The project includes a PyQt-based GUI that displays:

- Current relaxation score
- Session information
- Media currently playing
- Calibration progress
- EEG streaming status

The GUI was designed for touchscreen compatibility on Raspberry Pi systems.

---

# Troubleshooting

## BrainFlow Import Error

Reinstall BrainFlow:

```bash
python -m pip install brainflow
```

Verify your virtual environment is activated.

---

## Serial Port Not Found

Check available devices:

```bash
ls /dev/ttyUSB*
```

Update:

```python
params.serial_port
```

inside the code.

---

## Permission Denied on Serial Port

Run:

```bash
sudo usermod -a -G dialout $USER
```

Then reboot.

---

## MPV Not Found

Install MPV:

```bash
sudo apt install mpv
```

---

## Volume Control Not Working

Install ALSA utilities:

```bash
sudo apt install alsa-utils
```

Test manually:

```bash
amixer sset Master 50%
```

---

## GUI Lag / Video Stuttering

Potential solutions:

- Reduce video resolution
- Use 480p or 720p media instead of 1080p
- Minimize excessive console printing
- Avoid heavy GUI redraw operations
- Run GUI updates in a separate thread

---

# Future Improvements

- Add CSV session logging
- Improve GUI responsiveness
- Add simulation mode without hardware
- Add real-time relaxation plots
- Improve calibration logic
- Add automatic channel quality detection
- Add support for wireless streaming
- Improve media synchronization

---

# Authors

Virginia Tech ECE Capstone Project

Developed using:
- OpenBCI
- BrainFlow
- Python
- PyQt5
- Raspberry Pi

---

# References

BrainFlow Documentation:
https://brainflow.readthedocs.io/

OpenBCI:
https://openbci.com/

PyQt5:
https://pypi.org/project/PyQt5/

MPV:
https://mpv.io/
