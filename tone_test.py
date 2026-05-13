"""Play an MP4 as-is in mpv and test live system-volume changes.

- Uses the MP4's own embedded audio track.
- Video loops forever on the selected screen / HDMI output.
- System output volume is changed in real time using amixer.
"""

import json
import os
import socket
import subprocess
import threading
import time
from pathlib import Path


VIDEO_PATH = "capstone media files/handpan2.mp4"
VIDEO_SCREEN = 1
VIDEO_DISPLAY = ":1"
XAUTHORITY_PATH = None
MPV_AUDIO_DEVICE = None
INITIAL_GAIN = 0.30


current_gain = INITIAL_GAIN
current_gain_lock = threading.Lock()
stop_flag = False


def get_gain() -> float:
    with current_gain_lock:
        return current_gain


def set_gain(new_gain: float) -> None:
    global current_gain
    new_gain = max(0.0, min(1.0, float(new_gain)))
    with current_gain_lock:
        current_gain = new_gain


def gain_to_percent(gain: float) -> float:
    return max(0.0, min(1.0, float(gain))) * 100.0


def set_system_volume(gain: float) -> None:
    percent = int(round(gain_to_percent(gain)))
    subprocess.run(
        ["amixer", "sset", "Master", f"{percent}%"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def launch_player(video_path: str) -> subprocess.Popen:
    env = os.environ.copy()
    if VIDEO_DISPLAY:
        env["DISPLAY"] = VIDEO_DISPLAY
    if XAUTHORITY_PATH:
        env["XAUTHORITY"] = XAUTHORITY_PATH

    cmd = [
        "mpv",
        video_path,
        "--fs",
        f"--fs-screen={VIDEO_SCREEN}",
        "--loop-file=inf",
        "--vo=gpu",
        "--gpu-context=x11egl",
        "--keep-open=no",
    ]

    if MPV_AUDIO_DEVICE:
        cmd.append(f"--audio-device={MPV_AUDIO_DEVICE}")

    print(f"Launching video: {video_path}")
    print(f"DISPLAY={env.get('DISPLAY', '(not set)')}")
    print(f"XAUTHORITY={env.get('XAUTHORITY', '(not set)')}")

    return subprocess.Popen(cmd, env=env)


def demo_gain_changes() -> None:
    global stop_flag

    for value in [0.40]:
        if stop_flag:
            return
        time.sleep(3)
        set_gain(value)
        try:
            set_system_volume(value)
            print(f"Updated system volume to {int(round(gain_to_percent(value)))}%")
        except subprocess.CalledProcessError as exc:
            print(f"System volume update failed: {exc}")
            return


def main() -> None:
    video_file = Path(VIDEO_PATH)

    if not video_file.exists():
        raise FileNotFoundError(f"Video file not found: {video_file}")

    set_system_volume(get_gain())
    player = launch_player(str(video_file))

    gain_thread = threading.Thread(target=demo_gain_changes, daemon=True)
    gain_thread.start()

    try:
        player.wait()
    except KeyboardInterrupt:
        print("Stopping playback...")
    finally:
        global stop_flag
        stop_flag = True

        if player.poll() is None:
            player.terminate()
            try:
                player.wait(timeout=3)
            except subprocess.TimeoutExpired:
                player.kill()


if __name__ == "__main__":
    main()