"""
Simple BrainFlow streamer for Cyton + Daisy that computes band powers.
Usage:
  python streamer.py --port /dev/ttyUSB0 --window 2.0 --interval 1.0
"""
import time
import argparse
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

BANDS = {
    'delta': (1.0, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta':  (13.0, 30.0),
    'gamma': (30.0, 45.0),
}

def compute_bandpowers(channel_data, fs, band_defs):
    """
    channel_data: 1D numpy array
    fs: sampling rate
    returns dict of band->power (power computed by integrating PSD)
    """
    n = len(channel_data)
    if n < 2:
        return {b: 0.0 for b in band_defs}
    # detrend
    x = channel_data - np.mean(channel_data)
    # windowed FFT -> PSD estimate
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default='/dev/ttyUSB0', help='serial port for Cyton/Daisy')
    parser.add_argument('--window', type=float, default=2.0, help='window length in seconds for PSD')
    parser.add_argument('--interval', type=float, default=1.0, help='how often (s) to compute band powers')
    args = parser.parse_args()

    params = BrainFlowInputParams()
    params.serial_port = args.port

    board_id = BoardIds.CYTON_DAISY_BOARD.value
    board = BoardShim(board_id, params)
    try:
        board.prepare_session()
        sampling_rate = BoardShim.get_sampling_rate(board_id)
        eeg_channels = BoardShim.get_eeg_channels(board_id)
        n_window_samples = max(4, int(round(args.window * sampling_rate)))
        board.start_stream()

        print(f"Started stream on {args.port}, sampling_rate={sampling_rate}, eeg_channels={eeg_channels}")
        while True:
            # get the latest window
            raw = board.get_current_board_data(n_window_samples)
            if raw is None or raw.size == 0:
                time.sleep(0.1)
                continue
            results = {}
            for ch_idx in eeg_channels:
                if ch_idx >= raw.shape[0]:
                    continue
                ch_data = raw[ch_idx, -n_window_samples:]
                bp = compute_bandpowers(ch_data, sampling_rate, BANDS)
                results[f"ch{ch_idx}"] = bp
            ts = time.time()
            # print a compact line per interval
            out = {"ts": ts}
            out.update(results)
            print(out, flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            board.stop_stream()
        except Exception:
            pass
        try:
            board.release_session()
        except Exception:
            pass
        print("Stopped and released session.")

if __name__ == '__main__':
    main()