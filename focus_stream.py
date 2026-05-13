"""
Simple BrainFlow streamer for Cyton + Daisy that computes band powers and focus state.
Usage:
  python focus_stream.py --port /dev/ttyUSB0 --window 2.0 --interval 1.0 --calibrate 5. 
  No future changes need to be made to this file.
"""
import time
import argparse
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from focus import (
    compute_bandpowers, 
    compute_focus_metric_from_bands,
    compute_relaxation_metric_from_bands,
    calibrate_baseline, 
    FocusState
)
from volume_control import map_metric_to_volume, set_system_volume_percent, TonePlayer

BANDS = {
    'delta': (1.0, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta':  (13.0, 30.0),
    'gamma': (30.0, 45.0),
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default='/dev/ttyUSB0', help='serial port for Cyton/Daisy')
    parser.add_argument('--window', type=float, default=2.0, help='window length in seconds for PSD')
    parser.add_argument('--interval', type=float, default=1.0, help='how often (s) to compute band powers')
    parser.add_argument('--calibrate', type=float, default=10.0, help='seconds to calibrate baseline (set 0 to skip)')
    parser.add_argument('--smooth', type=float, default=0.3, help='EMA alpha for smoothing (0..1)')
    parser.add_argument('--high-z', type=float, default=1.0, help='z threshold to enter focused state')
    parser.add_argument('--low-z', type=float, default=0.5, help='z threshold to exit focused state')
    parser.add_argument('--hold', type=float, default=0.5, help='seconds metric must exceed high-z to enter focused')
    parser.add_argument('--channels', type=str, default='', help='comma-separated EEG channel indices to use (e.g. 0,1,2). Leave empty to auto-detect.')
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

        def detect_active_channels(board, candidates, n_samples, tries=5, std_threshold=1e-6):
            """
            Sample a few times and pick channels whose std > std_threshold (averaged over tries).
            Returns a filtered list of channel indices (subset of candidates).
            """
            stds = {ch: [] for ch in candidates}
            for _ in range(tries):
                raw = board.get_current_board_data(n_samples)
                if raw is None or raw.size == 0:
                    time.sleep(0.05)
                    continue
                for ch in candidates:
                    if ch >= raw.shape[0]:
                        continue
                    ch_data = raw[ch, -n_samples:]
                    stds[ch].append(float(np.std(ch_data)))
                time.sleep(0.02)
            mean_stds = {ch: (np.mean(stds[ch]) if len(stds[ch])>0 else 0.0) for ch in candidates}
            active = [ch for ch, s in mean_stds.items() if s > std_threshold]
            return active if active else list(candidates)

        # determine which channels to use (do this once, before calibration)
        if args.channels:
            try:
                use_channels = [int(x) for x in args.channels.split(',') if x.strip()!='']
            except Exception:
                use_channels = eeg_channels
        else:
            n_detect_samples = max(4, int(round(0.5 * sampling_rate)))  # 0.5s window
            use_channels = detect_active_channels(board, eeg_channels, n_detect_samples, tries=6, std_threshold=1e-6)

        print(f"Using EEG channels: {use_channels}")

        # Optional calibration (use only the selected channels)
        if args.calibrate and args.calibrate > 0:
            print(f"Calibrating baseline for {args.calibrate:.1f}s ... please relax")
            baseline_mean, baseline_std = calibrate_baseline(board, use_channels, sampling_rate, args.window, args.calibrate, BANDS)
            print(f"Calibration done: mean={baseline_mean:.4g}, std={baseline_std:.4g}")
        else:
            baseline_mean, baseline_std = 0.0, 1.0

        focus = FocusState(smooth_alpha=args.smooth, high_z=args.high_z, low_z=args.low_z, hold_time=args.hold)
        # optional: start a local tone player instead of changing system volume
        tone = TonePlayer(frequency=440.0); tone.start()

        while True:
            raw = board.get_current_board_data(n_window_samples)
            if raw is None or raw.size == 0:
                time.sleep(0.1)
                continue
            
            channel_metrics = []
            results = {}
            per_ch_metrics = {}
            for ch_idx in use_channels:
                if ch_idx >= raw.shape[0]:
                    continue
                ch_data = raw[ch_idx, -n_window_samples:]
                bp = compute_bandpowers(ch_data, sampling_rate, BANDS)
                results[f"ch{ch_idx}"] = bp
                # compute both metrics for debugging
                focus_m = compute_focus_metric_from_bands(bp)
                relax_m = compute_relaxation_metric_from_bands(bp)
                per_ch_metrics[ch_idx] = {"focus_m": float(focus_m), "relax_m": float(relax_m)}
                # default combined metric: mean of relaxation metric (higher => relaxed)
                channel_metrics.append(relax_m)
            ts = time.time()
            combined_metric = float(np.mean(channel_metrics)) if channel_metrics else 0.0

            # update focus state using focus metric if you still want focus detection
            # state, smoothed_z, raw_z = focus.update(combined_metric, baseline_mean, baseline_std, ts=ts)

            # map relaxation metric -> volume (no invert needed)
            vol = map_metric_to_volume(combined_metric, baseline_mean, baseline_std,
                                       invert=False, z_min=-2, z_max=2, min_v=0.01, max_v=0.9, curve='exp')
            # apply to tone or system
            tone.set_amplitude(vol)
            # set_system_volume_percent(int(vol*100))

            # debug output
            out = {
                "ts": ts,
                "combined_relax_metric": combined_metric,
                "baseline_mean": baseline_mean,
                "baseline_std": baseline_std,
                "volume": float(vol),
                "per_channel_metrics": per_ch_metrics,
            }
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