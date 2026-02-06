"""
EMG data processing pipeline (per-trial):
1) subtract per-channel mean (center at 0)
2) bandpass 20–450 Hz
3) notch 50 Hz
4) z-score normalize (per channel, within trial)
5) window into 200 ms frames with 50 ms step

Assumptions:
- Data is a pandas DataFrame with time rows and EMG channels as columns.
- Sampling rate (fs) is known or can be estimated from a time column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch


# ---------- Helpers: sampling rate ----------
def infer_fs_from_time(time_s: np.ndarray) -> float:
    """Infer sampling rate from a 1D time array in seconds."""
    dt = np.diff(time_s)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        raise ValueError("Cannot infer fs: time array has no valid increasing diffs.")
    dt_med = np.median(dt)
    return float(1.0 / dt_med)


# ---------- Filters ----------
def bandpass_filter(x: np.ndarray, fs: float, low_hz: float = 20.0, high_hz: float = 450.0, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass along axis=0 (time)."""
    nyq = 0.5 * fs
    low = low_hz / nyq
    high = high_hz / nyq

    if not (0 < low < 1) or not (0 < high < 1) or low >= high:
        raise ValueError(f"Invalid bandpass cutoffs for fs={fs}: low={low_hz}, high={high_hz}")

    b, a = butter(order, [low, high], btype="bandpass")
    return filtfilt(b, a, x, axis=0)


def notch_filter(x: np.ndarray, fs: float, notch_hz: float = 50.0, q: float = 30.0) -> np.ndarray:
    """Zero-phase IIR notch at notch_hz along axis=0 (time)."""
    nyq = 0.5 * fs
    w0 = notch_hz / nyq
    if not (0 < w0 < 1):
        raise ValueError(f"Invalid notch frequency {notch_hz} Hz for fs={fs}")
    b, a = iirnotch(w0=w0, Q=q)
    return filtfilt(b, a, x, axis=0)


# ---------- Normalization ----------
def zscore_per_channel(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score each channel (column) across time."""
    mu = np.mean(x, axis=0, keepdims=True)
    sigma = np.std(x, axis=0, ddof=0, keepdims=True)
    return (x - mu) / (sigma + eps)


# ---------- Windowing ----------
def sliding_window(
    x: np.ndarray,
    fs: float,
    window_ms: float = 200.0,
    step_ms: float = 50.0,
) -> np.ndarray:
    """
    Create overlapping windows.
    Returns array of shape (n_windows, window_samples, n_channels)
    """
    win = int(round(window_ms * 1e-3 * fs))
    step = int(round(step_ms * 1e-3 * fs))

    if win <= 0 or step <= 0:
        raise ValueError("window_ms and step_ms must be > 0.")
    if x.shape[0] < win:
        raise ValueError(f"Signal too short for one window: n={x.shape[0]}, win={win}")

    n = x.shape[0]
    starts = np.arange(0, n - win + 1, step, dtype=int)
    windows = np.stack([x[s : s + win, :] for s in starts], axis=0)
    return windows


# ---------- Full pipeline ----------
def process_emg_trial(
    df: pd.DataFrame,
    emg_cols: list[str],
    *,
    fs: float | None = None,
    time_col: str | None = None,
    bandpass_low: float = 20.0,
    bandpass_high: float = 450.0,
    notch_hz: float = 50.0,
    notch_q: float = 30.0,
    window_ms: float = 200.0,
    step_ms: float = 50.0,
) -> tuple[np.ndarray, float]:
    """
    Process one trial and return:
      windows: (n_windows, window_samples, n_channels)
      fs_used: sampling rate
    """
    if fs is None:
        if time_col is None:
            raise ValueError("Provide fs or time_col to infer fs.")
        time_vals = df[time_col].to_numpy(dtype=float)
        fs = infer_fs_from_time(time_vals)

    # Extract signal matrix: (time, channels)
    x = df[emg_cols].to_numpy(dtype=float)

    # 1) subtract offset (per channel mean for this trial)
    x = x - np.mean(x, axis=0, keepdims=True)

    # 2) bandpass 20–450 Hz
    x = bandpass_filter(x, fs=fs, low_hz=bandpass_low, high_hz=bandpass_high, order=4)

    # 3) notch 50 Hz
    x = notch_filter(x, fs=fs, notch_hz=notch_hz, q=notch_q)

    # 4) z-score normalization
    x = zscore_per_channel(x)

    # 5) windowing (200 ms windows, 50 ms step)
    windows = sliding_window(x, fs=fs, window_ms=window_ms, step_ms=step_ms)

    return windows, float(fs)


# ---------- Example usage ----------
if __name__ == "__main__":
    # Example: df has columns ["t", "ch1", "ch2", "ch3", ...]
    df = pd.read_csv("trial1.csv")  # replace with your file
    emg_cols = [c for c in df.columns if c.startswith("ch")]
    windows, fs_used = process_emg_trial(
        df,
        emg_cols,
        time_col="t",     # seconds
        fs=None,          # infer from time_col
    )
    print("fs =", fs_used)
    print("windows shape =", windows.shape)  # (n_windows, window_samples, n_channels)
