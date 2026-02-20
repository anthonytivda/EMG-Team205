from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch


# ---------- Helpers: sampling rate ----------
def infer_fs_from_time(time_s: np.ndarray) -> float:
    """Infer sampling rate from a 1D time array in seconds."""
    dt = np.diff(time_s)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        raise ValueError("Cannot infer fs: time array has no valid increasing diffs.")
    return float(1.0 / np.median(dt))


# ---------- Filters ----------
def bandpass_filter(
    x: np.ndarray,
    fs: float,
    low_hz: float = 20.0,
    high_hz: float = 45.0,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass along axis=0 (time)."""
    nyq = 0.5 * fs
    low = low_hz / nyq
    high = high_hz / nyq

    # keep safely below Nyquist to avoid numerical issues
    if high >= 0.999:
        high = 0.999

    if not (0 < low < 1) or not (0 < high < 1) or low >= high:
        raise ValueError(
            f"Invalid bandpass for fs={fs:.3f} Hz (Nyq={nyq:.3f}): low={low_hz}, high={high_hz}"
        )

    b, a = butter(order, [low, high], btype="bandpass")
    return filtfilt(b, a, x, axis=0)


def notch_filter(x: np.ndarray, fs: float, notch_hz: float = 50.0, q: float = 30.0) -> np.ndarray:
    """Zero-phase IIR notch at notch_hz along axis=0 (time)."""
    nyq = 0.5 * fs
    w0 = notch_hz / nyq
    if not (0 < w0 < 1):
        raise ValueError(f"Invalid notch {notch_hz} Hz for fs={fs:.3f} (Nyq={nyq:.3f})")
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
    bandpass_high: float = 45.0,
    do_notch: bool = False,          # DB1 fs=100 -> leave False
    notch_hz: float = 50.0,
    notch_q: float = 30.0,
    window_ms: float = 200.0,
    step_ms: float = 50.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Process one trial and return:
      x_proc:  (n_samples, n_channels) processed continuous signal
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

    # 2) bandpass (DB1-friendly)
    x = bandpass_filter(x, fs=fs, low_hz=bandpass_low, high_hz=bandpass_high, order=4)

    # 3) notch (only if feasible)
    if do_notch:
        # Only safe if notch is below Nyquist
        if notch_hz >= 0.5 * fs:
            raise ValueError(f"Notch {notch_hz} Hz is >= Nyquist ({0.5*fs:.2f} Hz). Disable notch or use higher fs.")
        x = notch_filter(x, fs=fs, notch_hz=notch_hz, q=notch_q)

    # 4) z-score normalization
    x = zscore_per_channel(x)

    # 5) windowing
    windows = sliding_window(x, fs=fs, window_ms=window_ms, step_ms=step_ms)

    return x, windows, float(fs)


if __name__ == "__main__":
    csv_path = "S1_A1_E1_export.csv"
    df = pd.read_csv(csv_path)

    # NinaPro export typically uses "Time" and "EMG_1"... "EMG_10"
    time_col = "Time" if "Time" in df.columns else None
    emg_cols = [c for c in df.columns if c.startswith("EMG_")]

    if time_col is None:
        raise ValueError(f"Time column not found. Columns are: {df.columns.tolist()[:30]} ...")

    if not emg_cols:
        raise ValueError(f"No EMG columns found (expected EMG_1..EMG_10). Columns are: {df.columns.tolist()[:30]} ...")

    # Process
    x_proc, windows, fs_used = process_emg_trial(
        df,
        emg_cols,
        time_col=time_col,
        fs=None,                 # infer from Time
        bandpass_low=20.0,
        bandpass_high=45.0,      # must be < 50 for fs=100
        do_notch=False,          # can't do 50 Hz notch at fs=100
        window_ms=200.0,
        step_ms=50.0,
    )

    print("fs =", fs_used)
    print("processed shape =", x_proc.shape)     # (n_samples, n_channels)
    print("windows shape =", windows.shape)      # (n_windows, win_samples, n_channels)

    # ---- Plot processed channel 1 ----
    t = df[time_col].to_numpy(dtype=float)

    plt.figure(figsize=(10, 4))
    plt.plot(t, x_proc[:, 0])
    plt.xlabel("Time (s)")
    plt.ylabel("Processed EMG_1 (z)")
    plt.title("Processed EMG Channel 1 (bandpass + mean removal + z-score)")
    plt.tight_layout()

    # If running in a headless terminal (Codespaces), save instead of show:
    try:
        plt.show()
    except Exception:
        out = "processed_emg_ch1.png"
        plt.savefig(out, dpi=200)
        print(f"Plot saved to {out}")