from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

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

    # keep safely below Nyquist
    if high >= 0.999:
        high = 0.999

    if not (0 < low < 1) or not (0 < high < 1) or low >= high:
        raise ValueError(
            f"Invalid bandpass for fs={fs:.3f} Hz (Nyq={nyq:.3f}): low={low_hz}, high={high_hz}"
        )

    b, a = butter(order, [low, high], btype="bandpass")
    return filtfilt(b, a, x, axis=0)


def notch_filter(
    x: np.ndarray,
    fs: float,
    notch_hz: float = 50.0,
    q: float = 30.0
) -> np.ndarray:
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


# ---------- Full continuous pipeline (CSV-friendly) ----------
def process_emg_continuous(
    df: pd.DataFrame,
    emg_cols: list[str],
    *,
    fs: float | None = None,
    time_col: str = "Time",
    bandpass_low: float = 20.0,
    bandpass_high: float = 45.0,
    do_notch: bool = False,
    notch_hz: float = 50.0,
    notch_q: float = 30.0,
    zscore: bool = True,
) -> tuple[pd.DataFrame, float]:
    """
    Returns a copy of df with processed EMG columns added (suffix '_proc').

    Output columns added: EMG_1_proc ... EMG_10_proc (based on emg_cols)
    """
    out = df.copy()

    if fs is None:
        if time_col not in out.columns:
            raise ValueError(f"Time column '{time_col}' not found.")
        t = pd.to_numeric(out[time_col], errors="coerce").to_numpy(dtype=float)
        if np.any(~np.isfinite(t)):
            raise ValueError(f"Time column '{time_col}' contains NaNs/non-numeric values.")
        fs = infer_fs_from_time(t)

    x = out[emg_cols].to_numpy(dtype=float)

    # 1) DC offset removal (per trial / full file)
    x = x - np.mean(x, axis=0, keepdims=True)

    # 2) Bandpass
    x = bandpass_filter(x, fs=fs, low_hz=bandpass_low, high_hz=bandpass_high, order=4)

    # 3) Notch (optional; for DB1 fs=100, 50Hz notch is NOT possible because Nyquist=50)
    if do_notch:
        if notch_hz >= 0.5 * fs:
            raise ValueError(
                f"Notch {notch_hz} Hz is >= Nyquist ({0.5*fs:.2f} Hz). "
                f"Disable notch or use a higher sampling rate."
            )
        x = notch_filter(x, fs=fs, notch_hz=notch_hz, q=notch_q)

    # 4) Z-score
    if zscore:
        x = zscore_per_channel(x)

    # Write back to dataframe
    for j, c in enumerate(emg_cols):
        out[f"{c}_proc"] = x[:, j]

    return out, float(fs)


# ---------- Batch: preprocess one CSV and save ----------
def preprocess_csv_to_csv(
    in_csv: str | Path,
    *,
    out_csv: str | Path | None = None,
    time_col: str = "Time",
    emg_prefix: str = "EMG_",
    bandpass_low: float = 20.0,
    bandpass_high: float = 45.0,
    do_notch: bool = False,
    zscore: bool = True,
) -> Path:
    """
    Load a NinaPro-exported CSV, preprocess EMG_1..EMG_10, and save to new CSV.
    Keeps original columns and adds EMG_*_proc columns.
    """
    in_csv = Path(in_csv)
    if not in_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_csv.resolve()}")

    df = pd.read_csv(in_csv)

    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found. Columns: {df.columns.tolist()[:25]} ...")

    emg_cols = [c for c in df.columns if c.startswith(emg_prefix)]
    if not emg_cols:
        raise ValueError(f"No EMG columns found with prefix '{emg_prefix}'. Columns: {df.columns.tolist()[:25]} ...")

    df_proc, fs = process_emg_continuous(
        df,
        emg_cols,
        fs=None,
        time_col=time_col,
        bandpass_low=bandpass_low,
        bandpass_high=bandpass_high,
        do_notch=do_notch,
        zscore=zscore,
    )

    if out_csv is None:
        out_csv = in_csv.with_name(in_csv.stem + "_processed.csv")
    out_csv = Path(out_csv)

    df_proc.to_csv(out_csv, index=False)
    print(f"Saved processed CSV: {out_csv.resolve()}")
    print(f"Inferred fs: {fs:.3f} Hz | Added columns: {[c + '_proc' for c in emg_cols]}")
    return out_csv


# ---------- Convenience for YOUR folder layout ----------
def preprocess_nina_exports_for_subject(
    *,
    subject: str = "S1",
    export_folder: str = "../Nina_DB1_CSV",
    out_folder: str = "../Nina_DB1_CSV_processed",
    patterns: Optional[list[str]] = None,
) -> list[Path]:
    """
    Preprocess multiple exported CSVs (e.g., S1_A1_E1_export.csv, S1_B1_E1_export.csv, S1_C1_E1_export.csv)
    and save them into a processed folder with the same filenames + '_processed'.
    """
    export_dir = Path(export_folder)
    out_dir = Path(out_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    if patterns is None:
        # default: grab all exported CSVs for this subject
        patterns = [f"{subject}_*_export.csv"]

    in_files: list[Path] = []
    for pat in patterns:
        in_files.extend(sorted(export_dir.glob(pat)))

    if not in_files:
        raise FileNotFoundError(
            f"No files matched patterns {patterns} in {export_dir.resolve()}"
        )

    saved: list[Path] = []
    for f in in_files:
        out_path = out_dir / (f.stem + "_processed.csv")
        saved.append(
            preprocess_csv_to_csv(
                f,
                out_csv=out_path,
                do_notch=False,      # DB1 @100Hz -> leave False
                bandpass_low=20.0,
                bandpass_high=45.0,
                zscore=True,
            )
        )

    return saved