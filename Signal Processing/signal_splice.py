from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class Segment:
    seg_id: int
    label: int
    repetition: int
    start_idx: int
    end_idx: int
    start_time: float
    end_time: float
    n_samples: int
    duration_s: float


def infer_fs_from_time(time_s: np.ndarray) -> float:
    """Infer sampling frequency from Time column in seconds."""
    dt = np.diff(time_s)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        raise ValueError("Cannot infer fs: Time has no valid increasing diffs.")
    med_dt = float(np.median(dt))
    return 1.0 / med_dt


def splice_by_restimulus(
    df: pd.DataFrame,
    *,
    time_col: str = "Time",
    label_col: str = "restimulus",
    rep_col: str = "rerepetition",
    emg_prefix: str = "EMG_",
    ignore_label: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, float, list[str]]:
    """
    Splice EMG into contiguous segments where:
      - label_col != ignore_label (typically 0 = rest)
      - label_col stays constant
      - rep_col stays constant

    Returns:
      segments_df: one row per segment, with emg arrays stored in columns (object dtype)
      meta_df: lightweight metadata table (no arrays)
      fs: inferred sampling frequency (Hz)
      emg_cols: list of EMG column names used
    """
    required = {time_col, label_col, rep_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    emg_cols = [c for c in df.columns if c.startswith(emg_prefix)]
    if not emg_cols:
        raise ValueError(f"No EMG columns found with prefix '{emg_prefix}'")

    # Ensure time is numeric and sorted (keep original row order otherwise)
    time_s = pd.to_numeric(df[time_col], errors="coerce").to_numpy()
    if np.any(~np.isfinite(time_s)):
        raise ValueError("Time column contains non-numeric or NaN values.")
    if not np.all(np.diff(time_s) >= 0):
        # If time is not monotonic, sort by time (common fix)
        df = df.sort_values(time_col).reset_index(drop=True)
        time_s = df[time_col].to_numpy()

    fs = infer_fs_from_time(time_s)

    labels = df[label_col].to_numpy()
    reps = df[rep_col].to_numpy()

    segments: List[Segment] = []
    rows = []

    start: Optional[int] = None
    cur_label: Optional[int] = None
    cur_rep: Optional[int] = None
    seg_id = 0

    def close_segment(end_idx: int) -> None:
        nonlocal seg_id, start, cur_label, cur_rep
        if start is None or cur_label is None or cur_rep is None:
            return
        s = start
        e = end_idx
        n = int(e - s + 1)
        st = float(time_s[s])
        et = float(time_s[e])
        dur = float(et - st)

        segments.append(
            Segment(
                seg_id=seg_id,
                label=int(cur_label),
                repetition=int(cur_rep),
                start_idx=int(s),
                end_idx=int(e),
                start_time=st,
                end_time=et,
                n_samples=n,
                duration_s=dur,
            )
        )

        seg_slice = df.iloc[s : e + 1]
        # Store each channel as a numpy array in the output row
        row = {
            "seg_id": seg_id,
            "label": int(cur_label),
            "repetition": int(cur_rep),
            "start_idx": int(s),
            "end_idx": int(e),
            "start_time": st,
            "end_time": et,
            "n_samples": n,
            "duration_s": dur,
        }
        for c in emg_cols:
            row[c] = seg_slice[c].to_numpy(dtype=float)
        rows.append(row)

        seg_id += 1
        start = None
        cur_label = None
        cur_rep = None

    for i, (lab, rep) in enumerate(zip(labels, reps)):
        if lab != ignore_label:
            if start is None:
                start = i
                cur_label = lab
                cur_rep = rep
            else:
                # If label or repetition changes, close previous segment and start new
                if lab != cur_label or rep != cur_rep:
                    close_segment(i - 1)
                    start = i
                    cur_label = lab
                    cur_rep = rep
        else:
            # lab == ignore_label => rest, close any open segment
            if start is not None:
                close_segment(i - 1)

    # Close final segment if file ends during movement
    if start is not None:
        close_segment(len(df) - 1)

    segments_df = pd.DataFrame(rows)

    meta_df = pd.DataFrame([s.__dict__ for s in segments])
    # Helpful sorting (label then repetition then time)
    meta_df = meta_df.sort_values(["label", "repetition", "start_time"]).reset_index(drop=True)

    return segments_df, meta_df, fs, emg_cols


if __name__ == "__main__":
    csv_path = "Nina_DB1_CSV/S1_A1_E1_export.csv"  # <- change if needed
    df = pd.read_csv(csv_path)

    segments_df, meta_df, fs, emg_cols = splice_by_restimulus(df)

    print(f"Sampling rate (inferred): {fs:.3f} Hz")
    print(f"EMG channels found: {len(emg_cols)} -> {emg_cols}")
    print(f"Segments found (restimulus != 0): {len(meta_df)}")
    print(meta_df.head(10))

    # Example: access the 0th segment EMG arrays
    # seg0_emg7 = segments_df.loc[0, "EMG_7"]  # numpy array

    # Optional: save metadata (no raw arrays) to CSV
    # meta_df.to_csv("spliced_segments_metadata.csv", index=False)

    # Optional: save each segment as its own CSV (comment in if you want)
    # for _, row in meta_df.iterrows():
    #     s, e = int(row["start_idx"]), int(row["end_idx"])
    #     out = df.iloc[s:e+1][["Time"] + emg_cols + ["restimulus", "rerepetition"]]
    #     out.to_csv(f"segment_{int(row['seg_id']):04d}_L{int(row['label'])}_R{int(row['repetition'])}.csv", index=False)