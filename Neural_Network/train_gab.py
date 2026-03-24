from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================
# CONFIG
# =========================
TRAIN_FILES = [
    Path("../Nina_DB1_CSV/Participant1/S1_A1_E1_export.csv"),
    Path("../Nina_DB1_CSV/Participant1/S1_A1_E2_export.csv"),
]

TEST_FILES = [
    Path("../Nina_DB1_CSV/Participant1/S1_A1_E3_export.csv"),
]

EMG_COLUMNS = [f"EMG_{i}" for i in range(1, 11)]
TIME_COLUMN = "Time"
LABEL_COLUMN = "restimulus"
REPETITION_COLUMN = "rerepetition"

WINDOW_MS = 200
STEP_MS = 50

RANDOM_STATE = 42


# =========================
# SIGNAL PROCESSING
# =========================
def infer_fs(time_vector: np.ndarray) -> float:
    dt = np.diff(time_vector)
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0]

    return float(1.0 / np.median(dt))


def remove_dc(emg: np.ndarray) -> np.ndarray:
    return emg - np.mean(emg, axis=0, keepdims=True)


def bandpass_filter(
    emg: np.ndarray,
    fs: float,
    lowcut: float = 20.0,
    highcut: float = 45.0,
    order: int = 4,
) -> np.ndarray:
    nyquist = fs / 2.0

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype="bandpass")
    return filtfilt(b, a, emg, axis=0)


def zscore_normalize(emg: np.ndarray) -> np.ndarray:
    mean = np.mean(emg, axis=0, keepdims=True)
    std = np.std(emg, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (emg - mean) / std


def preprocess_emg(time_vector: np.ndarray, emg: np.ndarray) -> Tuple[np.ndarray, float]:
    fs = infer_fs(time_vector)
    emg = remove_dc(emg)
    emg = bandpass_filter(emg, fs=fs)
    emg = zscore_normalize(emg)
    return emg, fs


# =========================
# FEATURES
# =========================
def mav(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x)))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))


def waveform_length(x: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(x))))


def zero_crossings(x: np.ndarray, threshold: float = 1e-3) -> float:
    x1 = x[:-1]
    x2 = x[1:]
    crossings = ((x1 * x2) < 0) & (np.abs(x1 - x2) >= threshold)
    return float(np.sum(crossings))


def slope_sign_changes(x: np.ndarray, threshold: float = 1e-3) -> float:
    x_prev = x[:-2]
    x_mid = x[1:-1]
    x_next = x[2:]

    ssc = (((x_mid - x_prev) * (x_mid - x_next)) > 0) & (
        (np.abs(x_mid - x_prev) >= threshold)
        | (np.abs(x_mid - x_next) >= threshold)
    )
    return float(np.sum(ssc))


def variance(x: np.ndarray) -> float:
    return float(np.var(x))


def extract_features(window: np.ndarray) -> np.ndarray:
    feats = []

    for ch in range(window.shape[1]):
        sig = window[:, ch]
        feats.extend([
            mav(sig),
            rms(sig),
            waveform_length(sig),
            zero_crossings(sig),
            slope_sign_changes(sig),
            variance(sig),
        ])

    return np.asarray(feats, dtype=np.float32)


# =========================
# DATA
# =========================
def load_emg_csv(file_path: Path):
    df = pd.read_csv(file_path)

    time_vector = df[TIME_COLUMN].to_numpy()
    emg = df[EMG_COLUMNS].to_numpy()
    labels = df[LABEL_COLUMN].to_numpy()
    repetitions = df[REPETITION_COLUMN].to_numpy()

    return time_vector, emg, labels, repetitions


def find_segments(mask):
    segments = []
    start = None

    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            segments.append((start, i))
            start = None

    if start is not None:
        segments.append((start, len(mask)))

    return segments


def extract_segments(emg, labels, repetitions):
    mask = labels > 0
    segments_idx = find_segments(mask)

    segments = []

    for start, end in segments_idx:
        seg_emg = emg[start:end]
        seg_labels = labels[start:end]
        seg_reps = repetitions[start:end]

        boundaries = [0]

        for i in range(1, len(seg_labels)):
            if seg_labels[i] != seg_labels[i - 1] or seg_reps[i] != seg_reps[i - 1]:
                boundaries.append(i)

        boundaries.append(len(seg_labels))

        for b0, b1 in zip(boundaries[:-1], boundaries[1:]):
            chunk = seg_emg[b0:b1]

            if len(chunk) == 0:
                continue

            segments.append({
                "emg": chunk,
                "label": int(seg_labels[b0])
            })

    return segments


def window_features(emg, label, fs):
    window_size = int(fs * WINDOW_MS / 1000)
    step_size = int(fs * STEP_MS / 1000)

    X, y = [], []

    if len(emg) < window_size:
        return np.empty((0, emg.shape[1] * 6)), np.empty((0,))

    for start in range(0, len(emg) - window_size + 1, step_size):
        window = emg[start:start + window_size]
        X.append(extract_features(window))
        y.append(label)

    return np.array(X), np.array(y)


def build_dataset(files):
    X_all, y_all = [], []

    for f in files:
        print(f"Processing: {f}")

        t, emg, labels, reps = load_emg_csv(f)
        emg, fs = preprocess_emg(t, emg)

        segments = extract_segments(emg, labels, reps)

        for seg in segments:
            X, y = window_features(seg["emg"], seg["label"], fs)

            if len(X) > 0:
                X_all.append(X)
                y_all.append(y)

    X = np.concatenate(X_all)
    y = np.concatenate(y_all)

    return X, y


# =========================
# MAIN
# =========================
def main():
    X_train, y_train = build_dataset(TRAIN_FILES)
    X_test, y_test = build_dataset(TEST_FILES)

    print("Train:", X_train.shape)
    print("Test:", X_test.shape)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nReport:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
