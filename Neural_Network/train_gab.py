from __future__ import annotations

print("✅ RF PIPELINE - CROSS-SUBJECT (10 Participants)")

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = BASE_DIR / "Nina_DB1_CSV"

EMG_COLUMNS = [f"EMG_{i}" for i in range(1, 11)]
TIME_COLUMN  = "Time"
LABEL_COLUMN = "restimulus"

# Sampling frequency is 100 Hz (confirmed from CSV timestamps: 0, 0.01, 0.02...)
# 200ms window = 20 samples | 50ms overlap → step = 150ms = 15 samples
WINDOW_SAMPLES = 20
STEP_SAMPLES   = 15

# Bandpass valid for fs=100 Hz (Nyquist = 50 Hz, so highcut must be < 50)
LOWCUT       = 20.0
HIGHCUT      = 45.0
FILTER_ORDER = 4

RANDOM_STATE         = 42
DOMINANT_LABEL_RATIO = 0.80  # minimum ratio for a label to be considered dominant in a window

# Cross-subject split: train on 7 participants, test on 3 completely unseen participants
TRAIN_PARTICIPANTS = [
    "Participant1", "Participant2", "Participant3", "Participant4",
    "Participant5", "Participant6", "Participant7"
]
TEST_PARTICIPANTS = [
    "Participant8", "Participant9", "Participant10"
]


# =========================
# PREPROCESSING
# =========================
def remove_dc(emg: np.ndarray) -> np.ndarray:
    """Remove DC offset by subtracting the mean of each channel."""
    return emg - np.mean(emg, axis=0, keepdims=True)


def bandpass_filter(emg: np.ndarray, fs: float = 100.0) -> np.ndarray:
    """Apply a Butterworth bandpass filter (20-45 Hz) to the EMG signal."""
    nyquist = fs / 2.0
    low  = LOWCUT  / nyquist
    high = HIGHCUT / nyquist
    b, a = butter(FILTER_ORDER, [low, high], btype="bandpass")
    return filtfilt(b, a, emg, axis=0)


def zscore_normalize(emg: np.ndarray) -> np.ndarray:
    """Normalize each channel to zero mean and unit variance."""
    mean = np.mean(emg, axis=0, keepdims=True)
    std  = np.std(emg, axis=0, keepdims=True)
    std  = np.where(std < 1e-8, 1.0, std)  # avoid division by zero
    return (emg - mean) / std


def preprocess_emg(emg: np.ndarray) -> np.ndarray:
    """Full preprocessing pipeline: DC removal -> bandpass filter -> z-score normalization."""
    emg = remove_dc(emg)
    emg = bandpass_filter(emg)
    emg = zscore_normalize(emg)
    return emg


# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(window: np.ndarray) -> np.ndarray:
    """
    Extract 14 time-domain features per channel -> 140 features total (10 channels).

    Features per channel:
        MAV, RMS, Variance, Waveform Length, Zero Crossings, Slope Sign Changes,
        IAV, AAC, DASDV, Willison Amplitude, Mean, Std, Max, Min
    """
    features: List[float] = []

    for ch in range(window.shape[1]):
        sig = window[:, ch]

        mav   = float(np.mean(np.abs(sig)))
        rms   = float(np.sqrt(np.mean(sig ** 2)))
        var   = float(np.var(sig))
        wl    = float(np.sum(np.abs(np.diff(sig))))           # Waveform Length
        mean  = float(np.mean(sig))
        std   = float(np.std(sig))
        mx    = float(np.max(sig))
        mn    = float(np.min(sig))
        iav   = float(np.sum(np.abs(sig)))                    # Integrated Absolute Value
        aac   = float(np.mean(np.abs(np.diff(sig)))) if len(sig) > 1 else 0.0  # Average Amplitude Change
        dasdv = float(np.sqrt(np.mean(np.diff(sig) ** 2)))   if len(sig) > 1 else 0.0  # DASDV

        # Zero Crossings
        x1, x2 = sig[:-1], sig[1:]
        zc = float(np.sum(((x1 * x2) < 0) & (np.abs(x1 - x2) >= 1e-3)))

        # Slope Sign Changes
        xp, xm, xn = sig[:-2], sig[1:-1], sig[2:]
        ssc = float(np.sum(
            ((xm - xp) * (xm - xn) > 0) &
            ((np.abs(xm - xp) >= 1e-3) | (np.abs(xm - xn) >= 1e-3))
        ))

        # Willison Amplitude
        wamp = float(np.sum(np.abs(np.diff(sig)) >= 1e-3)) if len(sig) > 1 else 0.0

        features.extend([mav, rms, var, wl, zc, ssc, iav, aac, dasdv, wamp, mean, std, mx, mn])

    return np.asarray(features, dtype=np.float32)


# =========================
# DATA LOADING
# =========================
def load_csv(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a single CSV file and return EMG array and label array."""
    df     = pd.read_csv(file_path)
    emg    = df[EMG_COLUMNS].to_numpy(dtype=np.float32)
    labels = df[LABEL_COLUMN].to_numpy(dtype=np.int64)
    return emg, labels


def build_windows(emg: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slide a fixed window over the EMG signal and extract features per window.
    Windows where rest (label=0) or no single gesture is dominant are discarded.
    """
    n_features = emg.shape[1] * 14
    X, y = [], []

    if len(emg) < WINDOW_SAMPLES:
        return np.empty((0, n_features), dtype=np.float32), np.empty((0,), dtype=np.int64)

    for start in range(0, len(emg) - WINDOW_SAMPLES + 1, STEP_SAMPLES):
        end           = start + WINDOW_SAMPLES
        window        = emg[start:end]
        window_labels = labels[start:end]

        unique, counts = np.unique(window_labels, return_counts=True)
        dominant_idx   = int(np.argmax(counts))
        label          = int(unique[dominant_idx])
        dominant_ratio = counts[dominant_idx] / len(window_labels)

        # Skip rest windows and ambiguous transition windows
        if label == 0:
            continue
        if dominant_ratio < DOMINANT_LABEL_RATIO:
            continue

        X.append(extract_features(window))
        y.append(label)

    if len(X) == 0:
        return np.empty((0, n_features), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def load_participant(participant_dir: Path, repetition_tags: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Load and concatenate multiple exercise files (e.g. E1+E2) for one participant."""
    X_all, y_all = [], []

    for tag in repetition_tags:
        matches = sorted(participant_dir.glob(f"*_{tag}_export.csv"))
        if len(matches) != 1:
            print(f"  WARNING: File *_{tag}_export.csv not found in {participant_dir.name}, skipping.")
            continue

        file_path = matches[0]
        print(f"  Loading: {file_path.name}")

        emg, labels = load_csv(file_path)
        emg = preprocess_emg(emg)
        X, y = build_windows(emg, labels)

        if len(X) == 0:
            print(f"  WARNING: No valid windows found in {file_path.name}")
            continue

        X_all.append(X)
        y_all.append(y)

    if len(X_all) == 0:
        return np.empty((0, 140), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return np.concatenate(X_all), np.concatenate(y_all)


def build_dataset(
    participant_names: List[str],
    repetition_tags: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a full dataset from a list of participants and exercise tags."""
    X_all, y_all, ids = [], [], []

    for name in participant_names:
        participant_dir = DATA_ROOT / name
        if not participant_dir.exists():
            print(f"WARNING: Folder not found: {participant_dir}, skipping.")
            continue

        print(f"\n📂 {name}")
        X, y = load_participant(participant_dir, repetition_tags)

        if len(X) == 0:
            continue

        X_all.append(X)
        y_all.append(y)
        ids.extend([name] * len(y))

    if len(X_all) == 0:
        raise ValueError("No valid data found. Check your DATA_ROOT path and folder names.")

    return np.concatenate(X_all), np.concatenate(y_all), np.asarray(ids)


# =========================
# EVALUATION
# =========================
def evaluate_per_participant(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    participant_ids: np.ndarray,
) -> pd.DataFrame:
    """Compute accuracy and macro F1 for each test participant individually."""
    rows = []
    for participant in np.unique(participant_ids):
        mask     = participant_ids == participant
        acc      = accuracy_score(y_true[mask], y_pred[mask])
        macro_f1 = f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)
        rows.append({
            "participant": participant,
            "n_samples":   int(np.sum(mask)),
            "accuracy":    round(acc, 4),
            "macro_f1":    round(macro_f1, 4),
        })
    return pd.DataFrame(rows).sort_values("participant").reset_index(drop=True)


# =========================
# MAIN
# =========================
def main() -> None:

    # --- Train: E1 + E2 + E3 from participants 1-7 -----------------------
    print("\n" + "="*55)
    print("LOADING TRAIN SET (Participants 1-7, E1+E2+E3)")
    print("="*55)
    X_train, y_train, _ = build_dataset(TRAIN_PARTICIPANTS, ["E1", "E2", "E3"])

    # --- Test: all exercises from participants 8-10 (completely unseen) --
    print("\n" + "="*55)
    print("LOADING TEST SET (Participants 8-10, E1+E2+E3)")
    print("="*55)
    X_test, y_test, test_ids = build_dataset(TEST_PARTICIPANTS, ["E1", "E2", "E3"])

    # --- Dataset summary -------------------------------------------------
    print("\n" + "="*55)
    print("DATASET SUMMARY")
    print("="*55)
    print(f"Train : {X_train.shape[0]} windows | {len(np.unique(y_train))} gesture classes")
    print(f"Test  : {X_test.shape[0]} windows | {len(np.unique(y_test))} gesture classes")
    print(f"Gesture labels in train : {sorted(np.unique(y_train).tolist())}")
    print(f"Gesture labels in test  : {sorted(np.unique(y_test).tolist())}")

    # --- Train model -----------------------------------------------------
    print("\n" + "="*55)
    print("TRAINING RANDOM FOREST")
    print("="*55)

    # Hyperparameters chosen based on EMG cross-subject literature and
    # partial grid search results: deep trees with log2 features performed best
    model = RandomForestClassifier(
        n_estimators=500,       # more trees = more stable predictions
        max_depth=None,         # let trees grow fully to capture complex patterns
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="log2",    # log2 slightly better than sqrt for high-dim EMG features
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    print("Training complete.")

    # --- Overall results -------------------------------------------------
    y_pred           = model.predict(X_test)
    overall_acc      = accuracy_score(y_test, y_pred)
    overall_macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print("\n" + "="*55)
    print("OVERALL RESULTS")
    print("="*55)
    print(f"Accuracy : {overall_acc:.4f}  ({overall_acc*100:.1f}%)")
    print(f"Macro F1 : {overall_macro_f1:.4f}")

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    # --- Per-participant results ------------------------------------------
    per_subject_df = evaluate_per_participant(y_test, y_pred, test_ids)

    print("\n" + "="*55)
    print("PER-PARTICIPANT RESULTS (test set)")
    print("="*55)
    print(per_subject_df.to_string(index=False))
    print(f"\nMean Accuracy : {per_subject_df['accuracy'].mean():.4f}")
    print(f"Mean Macro F1 : {per_subject_df['macro_f1'].mean():.4f}")


if __name__ == "__main__":
    main()
