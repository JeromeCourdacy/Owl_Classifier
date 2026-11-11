"""
Owl TinyML pipeline - Dataset access

Reads split.csv and turns its rows into sounds:
  - read_split_rows / read_train_rows : the rows of split.csv
  - group_by_recording                : rows grouped by audio file
  - load_windows                      : load one audio file once and cut all its windows

No augmentation here (see augment.py). All parameters come from config.py.
"""

import csv

import librosa
import numpy as np

import config


# ==========================================
# READING split.csv
# ==========================================

def read_split_rows(csv_path=config.SPLIT_CSV):
    """Return ALL rows of split.csv (train, validation, test), each with a unique row_id."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            row["row_id"] = i
            rows.append(row)
    return rows


def read_train_rows(csv_path=config.SPLIT_CSV):
    """Return only the TRAIN rows of split.csv."""
    return [r for r in read_split_rows(csv_path) if r["split"] == "train"]


# ==========================================
# LOADING AUDIO
# ==========================================

def group_by_recording(rows):
    """Group the rows by audio file, so that each file is loaded only once."""
    groups = {}
    for row in rows:
        groups.setdefault(row["recording"], []).append(row)
    return groups


def load_windows(recording, rows):
    """Load one recording once and cut all of its windows.

    Each window starts at start_s and is exactly WINDOW_SAMPLES long.
    Returns a list of (row, signal) pairs.
    """
    audio, _ = librosa.load(config.ROOT / recording, sr=config.SR, mono=True)
    if len(audio) < config.WINDOW_SAMPLES:
        audio = np.pad(audio, (0, config.WINDOW_SAMPLES - len(audio)))
    windows = []
    for row in rows:
        start = int(round(float(row["start_s"]) * config.SR))
        start = min(start, len(audio) - config.WINDOW_SAMPLES)
        windows.append((row, audio[start:start + config.WINDOW_SAMPLES]))
    return windows
