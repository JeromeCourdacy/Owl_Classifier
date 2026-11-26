"""
Owl TinyML pipeline - Dataset access

Reads split.csv and turns its rows into sounds:
  - read_split_rows / read_train_rows : the rows of split.csv
  - group_by_recording                : rows grouped by audio file
  - load_windows                      : load one audio file once and cut all its windows

No augmentation here (see augment.py). All parameters come from config.py.
"""

import csv
from decimal import Decimal
from pathlib import Path

import librosa
import numpy as np

import config


# ==========================================
# READING split.csv
# ==========================================

def clip_path(recording, start_s, end_s):
    """Name of the 2 s clip cut by Jerome from a full recording.

    The start and end times are written in nanoseconds after the XC ID, e.g.
    data/pygmy/XC1000474.wav, 0.0013125 -> 2.0013125 s
    -> data/pygmy/XC1000474_000001312500_002001312500.wav
    """
    ns = lambda s: int((Decimal(s) * 10**9).to_integral_value())
    rec = Path(recording)
    return (rec.parent / f"{rec.stem}_{ns(start_s):012d}_{ns(end_s):012d}.wav").as_posix()


def use_clip_if_needed(row):
    """Point a row to its pre-cut 2 s clip when the full recording is not in AUDIO_ROOT.

    The clip then starts at 0 s. The original recording and start time are kept
    in source_recording / source_start_s.
    """
    row["source_recording"], row["source_start_s"] = row["recording"], row["start_s"]
    if (config.AUDIO_ROOT / row["recording"]).exists():
        return row
    clip = clip_path(row["recording"], row["start_s"], row["end_s"])
    if (config.AUDIO_ROOT / clip).exists():
        row["recording"], row["start_s"] = clip, "0"
    return row


def read_split_rows(csv_path=config.SPLIT_CSV):
    """Return ALL rows of split.csv (train, validation, test), each with a unique row_id.

    Rows that point to a full recording we do not have are redirected to its 2 s clip.
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            row["row_id"] = i
            rows.append(use_clip_if_needed(row))
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
    audio, _ = librosa.load(config.AUDIO_ROOT / recording, sr=config.SR, mono=True)
    if len(audio) < config.WINDOW_SAMPLES:
        audio = np.pad(audio, (0, config.WINDOW_SAMPLES - len(audio)))
    windows = []
    for row in rows:
        start = int(round(float(row["start_s"]) * config.SR))
        start = min(start, len(audio) - config.WINDOW_SAMPLES)
        windows.append((row, audio[start:start + config.WINDOW_SAMPLES]))
    return windows
