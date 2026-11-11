"""
Owl TinyML pipeline - Waveform augmentation (adapted from Mathieu Bornet's 4.NoiseAugmenting.py)

Reads the TRAIN windows listed in split.csv and creates augmented copies of them,
to make the model robust to forest noise and to small natural variations of the songs.
Validation and test windows are never augmented.

Class-aware augmentation:
  - owl classes (pygmy, boreal): noise mixing, Gaussian noise, small pitch shift, small time stretch
  - negative class: only noise mixing and Gaussian noise (pitch/stretch would create unnatural sounds)

Every augmented clip keeps a link to its source recording (XC id) in a CSV manifest,
so that the grouped train/validation split stays leakage-free.
All parameters come from config.py.
"""

import csv
import random
import warnings
from pathlib import Path
import argparse

import librosa
import numpy as np
import soundfile as sf

import config

warnings.filterwarnings('ignore', category=UserWarning)


random.seed(config.SEED)
np.random.seed(config.SEED)

# ==========================================
# 1. CORE AUGMENTATION FUNCTIONS
# ==========================================

def calculate_rms(y):
    if np.all(y == 0): return 1e-10
    return np.sqrt(np.mean(y**2))

def add_environmental_noise(signal, noise, snr_db):
    rms_signal = calculate_rms(signal)
    rms_noise = calculate_rms(noise)
    rms_noise_target = rms_signal / (10 ** (snr_db / 20.0))
    noise_scaled = noise * (rms_noise_target / rms_noise)
    mixed = signal + noise_scaled
    max_val = np.max(np.abs(mixed))
    if max_val > 1.0: mixed = mixed / max_val
    return mixed

def add_gaussian_noise(signal, snr_db):
    noise = np.random.randn(len(signal))
    return add_environmental_noise(signal, noise, snr_db)

def apply_pitch_shift(signal, sr, n_steps):
    return librosa.effects.pitch_shift(y=signal, sr=sr, n_steps=n_steps)

def apply_time_stretch(signal, rate, target_length):
    stretched = librosa.effects.time_stretch(y=signal, rate=rate)
    if len(stretched) < target_length:
        stretched = np.pad(stretched, (0, target_length - len(stretched)))
    else:
        stretched = stretched[:target_length]
    return stretched


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



# ==========================================
# NOISE BANK
# ==========================================

def build_noise_bank(train_rows):
    """List the audio files used for noise mixing.

    Only TRAIN rows are used, so no test audio can leak into training.
    """
    return sorted({row["recording"] for row in train_rows
                   if row["subclass"] in config.NOISE_BANK_SUBCLASSES})
    

def get_noise_slice(noise_files, length):
    """Return `length` samples of noise taken at random from the noise bank.

    Short clips are joined end to end until there is enough audio,
    then a random part is cut out.
    """
    pieces, total = [], 0
    while total < length:
        path = random.choice(noise_files)
        noise, _ = librosa.load(config.ROOT / path, sr=config.SR, mono=True)
        pieces.append(noise)
        total += len(noise)
    noise = np.concatenate(pieces)
    start = random.randint(0, len(noise) - length)
    return noise[start:start + length]



# ==========================================
# ONE AUGMENTED COPY
# ==========================================

def choose_aug_types(cls, noise_files):
    """List the augmentations allowed for a class."""
    if cls == config.NOISE_CLASS:
        types = ["gauss"]
    else:
        types = ["gauss", "pitch", "stretch"]
    if noise_files:
        types.append("snr")
    return types


def augment_window(signal, cls, noise_files):
    """Create one augmented copy of a window.

    Returns (augmented_signal, augmentation_name, parameter_value).
    """
    aug = random.choice(choose_aug_types(cls, noise_files))
    if aug == "snr":
        param = random.uniform(*config.SNR_RANGE_DB)
        noise = get_noise_slice(noise_files, len(signal))
        out = add_environmental_noise(signal, noise, param)
    elif aug == "gauss":
        param = random.uniform(*config.GAUSS_SNR_RANGE_DB)
        out = add_gaussian_noise(signal, param)
    elif aug == "pitch":
        param = random.uniform(-config.PITCH_SEMITONES, config.PITCH_SEMITONES)
        out = apply_pitch_shift(signal, config.SR, param)
    else:  # "stretch"
        param = random.uniform(*config.STRETCH_RANGE)
        out = apply_time_stretch(signal, param, len(signal))
    return out.astype(np.float32), aug, param


# ==========================================
# MAIN LOOP
# ==========================================

def iter_all_windows(rows, noise_files, max_recordings=None, max_failures=20):
    """Yield every sound to turn into a spectrogram.

    For each window of `rows`: first the original (any split), then its
    augmented copies (train only, N_AUG_PER_WINDOW per class).
    Yields (row, kind, signal, aug, param), kind = "original", "aug0", "aug1", ...
    Each source recording is loaded only once.
    """
    groups = group_by_recording(rows)
    failures = 0
    for i, (recording, rec_rows) in enumerate(groups.items(), 1):
        if max_recordings is not None and i > max_recordings:
            break
        try:
            windows = load_windows(recording, rec_rows)
        except Exception as e:
            failures += 1
            print(f"  [skip] {recording}: {e}")
            if failures > max_failures:
                raise RuntimeError(f"Too many unreadable files ({failures}), stopping.")
            continue
        for row, signal in windows:
            yield row, "original", signal, "", ""
            if row["split"] == "train":
                for k in range(config.N_AUG_PER_WINDOW.get(row["class"], 0)):
                    out, aug, param = augment_window(signal, row["class"], noise_files)
                    yield row, f"aug{k}", out, aug, round(param, 3)
        if i % 200 == 0:
            print(f"  {i}/{len(groups)} recordings done")


## ==========================================
# COMMAND LINE
# ==========================================

def save_examples(train_rows, noise_files, n, out_dir):
    """Write n augmented windows, each with its original, as .wav files to listen to."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.wav"):
        old.unlink()
    rows = [r for r in train_rows if config.N_AUG_PER_WINDOW.get(r["class"], 0) > 0]
    sample = random.sample(rows, min(n, len(rows)))
    for recording, rec_rows in group_by_recording(sample).items():
        for row, signal in load_windows(recording, rec_rows):
            out, aug, param = augment_window(signal, row["class"], noise_files)
            name = f'{row["class"]}_{Path(recording).stem}_{float(row["start_s"]):.2f}s'
            sf.write(out_dir / f"{name}_original.wav", signal, config.SR)
            sf.write(out_dir / f"{name}_{aug}_{param:+.2f}.wav", out, config.SR)


def main():
    parser = argparse.ArgumentParser(description="Augment the TRAIN windows of split.csv.")
    parser.add_argument("--save-examples", type=int, default=0, metavar="N",
                        help="write N augmented examples (+ originals) as .wav to listen to")
    parser.add_argument("--examples-dir", type=Path, default=config.DATA / "augment_examples",
                        help="folder for the examples (emptied first)")
    args = parser.parse_args()

    rows = read_train_rows()
    noise_files = build_noise_bank(rows)
    print(f"{len(rows)} train windows, {len(noise_files)} noise files")
    for cls in config.CLASSES:
        n = sum(r["class"] == cls for r in rows)
        copies = n * config.N_AUG_PER_WINDOW.get(cls, 0)
        print(f"  {cls}: {n} windows -> {copies} augmented copies")

    if args.save_examples:
        save_examples(rows, noise_files, args.save_examples, args.examples_dir)
        print(f"Examples written to {args.examples_dir}")


if __name__ == "__main__":
    main()