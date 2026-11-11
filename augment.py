"""
Owl TinyML pipeline - Waveform augmentation (adapted from Mathieu Bornet's 4.NoiseAugmenting.py)

Creates augmented copies of the TRAIN windows listed in split.csv,
to make the model robust to forest noise and to small natural variations of the songs.
Validation and test windows are never augmented.

Class-aware augmentation:
  - owl classes (pygmy, boreal): noise mixing, Gaussian noise, small pitch shift, small time stretch
  - negative class: only noise mixing and Gaussian noise (pitch/stretch would create unnatural sounds)

Reading split.csv and loading audio is done in dataset.py;
spectrogram.py puts dataset.py and augment.py together.
All parameters come from config.py.
"""

import argparse
import random
import warnings
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

import config
import dataset

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
    for recording, rec_rows in dataset.group_by_recording(sample).items():
        for row, signal in dataset.load_windows(recording, rec_rows):
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

    rows = dataset.read_train_rows()
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
