"""
Owl TinyML pipeline - Spectrograms (adapted from Mathieu Bornet's 5.MelSpectrogramPreProcess.py)

One step of the pipeline: sound -> Mel spectrogram image -> .npy file + manifest line.
  - audio_to_melspectrogram : one window of sound -> 40 x 63 image (must match the firmware)
  - augment_spectrogram     : SpecAugment copy of an image (off while SPECAUG_COPIES = 0)
  - start_output            : create the output folder and open manifest.csv
  - process_sound           : sound -> image(s) -> .npy + manifest lines

The order of the steps (dataset -> augment -> spectrogram) is handled by pipeline.py.
All parameters come from config.py.
"""

import csv
import random
from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import lfilter

import config

random.seed(config.SEED)
np.random.seed(config.SEED)


# ==========================================
# SPECAUGMENT (image augmentation)
# ==========================================

AUG_CFG = {
    "time_max":   8,    # max width of a time mask (frames), out of 63 (~0.25 s)
    "p_time":     0.8,
    "freq_max":   3,    # max width of a freq mask (Mel bins), out of 40
    "p_freq":     0.8,
    "noise_db":   1.5,  # std of Gaussian noise in the dB domain
    "p_noise":    0.5,
    "blur_sigma": 0.7,  # ~3x3 kernel
    "p_blur":     0.3,
}


def spec_time_mask(spec, max_width, fill):
    spec = spec.copy()
    T = spec.shape[1]
    w = np.random.randint(1, min(max_width, T) + 1)
    t0 = np.random.randint(0, T - w + 1)
    spec[:, t0:t0 + w] = fill
    return spec


def spec_freq_mask(spec, max_width, fill):
    spec = spec.copy()
    F = spec.shape[0]
    w = np.random.randint(1, min(max_width, F) + 1)
    f0 = np.random.randint(0, F - w + 1)
    spec[f0:f0 + w, :] = fill
    return spec


def augment_spectrogram(spec, cfg=AUG_CFG):
    """SpecAugment on a single dB-scale spectrogram [n_mels, n_frames]."""
    out = spec.copy().astype(np.float32)
    fill = float(out.mean())
    if np.random.rand() < cfg["p_time"]:
        out = spec_time_mask(out, cfg["time_max"], fill)
    if np.random.rand() < cfg["p_freq"]:
        out = spec_freq_mask(out, cfg["freq_max"], fill)
    if np.random.rand() < cfg["p_noise"]:
        out = out + np.random.randn(*out.shape).astype(np.float32) * cfg["noise_db"]
    if np.random.rand() < cfg["p_blur"]:
        out = gaussian_filter(out, sigma=cfg["blur_sigma"])
    out = np.maximum(out, config.NOISE_FLOOR_DB)
    return out.astype(np.float32)


# ==========================================
# SOUND -> SPECTROGRAM
# ==========================================

def audio_to_melspectrogram(signal):
    """Convert one window (WINDOW_SAMPLES samples at SR) into a floored-dB
    Mel spectrogram of shape (N_MELS, n_frames). Must match the firmware."""
    y = lfilter(config.DC_FILTER_B, config.DC_FILTER_A, signal).astype(np.float32)
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=config.SR, n_fft=config.N_FFT, hop_length=config.HOP,
        win_length=config.N_FFT, center=False,
        n_mels=config.N_MELS, fmin=config.FMIN, fmax=config.FMAX,
    )
    if config.NORM_MODE == "peak":
        db_ref = max(float(mel_spec.max()), config.REF_FLOOR)
    else:
        db_ref = 1.0
    mel_spec_db = librosa.power_to_db(mel_spec, ref=db_ref)
    return np.maximum(mel_spec_db, config.NOISE_FLOOR_DB).astype(np.float32)


# ==========================================
# WRITING .npy + manifest.csv
# ==========================================

MANIFEST_FIELDS = ["file", "split", "class", "subclass", "recording", "start_s",
                   "row_id", "kind", "aug", "param"]


def npy_path(out_root, row, kind):
    """Where a spectrogram is saved: <out_root>/<split>/<class>/<row_id>_<recording>_<start>s_<kind>.npy"""
    rec_id = Path(row["recording"]).stem
    name = f'{row["row_id"]:06d}_{rec_id}_{float(row["start_s"]):.3f}s_{kind}.npy'
    return Path(out_root) / row["split"] / row["class"] / name


def start_output(out_root):
    """Create the output folder (must be new or empty) and open manifest.csv.

    Returns (manifest_file, manifest_writer). Close manifest_file at the end.
    """
    out_root = Path(out_root)
    if out_root.exists() and any(out_root.iterdir()):
        raise SystemExit(f"{out_root} is not empty: choose another version or delete it first.")
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_file = open(out_root / "manifest.csv", "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_FIELDS)
    writer.writeheader()
    return manifest_file, writer


def save_spectrogram(out_root, writer, row, kind, image, aug="", param=""):
    """Save one image as .npy and add its line to manifest.csv."""
    path = npy_path(out_root, row, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, image)
    writer.writerow({
        "file": path.relative_to(out_root).as_posix(),
        "split": row["split"], "class": row["class"], "subclass": row["subclass"],
        "recording": row["recording"], "start_s": row["start_s"],
        "row_id": row["row_id"], "kind": kind, "aug": aug, "param": param,
    })


def process_sound(out_root, writer, row, kind, signal, aug="", param=""):
    """Sound -> spectrogram -> .npy (+ SpecAugment copies for train originals).

    Returns the number of .npy files written.
    """
    spec = audio_to_melspectrogram(signal)
    save_spectrogram(out_root, writer, row, kind, spec, aug, param)
    written = 1
    if kind == "original" and row["split"] == "train":
        for k in range(config.SPECAUG_COPIES):
            save_spectrogram(out_root, writer, row, f"specaug{k}",
                             augment_spectrogram(spec), "specaug", "")
            written += 1
    return written
