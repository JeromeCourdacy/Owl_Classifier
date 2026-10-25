#!/usr/bin/env python3
"""
extract_calls.py — Automatically cut the passages where a bird is calling into
fixed-length clips (2 s by default), for any species.

How it works:
  1. keep only the frequency band of the call (band-pass filter, ffmpeg)
  2. measure the energy in that band every 50 ms
  3. a window is kept if it contains moments that are clearly louder than the
     file's background noise (--threshold dB) AND whose energy is concentrated
     in the band (--min-band-ratio), which rejects loud broadband sounds
  4. keep the most energetic windows, without overlap, and cut them from the
     ORIGINAL (unfiltered) audio

EXAMPLES
  # Boreal Owl: song fundamental ~650-950 Hz (European birds higher, N. American lower)
  python extract_calls.py data/boreal_train_full data/data_train_2s/boreal_train --band 550 1100 --max 3

  # Same for the test set
  python extract_calls.py data/boreal_test_full data/data_test_2s/boreal_test --band 550 1100 --max 3

  # Eurasian Pygmy Owl: whistles ~1.3-2.0 kHz (measured on our recordings)
  python extract_calls.py data/pygmy_train_full data/data_train_2s/pygmy_train --band 1150 2100 --max 3

  # Tawny Owl: hoots ~650-1000 Hz (same band as Boreal Owl)
  python extract_calls.py data/tawny_train_full data/data_train_2s/tawny_train --band 550 1100 --max 3

Show all options:  python extract_calls.py -h

The frequency band depends on the species: check it on a spectrogram
(Audacity, Edge Impulse, or Mathieu's u.spectral_analysis.py) before running.
Requires: Python 3 and ffmpeg. Nothing else to install.
"""

import argparse
import array
import glob
import math
import os
import shutil
import subprocess
import sys
import wave

SR = 16000
FRAME_S = 0.05            # 50 ms


def read_audio(path, audio_filter=None):
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path]
    if audio_filter:
        cmd += ["-af", audio_filter]
    cmd += ["-ac", "1", "-ar", str(SR), "-f", "s16le", "-"]
    data = subprocess.run(cmd, capture_output=True, check=True).stdout
    a = array.array("h")
    a.frombytes(data)
    return a


def write_wav(path, samples):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(samples.tobytes())


def frame_energies(signal, frame):
    return [sum(x * x for x in signal[i:i + frame]) / frame for i in range(0, len(signal) - frame + 1, frame)]


def extract(path, output, f_low, f_high, threshold_db, duration, max_per_file, min_gap_s,
            min_band_ratio, min_active):
    name = os.path.splitext(os.path.basename(path))[0]
    original = read_audio(path)
    band = read_audio(path, f"highpass=f={f_low},lowpass=f={f_high}")
    broad = read_audio(path, "highpass=f=100")              # whole useful spectrum (no rumble < 100 Hz)

    frame = int(FRAME_S * SR)
    n_clip = int(duration * SR)
    if len(original) < n_clip:
        return name, 0, "too short"

    # energy per frame in the call band, and over the whole spectrum
    e = frame_energies(band, frame)
    e_broad = frame_energies(broad, frame)[:len(e)]
    db = [10 * math.log10(v + 1e-9) for v in e]
    floor = sorted(db)[len(db) // 5]                       # background noise = 20th percentile
    # a "call" frame = clearly louder than the background AND energy concentrated in the band
    # (rejects loud broadband sounds: clicks, branches, high-pitched birds)
    active = [db[i] > floor + threshold_db and e[i] >= min_band_ratio * (e_broad[i] + 1e-9)
              for i in range(len(e))]

    # total energy of each window (sliding sum); keep windows that contain enough call
    L = int(round(duration / FRAME_S))
    if len(e) < L:
        return name, 0, "too short"
    min_active_frames = max(2, int(round(min_active * L)))  # e.g. 10% of 2 s = at least 0.2 s of call
    total, n_active = sum(e[:L]), sum(active[:L])
    windows = []
    for k in range(len(e) - L + 1):
        if k:
            total += e[k + L - 1] - e[k - 1]
            n_active += active[k + L - 1] - active[k - 1]
        if n_active >= min_active_frames:
            windows.append((total, k))
    windows.sort(reverse=True)

    gap = max(int(round(min_gap_s / FRAME_S)), L)          # no overlap
    picked = []
    for _, k in windows:
        if all(abs(k - p) >= gap for p in picked):
            picked.append(k)
            if len(picked) >= max_per_file:
                break

    for n, k in enumerate(sorted(picked), 1):
        start = min(k * frame, len(original) - n_clip)
        write_wav(os.path.join(output, f"{name}_call{n:02d}_{start / SR:.1f}s.wav"), original[start:start + n_clip])
    return name, len(picked), ""


def main():
    ap = argparse.ArgumentParser(
        description="Cut bird calls into fixed-length clips.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("EXAMPLES")[1].split("The frequency band")[0])
    ap.add_argument("input", help="folder (or single file) of full recordings: .wav or .mp3")
    ap.add_argument("output", help="folder where the clips are written")
    ap.add_argument("--band", nargs=2, type=float, metavar=("LOW", "HIGH"), default=(550, 1100),
                    help="frequency band of the call in Hz (default 550 1100 = Boreal Owl song)")
    ap.add_argument("--duration", type=float, default=2.0, help="clip length in s (default 2)")
    ap.add_argument("--max", type=int, default=3, help="max clips per recording (default 3)")
    ap.add_argument("--threshold", type=float, default=10, help="dB above the background noise (default 10)")
    ap.add_argument("--min-gap", type=float, default=0, help="minimum gap between two clips in s (default = duration)")
    ap.add_argument("--min-band-ratio", type=float, default=0.4,
                    help="minimum share of the energy inside the band during a call, 0 to 1 (default 0.4)")
    ap.add_argument("--min-active", type=float, default=0.1,
                    help="minimum share of the window filled by the call, 0 to 1 (default 0.1 = 0.2 s out of 2 s)")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found: activate the conda environment where it is installed.")
    if os.path.isfile(args.input):
        files = [args.input]
    else:
        files = sorted(glob.glob(os.path.join(args.input, "*.wav")) + glob.glob(os.path.join(args.input, "*.mp3")))
    if not files:
        sys.exit(f"No .wav or .mp3 file found in '{args.input}'")
    f_low, f_high = sorted(args.band)
    os.makedirs(args.output, exist_ok=True)

    print(f"{len(files)} files — band {f_low:.0f}–{f_high:.0f} Hz, {args.duration:g} s clips, "
          f"max {args.max} per file, threshold {args.threshold:g} dB\n")
    total = 0
    for f in files:
        try:
            name, n, note = extract(f, args.output, f_low, f_high, args.threshold, args.duration, args.max,
                                    args.min_gap, args.min_band_ratio, args.min_active)
        except subprocess.CalledProcessError:
            print(f"{os.path.basename(f)}: ! cannot read file (corrupted?)")
            continue
        total += n
        warning = "   <- no call detected: check the band or lower --threshold" if n == 0 and not note else ""
        print(f"{name}: {n} clip(s) {note}{warning}")

    print(f"\nDone: {total} clips in '{args.output}'")
    print("Listen to a few and delete those without the target species before training.")


if __name__ == "__main__":
    main()
