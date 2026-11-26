"""
Owl TinyML pipeline - Orchestrator (dataset -> augment -> spectrogram)

Builds one versioned dataset of spectrograms from split.csv:
  1. read split.csv and build the noise bank (TRAIN noise files only)
  2. create data/processed/<version>/ and open manifest.csv
  3. for each audio file: load it once and cut its windows (dataset.py)
  4. every window -> spectrogram .npy (spectrogram.py)
  5. TRAIN windows only -> N augmented copies (augment.py) -> spectrogram .npy

Output: data/processed/<version>/{train,validation,test}/<class>/*.npy + manifest.csv
        + config_used.py (copy of config.py) + summary.txt

Usage:
  python -m scripts.model.pipeline                         # full dataset -> data/processed/data_v1
  python -m scripts.model.pipeline --version data_test --limit 20    # quick test on 20 audio files
"""

import argparse
import random
import shutil
import time
import warnings
from collections import Counter
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config
from scripts.model import augment, dataset, spectrogram


warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    parser = argparse.ArgumentParser(description="Build a versioned spectrogram dataset from split.csv.")
    parser.add_argument("--version", default="data_v1",
                        help="name of the output folder in data/processed (must be new or empty)")
    parser.add_argument("--limit", type=int, default=0, metavar="N",
                        help="only process N audio files picked at random (quick test); 0 = all")
    args = parser.parse_args()


    random.seed(config.SEED)
    np.random.seed(config.SEED)

    # --- 1. Rows of split.csv + noise bank ---
    rows = dataset.read_split_rows()
    train_rows = [r for r in rows if r["split"] == "train"]
    noise_files = augment.build_noise_bank(train_rows)

    groups = dataset.group_by_recording(rows)
    recordings = list(groups)
    if args.limit:
        recordings = random.Random(config.SEED).sample(recordings, min(args.limit, len(recordings)))

    n_windows = sum(len(groups[rec]) for rec in recordings)
    print(f"split.csv     : {config.SPLIT_CSV}")
    print(f"Audio files   : {len(recordings)} ({n_windows} windows)")
    print(f"Noise bank    : {len(noise_files)} files")
    print(f"Augmentation  : {config.N_AUG_PER_WINDOW} copies per train window")

    # --- 2. Output folder + manifest.csv ---
    out_root = config.PROCESSED / args.version
    manifest_file, writer = spectrogram.start_output(out_root)
    shutil.copy(Path(config.__file__), out_root / "config_used.py")
    print(f"Output        : {out_root}\n")

    counts = Counter()          # (split, class, "original"/"augmented") -> number of .npy
    aug_counts = Counter()      # augmentation name -> number of copies
    failed = []                 # audio files that could not be read
    t0 = time.time()

    try:
        for i, recording in enumerate(recordings, 1):
            # --- 3. Load the file once, cut all its windows ---
            try:
                windows = dataset.load_windows(recording, groups[recording])
            except Exception as e:
                failed.append(f"{recording}: {e}")
                continue

            for row, signal in windows:
                # --- 4. Original window -> spectrogram (all splits) ---
                n = spectrogram.process_sound(out_root, writer, row, "original", signal)
                counts[(row["split"], row["class"], "original")] += n

                # --- 5. Augmented copies (train only) ---
                if row["split"] != "train":
                    continue
                for k in range(config.N_AUG_PER_WINDOW.get(row["class"], 0)):
                    aug_signal, aug, param = augment.augment_window(signal, row["class"], noise_files)
                    spectrogram.process_sound(out_root, writer, row, f"aug{k}", aug_signal, aug, f"{param:.3f}")
                    counts[(row["split"], row["class"], "augmented")] += 1
                    aug_counts[aug] += 1

            if i % 500 == 0 or i == len(recordings):
                elapsed = time.time() - t0
                remaining = elapsed / i * (len(recordings) - i)
                print(f"  {i}/{len(recordings)} files | {elapsed / 60:.1f} min | ~{remaining / 60:.1f} min left")
    finally:
        manifest_file.close()

    # --- Summary (printed + saved in summary.txt) ---
    lines = [f"Version   : {args.version}",
             f"split.csv : {config.SPLIT_CSV}",
             f"Seed      : {config.SEED} | limit: {args.limit or 'all'}",
             f"Time      : {(time.time() - t0) / 60:.1f} min", "",
             f"{'split':<11s} {'class':<10s} {'original':>9s} {'augmented':>10s} {'total':>8s}"]
    for split in ("train", "validation", "test"):
        for cls in config.CLASSES:
            o = counts[(split, cls, "original")]
            a = counts[(split, cls, "augmented")]
            if o or a:
                lines.append(f"{split:<11s} {cls:<10s} {o:9d} {a:10d} {o + a:8d}")
    lines.append(f"\nAugmentations used: {dict(aug_counts)}")
    lines.append(f"Failed audio files: {len(failed)}")
    lines += [f"  {f}" for f in failed]

    summary = "\n".join(lines)
    (out_root / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    print("\n" + summary)


if __name__ == "__main__":
    main()
