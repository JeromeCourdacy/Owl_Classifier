#!/usr/bin/env python3
"""Flag new negative-species clips using BirdNET (no BirdBox).

python scan_ambiguous.py
python scan_ambiguous.py --threshold 0.3 --output ambiguous.csv

Scores are BirdNET confidence scores, not calibrated presence probabilities.
Each two-second clip is mixed to mono, resampled to 48 kHz and padded with
0.5 seconds of silence on each side. Flag if EITHER owl score is strictly above
the threshold, even when another species ranks first. Audio and split.csv are
never changed. Relative file paths are relative to the project root.
The CSV is flushed as flags are found; an interrupted/failed run is incomplete.
Reruns replace the CSV. --limit supports a quick smoke test.
"""
import argparse
import csv
from pathlib import Path
import sys
import numpy as np
import soundfile as sf
import soxr
from tqdm import tqdm
from create_split import SPECIES
from detect_owls import SAMPLE_RATE, WINDOW, birdnet_predictor, ensure_birdnet_assets

FIELDS = ['recording_id', 'file_path', 'expected_species', 'expected_species_score',
          'pygmy_score', 'boreal_score']


def load_clip(path):
    audio, sr = sf.read(path, dtype='float32', always_2d=True)
    if abs(len(audio) / sr - 2) > 1 / sr:
        raise ValueError('Expected a two-second clip')
    audio = audio.mean(axis=1)
    if not np.all(np.isfinite(audio)):
        raise ValueError('Nonfinite audio samples')
    if sr != SAMPLE_RATE:
        audio = soxr.resample(audio, sr, SAMPLE_RATE, quality='HQ')
    if abs(len(audio) - WINDOW) > 1:
        raise ValueError('Unexpected resampled length')
    return np.pad(audio, (0, max(0, WINDOW - len(audio))))[:WINDOW]


def flagged(scores, indices, threshold):
    return any(float(scores[indices[name]]) > threshold for name in ('pygmy', 'boreal'))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path, default=Path('ambiguous.csv'))
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--limit', type=int, help='Only scan the first N clips (testing)')
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1 or (args.limit is not None and args.limit < 1):
        parser.error('Threshold must be in [0, 1]; limit must be positive')
    root = args.root.resolve()
    clips = []
    for species in SPECIES:
        folder = root / 'data' / 'negatives' / species.replace(' ', '_') / 'clips'
        paths = sorted(folder.glob('*.wav'))
        if not paths:
            raise ValueError(f'No clips found: {folder}')
        clips.extend((path, species) for path in paths)
    if args.limit is not None:
        clips = clips[:args.limit]
    ensure_birdnet_assets()
    predict, indices = birdnet_predictor()
    from birdnet_analyzer import config
    labels = Path(config.BIRDNET_LABELS_FILE).read_text().splitlines()
    species_indices = {label.split('_', 1)[0]: i for i, label in enumerate(labels)}
    missing = set(SPECIES) - species_indices.keys()
    if missing:
        raise ValueError(f'Expected species absent from model: {sorted(missing)}')
    output = args.output if args.output.is_absolute() else root / args.output
    if output.resolve() in {p.resolve() for p in root.glob('*.csv') if p.name != 'ambiguous.csv'}:
        raise ValueError('Output must not overwrite an existing dataset CSV')
    output.parent.mkdir(parents=True, exist_ok=True)
    count = errors = 0
    per_species = {species: 0 for species in SPECIES}
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        handle.flush()
        with tqdm(clips, desc='Checking clips', unit='clip', mininterval=5) as bar:
            for path, species in bar:
                try:
                    scores = predict(load_clip(path))  # predictor pads to 3 s
                    if not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
                        raise ValueError('Invalid BirdNET scores')
                except Exception as exc:
                    errors += 1
                    tqdm.write(f'ERROR {path}: {exc}', file=sys.stderr)
                    continue
                if flagged(scores, indices, args.threshold):
                    writer.writerow({
                        'recording_id': path.stem.split('_', 1)[0],
                        'file_path': path.relative_to(root).as_posix(),
                        'expected_species': species,
                        'expected_species_score': f'{float(scores[species_indices[species]]):.10g}',
                        'pygmy_score': f'{float(scores[indices["pygmy"]]):.10g}',
                        'boreal_score': f'{float(scores[indices["boreal"]]):.10g}',
                    })
                    handle.flush()
                    count += 1
                    per_species[species] += 1
                bar.set_postfix(flagged=count, errors=errors, refresh=False)
    for species, n in per_species.items():
        print(f'{species}: {n:,} flagged')
    print(f'Scanned {len(clips):,} clips; flagged {count:,}; errors {errors:,}. Output: {output}', flush=True)
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
