#!/usr/bin/env python3
"""Create split.csv without moving audio or extracting clips.

Run: python create_split.py [--seed 42] [--output split.csv]
Paths in the recording column are relative to --root (default: this script's
folder). Classes are pygmy, boreal, negative. Negative subclasses identify the
species, risoux_background, or owl_background_pygmy/boreal. start_s/end_s are
seconds in the referenced file. All XC copies/windows share one source group,
including negatives from owl recordings. Risoux files each represent one source
recording. This prevents source-recording leakage, not recordist/site leakage.

70/10/20 is targeted by sample count within each class/subclass, preserving the
existing class proportions (no downsampling). Indivisible source groups mean
ratios are approximate. The seed makes the allocation reproducible for unchanged
inputs. Only the Python standard library and the existing numpy dependency are
needed. Existing split.csv is replaced atomically after validation.
"""
import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re
import os
import tempfile

import numpy as np

SPECIES = (
    'Otus scops', 'Strix aluco', 'Sitta europaea', 'Pyrrhula pyrrhula',
    'Cuculus canorus', 'Certhia familiaris', 'Periparus ater',
    'Lophophanes cristatus', 'Parus major', 'Poecile montanus',
    'Cyanistes caeruleus', 'Regulus regulus', 'Regulus ignicapilla',
    'Erithacus rubecula', 'Turdus merula', 'Turdus philomelos',
    'Fringilla coelebs', 'Loxia curvirostra', 'Troglodytes troglodytes',
)
AUDIO = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.wma', '.aiff', '.aif', '.audio'}
SPLITS = ('train', 'validation', 'test')
RATIOS = np.array([0.7, 0.1, 0.2])
FIELDS = ('class', 'subclass', 'recording', 'start_s', 'end_s', 'split')


@dataclass(frozen=True)
class Sample:
    label: str
    subclass: str
    recording: str
    start: str
    end: str
    group: str

    @property
    def stratum(self):
        return (self.label, self.subclass)


def source_group(path):
    match = re.match(r'^(XC\d+)(?:_|$)', path.stem)
    if match:
        return match.group(1)
    # Risoux filenames retain the original recorder/date/time identifier.
    return 'risoux:' + path.stem


def load_samples(root):
    samples = []
    checked = set()

    def relative_file(path):
        relative = path.relative_to(root).as_posix()
        if relative not in checked:
            if not path.is_file():
                raise ValueError(f'Missing audio: {path}')
            checked.add(relative)
        return relative

    for filename, label in [('pygmy_detections.csv', 'pygmy'),
                            ('boreal_detections.csv', 'boreal'),
                            ('negatives_owls.csv', 'negative')]:
        with (root / filename).open(newline='') as handle:
            reader = csv.DictReader(handle)
            required = {'recording_id', 'file_path', 'start_s', 'end_s'}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f'{filename}: missing columns {required}')
            for row in reader:
                # Old absolute paths survive relocation by retaining data/... .
                parts = Path(row['file_path']).parts
                if 'data' not in parts:
                    raise ValueError(f'Cannot resolve audio path: {row["file_path"]}')
                path = root.joinpath(*parts[parts.index('data'):])
                relative = relative_file(path)
                start, end = float(row['start_s']), float(row['end_s'])
                if not (np.isfinite(start) and np.isfinite(end) and start >= 0
                        and abs(end - start - 2) < 1e-5):
                    raise ValueError(f'Invalid two-second interval in {filename}: {row}')
                group = source_group(path)
                if group != row['recording_id']:
                    raise ValueError(f'Recording ID disagrees with filename: {row}')
                subclass = ''
                if label == 'negative':
                    if path.parent.name not in ('pygmy', 'boreal'):
                        raise ValueError(f'Unexpected owl background path: {path}')
                    subclass = 'owl_background_' + path.parent.name
                samples.append(Sample(label, subclass, relative,
                                      row['start_s'], row['end_s'], group))

    for species in SPECIES:
        folder = root / 'data' / 'negatives' / species.replace(' ', '_')
        clips = sorted((folder / 'clips').glob('*.wav'))
        if not clips:
            raise ValueError(f'No clips found: {folder}')
        for path in clips:
            if not re.fullmatch(r'XC\d+_\d+_\d+', path.stem):
                raise ValueError(f'Cannot identify clip source: {path}')
            samples.append(Sample('negative', species, relative_file(path),
                                  '0', '2', source_group(path)))

    folder = root / 'data' / 'negatives' / 'risoux_background'
    clips = sorted(p for p in folder.rglob('*') if p.suffix.lower() in AUDIO and p.is_file())
    if not clips:
        raise ValueError(f'No clips found: {folder}')
    for path in clips:
        samples.append(Sample('negative', 'risoux_background', relative_file(path),
                              '0', '2', source_group(path)))
    keys = {(s.label, s.subclass, s.recording, float(s.start), float(s.end)) for s in samples}
    if len(keys) != len(samples):
        raise ValueError('Duplicate sample rows in input; resolve before splitting.')
    return samples


def allocate(samples, seed):
    """Greedy grouped stratification followed by improving whole-group moves."""
    strata = sorted({s.stratum for s in samples})
    groups = sorted({s.group for s in samples})
    si = {s: i for i, s in enumerate(strata)}
    gi = {g: i for i, g in enumerate(groups)}
    counts = np.zeros((len(groups), len(strata)), dtype=np.int64)
    for sample in samples:
        counts[gi[sample.group], si[sample.stratum]] += 1
    totals = counts.sum(axis=0)
    target = RATIOS[:, None] * totals
    # Equal weight per stratum; error measured relative to its total size.
    weights = 1.0 / totals.astype(float) ** 2
    rng = np.random.default_rng(seed)
    # Large groups first; random tie-breaking avoids filename-order bias.
    importance = (counts / totals).max(axis=1)
    order = np.lexsort((rng.random(len(groups)), -importance))
    assigned = np.full(len(groups), -1, dtype=int)
    actual = np.zeros_like(target)
    for i in order:
        delta = ((2 * (actual - target) * counts[i] + counts[i] ** 2) * weights).sum(axis=1)
        dest = int(np.argmin(delta))
        assigned[i] = dest
        actual[dest] += counts[i]
    for _ in range(30):
        moved = False
        for i in rng.permutation(len(groups)):
            current = assigned[i]
            c = counts[i]
            remove = ((-2 * (actual[current] - target[current]) * c + c ** 2) * weights).sum()
            changes = ((2 * (actual - target) * c + c ** 2) * weights).sum(axis=1) + remove
            changes[current] = 0
            dest = int(np.argmin(changes))
            if changes[dest] < -1e-14:
                actual[current] -= c
                actual[dest] += c
                assigned[i] = dest
                moved = True
        if not moved:
            break
    return {g: SPLITS[assigned[i]] for i, g in enumerate(groups)}


def write_csv(samples, assignment, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', newline='', encoding='utf-8',
                                         dir=output.parent, delete=False) as handle:
            temporary = Path(handle.name)
            writer = csv.writer(handle)
            writer.writerow(FIELDS)
            for s in samples:
                writer.writerow((s.label, s.subclass, s.recording, s.start, s.end,
                                 assignment[s.group]))
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path, default=Path('split.csv'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    root = args.root.resolve()
    print('Loading dataset...', flush=True)
    samples = load_samples(root)
    print(f'Assigning {len(samples):,} windows from {len({s.group for s in samples}):,} source recordings...', flush=True)
    assignment = allocate(samples, args.seed)
    summary = Counter((s.stratum, assignment[s.group]) for s in samples)
    print(f'{"Class / subclass":46} {"train":>16} {"validation":>16} {"test":>16}')
    for stratum in sorted({s.stratum for s in samples}):
        counts = [summary[stratum, split] for split in SPLITS]
        total = sum(counts)
        print(f'{" / ".join(filter(None, stratum)):46} ' + ' '.join(f'{n:7,d} ({n/total:5.1%})' for n in counts))
        if not all(counts):
            raise ValueError(f'{stratum} cannot be represented in every split with this allocation.')
    output = args.output if args.output.is_absolute() else root / args.output
    write_csv(samples, assignment, output)
    print(f'Wrote {output}. All windows from each source recording share one split.')


if __name__ == '__main__':
    main()
