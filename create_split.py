#!/usr/bin/env python3
"""Create split.csv without moving audio or extracting clips.

Run: python create_split.py [--seed 42] [--output split.csv]
Paths in the recording column are relative to --root (default: this script's
folder). Classes are pygmy, boreal, negative. Negative subclasses identify the
species, risoux_background, or owl_background_pygmy/boreal. start_s/end_s are
seconds in the referenced file. All XC copies/windows share one source group,
including negatives from owl recordings. Risoux files each represent one source
recording. This prevents source-recording leakage, not recordist/site leakage.

Exclude exact clip paths listed in ambiguous.csv (no audio deletion), then
sample each class down to the smallest remaining class. Negative subclasses
receive near-equal quotas, redistributing unused capacity from small subclasses.
Sampling keeps one random window per source recording where the quota permits,
then randomly fills the remaining slots without replacement. When a subclass has
more recordings than slots (notably Risoux), randomly select source recordings.
70/10/20 is targeted within each class/subclass; whole-source groups make ratios
approximate. Seeded runs are reproducible for unchanged inputs. Existing
split.csv is replaced atomically only after validation. Run python split.py;
create_split.py remains supported. --dry-run reports counts without writing.
"""
import argparse
import csv
from collections import Counter, defaultdict
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


def exclude_ambiguous(samples, csv_path, root):
    """Exclude listed files, not all siblings from the same XC recording."""
    blocked = set()
    with csv_path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        if 'file_path' not in (reader.fieldnames or []):
            raise ValueError(f'{csv_path}: missing file_path column')
        for row in reader:
            path = Path(row['file_path'])
            if not row['file_path'].strip() or '..' in path.parts:
                raise ValueError(f'Invalid ambiguous file path: {path}')
            if path.is_absolute():
                if 'data' not in path.parts:
                    raise ValueError(f'Cannot relocate ambiguous path: {path}')
                path = Path(*path.parts[path.parts.index('data'):])
            blocked.add(path.as_posix())
    known = {s.recording for s in samples}
    unknown = blocked - known
    if unknown:
        raise ValueError(f'{len(unknown)} ambiguous paths are absent from input data: {sorted(unknown)[:3]}')
    remaining = [s for s in samples if s.recording not in blocked]
    print(f'Excluded {len(samples) - len(remaining):,} windows from {len(blocked):,} flagged files.')
    return remaining


def balanced_quotas(capacities, total, rng):
    """Integer water filling: near-equal counts subject to available capacity."""
    if total > sum(capacities.values()) or total < len(capacities):
        raise ValueError('Insufficient budget/capacity to represent every negative subclass')
    quotas = dict.fromkeys(capacities, 0)
    order = list(sorted(capacities))
    rng.shuffle(order)
    while total:
        active = [key for key in order if quotas[key] < capacities[key]]
        for key in active:
            if not total:
                break
            quotas[key] += 1
            total -= 1
    return quotas


def sample_windows(pool, count, rng):
    """Maximize source coverage, then randomly sample unused windows."""
    if count == len(pool):
        return list(pool)
    groups = defaultdict(list)
    for index, sample in enumerate(pool):
        groups[sample.group].append(index)
    group_names = sorted(groups)
    if count < len(group_names):
        chosen_groups = rng.choice(len(group_names), size=count, replace=False)
        chosen = {int(rng.choice(groups[group_names[i]])) for i in chosen_groups}
    else:
        chosen = {int(rng.choice(groups[g])) for g in group_names}
        available = [i for i in range(len(pool)) if i not in chosen]
        if count > len(chosen):
            chosen.update(map(int, rng.choice(available, size=count-len(chosen), replace=False)))
    return [sample for i, sample in enumerate(pool) if i in chosen]


def balance_samples(samples, seed):
    rng = np.random.default_rng(seed)
    pools = defaultdict(list)
    for sample in samples:
        pools[sample.label].append(sample)
    if set(pools) != {'pygmy', 'boreal', 'negative'}:
        raise ValueError('All three classes must contain eligible samples')
    target = min(map(len, pools.values()))
    selected = []
    for label in ('pygmy', 'boreal'):
        selected.extend(sample_windows(pools[label], target, rng))
    negative_pools = defaultdict(list)
    for sample in pools['negative']:
        negative_pools[sample.subclass].append(sample)
    quotas = balanced_quotas({k:len(v) for k,v in negative_pools.items()}, target, rng)
    for subclass in sorted(negative_pools):
        pool = negative_pools[subclass]
        selected.extend(sample_windows(pool, quotas[subclass], rng))
        n_sources = len({s.group for s in pool})
        if n_sources > quotas[subclass]:
            print(f'{subclass}: randomly retaining {quotas[subclass]:,} of {n_sources:,} sources (subclass budget).')
    assert Counter(s.label for s in selected) == dict.fromkeys(pools, target)
    print(f'Balanced classes: {target:,} each; {len(selected):,} total windows.')
    return selected


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
    parser.add_argument('--ambiguous', type=Path, default=Path('ambiguous.csv'))
    parser.add_argument('--dry-run', action='store_true', help='Validate and report without writing split.csv')
    args = parser.parse_args()
    root = args.root.resolve()
    print('Loading dataset...', flush=True)
    samples = load_samples(root)
    expected_strata = {s.stratum for s in samples}
    ambiguous = args.ambiguous if args.ambiguous.is_absolute() else root / args.ambiguous
    samples = exclude_ambiguous(samples, ambiguous, root)
    if {s.stratum for s in samples} != expected_strata:
        raise ValueError('Ambiguity exclusion removed an entire class/subclass')
    samples = balance_samples(samples, args.seed)
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
    if args.dry_run:
        print('Dry run complete; no files changed.')
        return
    protected = {root / name for name in ('ambiguous.csv', 'pygmy_detections.csv', 'boreal_detections.csv', 'negatives_owls.csv')}
    protected.add(ambiguous.resolve())
    if output.resolve() in protected:
        raise ValueError('Output cannot overwrite an input CSV')
    write_csv(samples, assignment, output)
    print(f'Wrote {output}. All windows from each source recording share one split.')


if __name__ == '__main__':
    main()
