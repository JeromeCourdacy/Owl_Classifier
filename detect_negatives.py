#!/usr/bin/env python3
"""Find candidate two-second negatives using BirdNET alone (no BirdBox).

  conda activate owls
  python detect_negatives.py
  python detect_negatives.py data/pygmy data/boreal --threshold 0.05
  python detect_negatives.py data/pygmy/XC1008185.mp3 --save-clips
  python detect_negatives.py --self-test

Defaults: recursively scan data/, write negatives_owls.csv, no audio export.
Scan full audio with 3-second windows at a 1-second hop, including an end-anchored
last window. Remove the whole window plus a 1-second margin whenever EITHER owl
score is >=0.01, regardless of which species ranks first. Invalid scores also
exclude a window. Divide every remaining range into consecutive, nonoverlapping
two-second slots, starting at that range's beginning. Discard leftover fragments
shorter than two seconds. There is no cap, sampling, or sliding at this stage.
Recheck each exact PCM16 slot with 0.5 seconds of silence on each side; both
scores must remain strictly below the threshold. Keep all passing slots.
No samples from outside the candidate are used during this final check.

Recordings stream through the decoder/resampler in bounded chunks. A second
progress bar shows the current recording's scan. Files must be decodable by
SoundFile; corrupt/unsupported files are reported and skipped. The output is
flushed per recording; it is replaced on reruns. Positive CSVs are never read or
modified. --save-clips exports only accepted WAVs to <CSV_stem>_clips/ (empty or
absent). The CSV file_path always refers to the original recording. Its owl
scores are from the final padded two-second candidate, not absence probabilities.

These are CANDIDATE negatives, not guaranteed owl-free recordings. Listen to
samples to tune the threshold. Keep all positives and negatives from a source
recording in the same train/validation/test split. Other birds/owls are allowed:
this excludes only Glaucidium passerinum and Aegolius funereus. Existing owls
requirements and model cache are reused; missing BirdNET weights download once.
"""

import argparse
import csv
import hashlib
import math
import os
from pathlib import Path
import sys
import tempfile

from detect_owls import (SAMPLE_RATE, WINDOW, birdnet_predictor,
                         discover_recordings, ensure_birdnet_assets, storage)

FRAME = 3 * SAMPLE_RATE
FIELDS = ["recording_id", "file_path", "start_s", "end_s", "pygmy_score", "boreal_score"]


def pcm16(audio):
    import numpy as np
    if not np.all(np.isfinite(audio)):
        raise ValueError("Nonfinite audio samples")
    return np.clip(np.rint(audio * 32768), -32768, 32767).astype(np.int16)


def scan_windows(path, hop):
    """Decode/resample continuously; cover the complete timeline with <=3s windows."""
    import numpy as np
    import soundfile as sf
    import soxr
    buffer = np.empty(0, dtype=np.float32)
    tail = buffer
    offset = total = last_end = 0
    with sf.SoundFile(path) as source:
        resampler = None if source.samplerate == SAMPLE_RATE else soxr.ResampleStream(
            source.samplerate, SAMPLE_RATE, 1, dtype="float32", quality="HQ")
        block_size = source.samplerate * 60
        block = source.read(block_size, dtype="float32", always_2d=True)
        while len(block):
            following = source.read(block_size, dtype="float32", always_2d=True)
            mono = block.mean(axis=1)
            if not np.all(np.isfinite(mono)):
                raise ValueError("Nonfinite audio samples")
            if resampler is not None:
                mono = resampler.resample_chunk(mono, last=not len(following))
            total += len(mono)
            tail = np.concatenate((tail, mono))[-FRAME:].copy()
            buffer = np.concatenate((buffer, mono))
            while len(buffer) >= FRAME:
                last_end = offset + FRAME
                yield offset, last_end, buffer[:FRAME]
                buffer = buffer[hop:]
                offset += hop
            block = following
    if not total:
        raise ValueError("Empty recording")
    if total < FRAME:
        yield 0, total, tail
    elif last_end < total:
        yield total - FRAME, total, tail


def owl_scores(scores, indices):
    import numpy as np
    values = tuple(float(scores[indices[name]]) for name in ("pygmy", "boreal"))
    if not np.all(np.isfinite(scores)) or not all(0 <= value <= 1 for value in values):
        return float("inf"), float("inf")
    return values


def add_exclusion(intervals, start, end):
    """Merge chronologically generated exclusions, including touching intervals."""
    if intervals and start <= intervals[-1][1]:
        intervals[-1] = (intervals[-1][0], max(end, intervals[-1][1]))
    else:
        intervals.append((start, end))


def safe_ranges(total, exclusions):
    ranges = []
    cursor = 0
    for start, end in exclusions:
        start, end = max(0, start), min(total, end)
        if start - cursor >= WINDOW:
            ranges.append((cursor, start))
        cursor = max(cursor, end)
    if total - cursor >= WINDOW:
        ranges.append((cursor, total))
    return ranges


def candidates(ranges):
    """Tile every eligible range with all complete, nonoverlapping 2-second slots."""
    for start, end in ranges:
        for left in range(start, end - WINDOW + 1, WINDOW):
            yield left, left + WINDOW


def read_candidate(source, start, end):
    """Read only the chosen source interval and create the exact scored PCM16 data."""
    import numpy as np
    import soxr
    first = round(start * source.samplerate / SAMPLE_RATE)
    last = round(end * source.samplerate / SAMPLE_RATE)
    source.seek(first)
    audio = source.read(last - first, dtype="float32", always_2d=True)
    if len(audio) != last - first:
        raise ValueError("Incomplete candidate interval")
    audio = audio.mean(axis=1)
    if source.samplerate != SAMPLE_RATE:
        audio = soxr.resample(audio, source.samplerate, SAMPLE_RATE, quality="HQ")
    if abs(len(audio) - WINDOW) > 1:
        raise ValueError("Unexpected resampled candidate length")
    return pcm16(np.pad(audio, (0, max(0, WINDOW - len(audio))))[:WINDOW])


def process(path, predict, indices, args):
    import numpy as np
    import soundfile as sf
    from tqdm import tqdm
    duration = sf.info(path).duration
    exclusions = []
    total = 0
    scan_count = 0
    margin = round(args.margin * SAMPLE_RATE)
    with tqdm(total=duration, desc=path.name, unit="s", position=1, leave=False,
              dynamic_ncols=True) as bar:
        for start, end, audio in scan_windows(path, round(args.hop * SAMPLE_RATE)):
            scores = owl_scores(predict(pcm16(audio).astype(np.float32) / 32768), indices)
            if max(scores) >= args.threshold:
                add_exclusion(exclusions, max(0, start - margin), end + margin)
            total = max(total, end)
            scan_count += 1
            bar.update(max(0, min(duration, end / SAMPLE_RATE) - bar.n))
    if total < WINDOW:
        raise ValueError("Recording is shorter than two seconds")
    ranges = safe_ranges(total, exclusions)
    rows = []
    source_hash = hashlib.sha256(str(path).encode()).hexdigest()[:12]
    with sf.SoundFile(path) as source:
        slot_count = sum((end-start) // WINDOW for start,end in ranges)
        for start, end in tqdm(candidates(ranges), total=slot_count, desc="Checking negatives",
                               unit="slot", position=1, leave=False, dynamic_ncols=True):
            clip = read_candidate(source, start, end)
            pygmy, boreal = owl_scores(predict(clip.astype(np.float32) / 32768), indices)
            if max(pygmy, boreal) >= args.threshold:
                continue
            if args.save_clips:
                args.clips_dir.mkdir(parents=True, exist_ok=True)
                sf.write(args.clips_dir / f"{path.stem}_{source_hash}_{start}_{end}.wav",
                         clip, SAMPLE_RATE, subtype="PCM_16")
            rows.append({"recording_id": path.stem, "file_path": str(path),
                         "start_s": f"{start / SAMPLE_RATE:.9f}",
                         "end_s": f"{end / SAMPLE_RATE:.9f}",
                         "pygmy_score": f"{pygmy:.10g}", "boreal_score": f"{boreal:.10g}"})
    rows.sort(key=lambda row: float(row["start_s"]))
    tqdm.write(f"{path.name}: {scan_count} scan windows, "
               f"{sum(end-start for start,end in ranges) / SAMPLE_RATE:.1f}s eligible, "
               f"{len(rows)} candidate negatives")
    return rows


def self_test():
    import numpy as np
    import soundfile as sf
    indices = {"pygmy": 0, "boreal": 1}
    assert max(owl_scores(np.array([.02, .001, .99]), indices)) >= .01
    assert max(owl_scores(np.array([.001, .02, .99]), indices)) >= .01
    assert max(owl_scores(np.array([.001, .002, .99]), indices)) < .01
    assert max(owl_scores(np.array([np.nan, 0, .99]), indices)) >= .01
    exclusions = []
    add_exclusion(exclusions, SAMPLE_RATE, 3 * SAMPLE_RATE)
    add_exclusion(exclusions, 2 * SAMPLE_RATE, 4 * SAMPLE_RATE)
    assert exclusions == [(SAMPLE_RATE, 4 * SAMPLE_RATE)]
    assert safe_ranges(8 * SAMPLE_RATE, exclusions) == [(4 * SAMPLE_RATE, 8 * SAMPLE_RATE)]
    assert safe_ranges(5 * SAMPLE_RATE, [(0, 10 * SAMPLE_RATE)]) == []
    ranges = [(0, 9 * SAMPLE_RATE), (15 * SAMPLE_RATE, 22 * SAMPLE_RATE)]
    chosen = list(candidates(ranges))
    assert chosen == list(candidates(ranges))
    assert list(candidates([(0, 45 * SAMPLE_RATE)])) == [(i * WINDOW, (i+1) * WINDOW) for i in range(22)]
    assert list(candidates([(0, WINDOW-1)])) == []
    assert list(candidates([(5, 5+WINDOW)])) == [(5, 5+WINDOW)]
    ordered = sorted(chosen)
    assert len(chosen) == 7
    assert all(end-start == WINDOW and any(a <= start < end <= b for a,b in ranges)
               for start,end in chosen)
    assert all(a[1] <= b[0] for a,b in zip(ordered, ordered[1:]))
    with tempfile.TemporaryDirectory() as folder:
        for rate, seconds in ((44100, 5.2), (48000, 2), (48000, 3), (48000, 61.2)):
            path = Path(folder) / "audio.wav"
            sf.write(path, np.zeros(round(seconds * rate), dtype=np.float32), rate)
            windows = list(scan_windows(path, SAMPLE_RATE))
            total = round(seconds * SAMPLE_RATE)
            assert windows[0][0] == 0 and windows[-1][1] == total
            assert all(len(audio) == end-start <= FRAME for start,end,audio in windows)
            assert all(a[1] >= b[0] for a,b in zip(windows, windows[1:]))
            with sf.SoundFile(path) as source:
                assert read_candidate(source, 0, WINDOW).shape == (WINDOW,)
    print("Both-species exclusion, margins, uncapped nonoverlapping slots, streaming and exact-length checks passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recordings", nargs="*", type=Path)
    parser.add_argument("--output", type=Path, default=Path("negatives_owls.csv"))
    parser.add_argument("--threshold", type=float, default=.01)
    parser.add_argument("--margin", type=float, default=1., help="Seconds excluded on each side of suspect intervals")
    parser.add_argument("--hop", type=float, default=1., help="Step between three-second scan windows")
    parser.add_argument("--save-clips", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.threshold) or not 0 < args.threshold <= 1:
        parser.error("--threshold must be in (0, 1]")
    if not math.isfinite(args.hop) or not 1 / SAMPLE_RATE <= args.hop <= 3:
        parser.error("--hop must be between one sample and three seconds")
    if not math.isfinite(args.margin) or args.margin < 0:
        parser.error("--margin must be finite and nonnegative")
    if args.self_test:
        self_test()
        return 0
    try:
        paths = discover_recordings(args.recordings or [Path("data")])
    except ValueError as error:
        parser.error(str(error))
    if not paths:
        parser.error("No supported audio recordings found")
    if args.output.suffix.lower() != ".csv" or args.output.resolve() in paths:
        parser.error("--output must be a CSV distinct from the recordings")
    positive_csvs = {Path(f"{species}_detections.csv").resolve() for species in ("pygmy", "boreal")}
    if args.output.resolve() in positive_csvs:
        parser.error("Choose a separate negative CSV; positive outputs must be preserved")
    args.clips_dir = args.output.parent / f"{args.output.stem}_clips"
    if args.save_clips and args.clips_dir.exists():
        if not args.clips_dir.is_dir() or any(args.clips_dir.iterdir()):
            parser.error(f"Clip directory must be empty or absent: {args.clips_dir}")
    cache = storage() / "cache"
    for name in ("matplotlib", "fontconfig", "numba"):
        (cache / name).mkdir(parents=True, exist_ok=True)
    for key, value in {"MPLCONFIGDIR": cache / "matplotlib", "XDG_CACHE_HOME": cache,
                       "NUMBA_CACHE_DIR": cache / "numba"}.items():
        os.environ.setdefault(key, str(value))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    ensure_birdnet_assets()
    predict, indices = birdnet_predictor()
    from tqdm import tqdm
    args.output.parent.mkdir(parents=True, exist_ok=True)
    failures = count = 0
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        output.flush()
        with tqdm(paths, desc="Recordings", unit="recording", position=0, dynamic_ncols=True) as progress:
            for path in progress:
                try:
                    rows = process(path, predict, indices, args)
                    writer.writerows(rows)
                    output.flush()
                    count += len(rows)
                except Exception as error:
                    tqdm.write(f"FAILED {path}: {type(error).__name__}: {error}", file=sys.stderr)
                    failures += 1
                progress.set_postfix(negatives=count, failed=failures, refresh=False)
    print(f"{count} candidate negatives → {args.output.resolve()}; failed recordings: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
