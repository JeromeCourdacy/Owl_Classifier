#!/usr/bin/env python3
"""Find randomized two-second pygmy/boreal owl windows across audio recordings.

Setup (Python 3.12):
  conda create -n owls python=3.12 pip -y
  conda activate owls
  pip install -r requirements.txt
Run the entire dataset (data/ by default, recursively):
  python detect_owls.py --seed 42
  python detect_owls.py data/pygmy data/boreal --seed 42
Test specific recordings, optionally exporting WAV clips:
  python detect_owls.py test.mp3 test2.mp3 --seed 42 --save-clips
  python detect_owls.py --self-test

Writes exactly two CSVs: pygmy_detections.csv and boreal_detections.csv.
Use --output-dir to choose their parent directory. Columns: recording_id (file
stem), file_path (absolute original audio path), confidence, start_s, end_s.
Each input is analyzed once. The winning BirdNET species determines the output
CSV, regardless of the source folder; other species are discarded. The model
label list is checked for BOTH target species before analysis. Just-Bird is a
single-class bird detector, not an owl species classifier.

BirdBox positions two-second intervals with uniformly randomized context before
and after each box, keeping the whole box inside the interval where possible.
Long boxes are split into two-second pieces, which have no room for jitter.
The exact PCM16 candidate is silence-padded by 0.5 s on each side for BirdNET.
Only top-ranked target predictions scoring >= --min-confidence (default 0.5)
are reported. Identical intervals are deduplicated; overlapping ones remain.
--seed reproduces random positions per recording ID. Duplicate file stems in
separate folders are allowed; file_path distinguishes their CSV entries.

Default output is CSV only. --save-clips also exports accepted mono PCM16 WAVs
at 48 kHz to pygmy_detections_clips/ and boreal_detections_clips/ in the output
directory. These directories must be empty or absent. Clip filenames include
recording ID, a hash of the source path, and start/end sample indices.

Models load once and recordings are processed sequentially. CSVs are replaced
on each run and flushed after each completed recording; interrupted runs may
contain partial results. Unreadable/invalid recordings are reported on stderr
and skipped; the program exits nonzero if any failed. Inputs under two seconds
are invalid. Supported extensions include WAV, MP3, FLAC, OGG, M4A, WMA, AIFF,
AIF and the downloader's .audio fallback. Metadata files are ignored.

These are model-selected intervals, not human-verified labels or exact call
boundaries. Simultaneous species are not separated. BirdBox source and weights
download automatically on first run and stay in the conda environment with
runtime caches. Later runs reuse them offline.
BirdBox: https://github.com/birdnet-team/BirdBox (MIT code, AGPL-3.0 weights).
BirdNET: https://github.com/birdnet-team/BirdNET-Analyzer (model terms apply).
"""

import argparse
import csv
from contextlib import ExitStack, redirect_stdout
import hashlib
import math
import os
import random
from pathlib import Path
import shutil
import sys
import tempfile
from urllib.request import urlopen
import zipfile

BIRDBOX_COMMIT = "1fefa6d304822185ac53bd39cd8888b1bd2b88d1"
BIRDBOX_WEIGHT_URL = "https://github.com/birdnet-team/BirdBox/releases/download/v2.0.0/Just-Bird_fp32.pt"
BIRDBOX_WEIGHT_SHA = "e54f6e3886a93691a56d653f5c2c773a7f13f6f50cd75e61f66eaea19af89ebc"
TARGETS = {"pygmy": "Glaucidium passerinum", "boreal": "Aegolius funereus"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".wma", ".aiff", ".aif", ".audio"}
SAMPLE_RATE = 48000
WINDOW = 2 * SAMPLE_RATE
FIELDS = ["recording_id", "file_path", "confidence", "start_s", "end_s"]


def storage():
    return Path(sys.prefix) / "share" / "owls"


def download(url, destination, sha256=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with urlopen(url, timeout=120) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        if sha256 and hashlib.sha256(partial.read_bytes()).hexdigest() != sha256:
            raise RuntimeError(f"Checksum mismatch: {destination.name}")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def ensure_assets():
    """Fetch missing external code/models once; reuse installed assets offline."""
    root = storage()
    source = root / "birdbox"
    marker = source / "COMMIT"
    if not marker.exists() or marker.read_text().strip() != BIRDBOX_COMMIT:
        print("Installing pinned BirdBox source inside the environment...", flush=True)
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "birdbox.zip"
            download(f"https://codeload.github.com/birdnet-team/BirdBox/zip/{BIRDBOX_COMMIT}", archive)
            with zipfile.ZipFile(archive) as zipped:
                prefix = f"BirdBox-{BIRDBOX_COMMIT}/"
                # Copy only inference source, taxonomy resources and licence files.
                for member in zipped.infolist():
                    relative = Path(member.filename).relative_to(prefix)
                    if not relative.parts or ".." in relative.parts:
                        continue
                    if relative.parts[0] not in {"src", "taxonomies", "LICENSE", "LICENSE-AGPL-3.0", "NOTICE.md"}:
                        continue
                    target = source / relative
                    if member.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(zipped.read(member))
        marker.write_text(BIRDBOX_COMMIT + "\n")
    weights = root / "Just-Bird_fp32.pt"
    if not weights.exists() or hashlib.sha256(weights.read_bytes()).hexdigest() != BIRDBOX_WEIGHT_SHA:
        print("Downloading BirdBox Just-Bird weights...", flush=True)
        download(BIRDBOX_WEIGHT_URL, weights, BIRDBOX_WEIGHT_SHA)
    ensure_birdnet_assets()


def ensure_birdnet_assets():
    """Provision BirdNET alone, without downloading or importing BirdBox."""
    from birdnet_analyzer import config, utils
    if not all(Path(path).is_file() for path in (config.BIRDNET_MODEL_PATH, config.BIRDNET_LABELS_FILE)):
        print("Downloading BirdNET weights...", flush=True)
        utils.ensure_model_exists()


def pieces(start, end, length=WINDOW):
    """Cover a box with <=length pieces, end-anchoring any long-box tail."""
    if end <= start:
        return []
    if end - start <= length:
        return [(start, end)]
    starts = list(range(start, end - length + 1, length))
    if starts[-1] != end - length:
        starts.append(end - length)
    return [(s, s + length) for s in starts]


def clip_bounds(start, end, total, rng, length=WINDOW):
    """Uniformly sample a full-length crop containing the complete box/piece."""
    if total < length:
        raise ValueError("Recording is shorter than two seconds")
    if not 0 <= start < end <= total or end - start > length:
        raise ValueError("Box/piece must fit inside the recording and crop")
    earliest = max(0, end - length)
    latest = min(start, total - length)
    left = rng.randint(earliest, latest)
    return left, left + length


def padded(audio):
    import numpy as np
    if not 0 < len(audio) <= 3 * SAMPLE_RATE:
        raise ValueError("Expected between zero and three seconds of nonempty audio")
    missing = 3 * SAMPLE_RATE - len(audio)
    return np.pad(audio, (missing // 2, missing - missing // 2)).astype("float32")


def is_target(scores, target_index, threshold):
    import numpy as np
    return bool(np.all(np.isfinite(scores)) and int(np.argmax(scores)) == target_index
                and float(scores[target_index]) >= threshold)


def birdnet_predictor():
    import numpy as np
    from birdnet_analyzer import config, model
    if not Path(config.BIRDNET_MODEL_PATH).is_file():
        raise RuntimeError("BirdNET model download did not complete; rerun with internet access")
    config.MODEL_PATH = config.BIRDNET_MODEL_PATH
    config.TFLITE_THREADS = 2
    labels = Path(config.BIRDNET_LABELS_FILE).read_text().splitlines()
    scientific_names = {name.split("_", 1)[0]: i for i, name in enumerate(labels)}
    missing = set(TARGETS.values()) - scientific_names.keys()
    if missing:
        raise RuntimeError(f"BirdNET model does not cover: {', '.join(sorted(missing))}")
    target_indices = {key: scientific_names[name] for key, name in TARGETS.items()}
    print("BirdNET coverage: " + ", ".join(TARGETS.values()), flush=True)

    def predict(audio):
        logits = model.predict(np.expand_dims(padded(audio), 0))[0]
        scores = model.flat_sigmoid(logits, sensitivity=-1, bias=1.0)
        if len(scores) != len(labels):
            raise RuntimeError("BirdNET labels and model output do not match")
        return scores

    return predict, target_indices


def process(path, detector, predict, target_indices, args):
    import librosa
    import numpy as np
    import soundfile as sf
    rng = random.Random(f"{args.seed}:{path.stem}" if args.seed is not None else None)
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(audio) < WINDOW or not np.all(np.isfinite(audio)):
        raise ValueError(f"{path}: audio must be finite and at least two seconds long")
    print(f"{path.name}: locating BirdBox intervals...", flush=True)
    boxes = detector.detect_single_file(str(path))
    print(f"{path.name}: checking {len(boxes)} BirdBox intervals with BirdNET...", flush=True)
    checked = set()
    rows = {key: [] for key in TARGETS}
    source_hash = hashlib.sha256(str(path).encode()).hexdigest()[:12]
    for box in boxes:
        start = max(0, math.floor(float(box["time_start"]) * SAMPLE_RATE))
        end = min(len(audio), math.ceil(float(box["time_end"]) * SAMPLE_RATE))
        for left, right in pieces(start, end):
            bounds = clip_bounds(left, right, len(audio), rng)
            if bounds in checked:
                continue
            checked.add(bounds)
            # Score the same PCM16 samples whether or not WAV export is enabled.
            pcm = np.clip(np.rint(audio[bounds[0]:bounds[1]] * 32768), -32768, 32767).astype(np.int16)
            scores = predict(pcm.astype(np.float32) / 32768)
            species = accepted_species(scores, target_indices, args.min_confidence)
            if species is not None:
                if args.save_clips:
                    clips_dir = args.output_dir / f"{species}_detections_clips"
                    clips_dir.mkdir(parents=True, exist_ok=True)
                    destination = clips_dir / f"{path.stem}_{source_hash}_{bounds[0]}_{bounds[1]}.wav"
                    sf.write(destination, pcm, SAMPLE_RATE, subtype="PCM_16")
                rows[species].append({"recording_id": path.stem, "file_path": str(path),
                                      "confidence": f"{float(scores[target_indices[species]]):.6f}",
                                      "start_s": f"{bounds[0] / SAMPLE_RATE:.9f}",
                                      "end_s": f"{bounds[1] / SAMPLE_RATE:.9f}"})
    for values in rows.values():
        values.sort(key=lambda row: float(row["start_s"]))
    print(f"{path.name}: {len(boxes)} boxes, {len(checked)} intervals checked; "
          + ", ".join(f"{key}={len(values)}" for key, values in rows.items()), flush=True)
    return rows


def accepted_species(scores, target_indices, threshold):
    import numpy as np
    winner = int(np.argmax(scores))
    for species, index in target_indices.items():
        if winner == index and is_target(scores, index, threshold):
            return species
    return None


def discover_recordings(inputs):
    paths = set()
    for item in inputs:
        if item.is_dir():
            paths.update(p.resolve() for p in item.rglob("*")
                         if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)
        elif item.is_file() and item.suffix.lower() in AUDIO_EXTENSIONS:
            paths.add(item.resolve())
        else:
            raise ValueError(f"Not a supported audio file or directory: {item}")
    return sorted(paths)


def self_test():
    import numpy as np
    assert pieces(0, 5, 2) == [(0, 2), (2, 4), (3, 5)]
    assert pieces(2, 3, 2) == [(2, 3)]
    assert pieces(3, 3, 2) == []
    rng = random.Random(42)
    assert clip_bounds(0, 1, 10, rng, 2) == (0, 2)
    assert clip_bounds(9, 10, 10, rng, 2) == (8, 10)
    assert clip_bounds(3, 5, 10, rng, 2) == (3, 5)
    a, b = random.Random(42), random.Random(42)
    positions = [clip_bounds(4, 5, 10, a, 4) for _ in range(100)]
    assert positions == [clip_bounds(4, 5, 10, b, 4) for _ in range(100)]
    assert len(set(positions)) > 1
    assert all(0 <= left <= 4 < 5 <= right <= 10 and right-left == 4 for left,right in positions)
    try:
        clip_bounds(0, 1, 1, rng, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("Short recordings must be rejected")
    sample = np.ones(WINDOW, dtype=np.float32)
    result = padded(sample)
    assert len(result) == 3 * SAMPLE_RATE
    assert np.all(result[:SAMPLE_RATE // 2] == 0)
    assert np.all(result[SAMPLE_RATE // 2:-SAMPLE_RATE // 2] == 1)
    assert np.all(result[-SAMPLE_RATE // 2:] == 0)
    assert is_target(np.array([0.9, 0.2]), 0, 0.5)
    assert not is_target(np.array([0.9, 0.95]), 0, 0.5)
    assert not is_target(np.array([0.49, 0.2]), 0, 0.5)
    assert not is_target(np.array([np.nan, 0.2]), 0, 0.5)
    indices = {"pygmy": 0, "boreal": 1}
    assert accepted_species(np.array([.9, .2, .1]), indices, .5) == "pygmy"
    assert accepted_species(np.array([.1, .9, .2]), indices, .5) == "boreal"
    assert accepted_species(np.array([.6, .7, .9]), indices, .5) is None
    assert accepted_species(np.array([.2, .4, .1]), indices, .5) is None
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        (root / "nested").mkdir()
        (root / "nested" / "XC1.MP3").touch()
        (root / "XC1.wav").touch()
        (root / "manifest.jsonl").touch()
        paths = discover_recordings([root, root / "nested"])
        assert len(paths) == 2 and all(path.stem == "XC1" for path in paths)
    print("Random crop, padding, both-species routing and recursive discovery checks passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recordings", nargs="*", type=Path, help="Audio files or recursive directories (default: data/)")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Parent of the two output CSVs (default: current directory)")
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--box-confidence", type=float, default=0.15)
    parser.add_argument("--seed", type=int, help="Reproduce random crop positions (default: fresh randomness)")
    parser.add_argument("--save-clips", action="store_true", help="Also export accepted two-second WAVs, grouped by species")
    args = parser.parse_args()
    if not 0 < args.min_confidence <= 1 or not 0 < args.box_confidence <= 1:
        parser.error("Confidence thresholds must be in (0, 1]")
    cache = storage() / "cache"
    for subdir in ("matplotlib", "fontconfig", "ultralytics", "numba"):
        (cache / subdir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(cache / "ultralytics"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache / "numba"))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if args.self_test:
        self_test()
        return 0
    try:
        paths = discover_recordings(args.recordings or [Path("data")])
    except ValueError as error:
        parser.error(str(error))
    if not paths:
        parser.error("No supported audio recordings found")
    if args.output_dir.exists() and not args.output_dir.is_dir():
        parser.error("--output-dir must be a directory")
    if args.save_clips:
        for species in TARGETS:
            folder = args.output_dir / f"{species}_detections_clips"
            if folder.exists() and (not folder.is_dir() or any(folder.iterdir())):
                parser.error(f"Clip directory must be empty or absent: {folder}")
    ensure_assets()
    sys.path.insert(0, str(storage() / "birdbox" / "src"))
    print("Loading BirdBox Just-Bird (general bird detector)...", flush=True)
    from inference.detect_birds import BirdCallDetector
    detector = BirdCallDetector(str(storage() / "Just-Bird_fp32.pt"), "Just-Bird",
                                conf_threshold=args.box_confidence, phrase_gap=0,
                                num_workers=1, verbose=False)
    print("Loading BirdNET...", flush=True)
    predict, target_indices = birdnet_predictor()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = dict.fromkeys(TARGETS, 0)
    failures = 0
    with ExitStack() as stack:
        files, writers = {}, {}
        for species in TARGETS:
            destination = args.output_dir / f"{species}_detections.csv"
            files[species] = stack.enter_context(destination.open("w", newline="", encoding="utf-8"))
            writers[species] = csv.DictWriter(files[species], fieldnames=FIELDS)
            writers[species].writeheader()
            files[species].flush()
        from tqdm import tqdm
        from tqdm.contrib import DummyTqdmFile
        # Route per-recording messages above the bar so it stays readable.
        with tqdm(total=len(paths), desc="Recordings", unit="recording", dynamic_ncols=True) as progress:
            with redirect_stdout(DummyTqdmFile(sys.stdout)):
                for path in paths:
                    progress.set_postfix(remaining=len(paths) - progress.n, failed=failures, refresh=False)
                    try:
                        rows = process(path, detector, predict, target_indices, args)
                        for species in TARGETS:
                            writers[species].writerows(rows[species])
                            files[species].flush()
                            counts[species] += len(rows[species])
                    except Exception as error:
                        tqdm.write(f"FAILED {path}: {type(error).__name__}: {error}", file=sys.stderr)
                        failures += 1
                    progress.set_postfix(remaining=len(paths) - progress.n - 1, failed=failures, refresh=False)
                    progress.update(1)
    for species, count in counts.items():
        print(f"{species}: {count} windows → {(args.output_dir / f'{species}_detections.csv').resolve()}")
    print(f"Recordings completed: {len(paths)-failures}/{len(paths)}; failed: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
