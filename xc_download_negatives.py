#!/usr/bin/env python3
"""Download European foreground hard negatives from xeno-canto, with progress.

From this repository, using the existing owls environment:
  conda activate owls
  python xc_download_negatives.py

Optional:
  python xc_download_negatives.py --dry-run
  python xc_download_negatives.py --species "Otus scops" --hours 5

Reads XC_KEY from .env or the environment. Defaults to the 19 requested species
and five hours each. Original complete recordings go to data/negatives/<Genus_species>/.
The final complete recording can take the total above five hours. No cropping,
conversion or acoustic filtering is performed. All qualities/types are allowed.

Uses area:europe and exact foreground scientific-name checks. Rejects recordings
listing either target owl in background metadata, and conservatively rejects
remarks mentioning either owl. Missing/malformed background metadata is rejected.
An empty background list is allowed. Unannotated owls can still be audible; this
is metadata screening, not a guarantee of acoustic absence.

Each folder has one append-only manifest.jsonl with original metadata (including
licence, recordist and source), selection decisions, duration and filename. Reruns
revalidate previous successful downloads, count them toward the target and skip
existing files. Failed/interrupted downloads can be retried. Existing files are
never deleted; a newly downloaded file that fails decoding is removed.

--dry-run fetches metadata and reports estimated availability without creating
folders or downloading audio. A shortfall is reported if eligible downloadable
recordings cannot reach the target. Progress reflects audio duration, not bytes.
"""

import argparse
import json
import math
from pathlib import Path
import random
import re
import sys
import time
import unicodedata

import soundfile as sf
from tqdm import tqdm

import xc_download as xc

SPECIES = (
    "Otus scops", "Strix aluco", "Sitta europaea", "Pyrrhula pyrrhula",
    "Cuculus canorus", "Certhia familiaris", "Periparus ater", "Lophophanes cristatus",
    "Parus major", "Poecile montanus", "Cyanistes caeruleus", "Regulus regulus",
    "Regulus ignicapilla", "Erithacus rubecula", "Turdus merula", "Turdus philomelos",
    "Fringilla coelebs", "Loxia curvirostra", "Troglodytes troglodytes",
)


def normalized(text):
    text = unicodedata.normalize("NFKD", text).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", text))


OWL_NAMES = tuple(normalized(name) for name in (
    "Aegolius funereus", "A. funereus", "Boreal Owl", "Tengmalm's Owl",
    "Tengmalms Owl", "Glaucidium passerinum", "G. passerinum", "Eurasian Pygmy Owl",
    "Pygmy Owl",
))


def mentions_owl(text):
    value = " " + normalized(text) + " "
    return any(" " + name + " " in value for name in OWL_NAMES)


def exclusion_reason(recording, species):
    actual = f'{recording.get("gen", "")} {recording.get("sp", "")}'
    if actual.casefold() != species.casefold():
        return "foreground species mismatch"
    background = recording.get("also")
    if not isinstance(background, list) or any(not isinstance(name, str) for name in background):
        return "missing or malformed background metadata"
    if any(mentions_owl(name) for name in background):
        return "target owl listed in background"
    if mentions_owl(str(recording.get("rmk") or "")):
        return "remarks mention a target owl"
    if not recording.get("file"):
        return "audio unavailable"
    if not str(recording.get("id", "")).isdigit():
        return "invalid recording ID"
    return None


def metadata_seconds(value):
    """XC lengths are minutes:seconds or hours:minutes:seconds."""
    try:
        parts = [float(part) for part in str(value).split(":")]
        if not 1 <= len(parts) <= 3 or any(not math.isfinite(x) or x < 0 for x in parts):
            return 0.0
        seconds = 0.0
        for part in parts:
            seconds = seconds * 60 + part
        return seconds
    except (ValueError, TypeError):
        return 0.0


def fetch_recordings(species, key):
    from urllib.parse import urlencode
    genus, epithet = species.split()
    query = f"gen:{genus} sp:{epithet} area:europe"
    seen = set()
    rows = []
    page = 1
    with tqdm(total=None, desc=f"Metadata: {species}", unit="page", position=1,
              leave=False, dynamic_ncols=True) as progress:
        while True:
            params = urlencode({"query": query, "key": key, "page": page, "per_page": 500})
            with xc.open_url(f"{xc.API}?{params}") as response:
                try:
                    data = json.load(response)
                except (ValueError, UnicodeError):
                    raise RuntimeError("API returned invalid JSON") from None
            if not isinstance(data, dict) or not isinstance(data.get("recordings"), list) or "numPages" not in data:
                raise RuntimeError("Unexpected API response; check XC_KEY and API access")
            pages = max(1, int(data["numPages"]))
            progress.total = pages
            for recording in data["recordings"]:
                recording_id = str(recording.get("id", ""))
                if recording_id not in seen:
                    seen.add(recording_id)
                    rows.append(recording)
            progress.update(1)
            if page >= pages:
                break
            page += 1
    return rows


def duration(path):
    """Verify a decodable stream and obtain its duration in seconds."""
    with sf.SoundFile(path) as source:
        seconds = source.frames / source.samplerate
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Invalid audio duration")
        if not len(source.read(min(source.samplerate, source.frames))):
            raise ValueError("Empty audio stream")
    return seconds


def previous_files(manifest):
    results = {}
    if manifest.is_file():
        with manifest.open(encoding="utf-8") as source:
            for line in source:
                try:
                    entry = json.loads(line)
                    if entry.get("status") in {"downloaded", "existing"}:
                        name = entry.get("filename")
                        if isinstance(name, str) and Path(name).name == name:
                            results[str(entry["recording"]["id"])] = name
                except (ValueError, KeyError, TypeError):
                    continue  # An interrupted last line does not lose earlier provenance.
    return results


def choose_next(pending, remaining):
    """Prefer recordings that fit; minimize overshoot for the last recording."""
    for index, recording in enumerate(pending):
        seconds = metadata_seconds(recording.get("length"))
        if 0 < seconds <= remaining:
            return pending.pop(index)
    index = min(range(len(pending)), key=lambda i: metadata_seconds(pending[i].get("length")) or float("inf"))
    return pending.pop(index)


def collect(species, args, key):
    records = fetch_recordings(species, key)
    eligible, rejected = [], []
    for recording in records:
        reason = exclusion_reason(recording, species)
        (rejected if reason else eligible).append((recording, reason) if reason else recording)
    available = sum(metadata_seconds(rec.get("length")) for rec in eligible)
    tqdm.write(f"{species}: {len(eligible)} eligible / {len(records)} records; "
               f"estimated available audio {available / 3600:.2f} h")
    if args.dry_run:
        return {"species": species, "hours": available / 3600, "eligible": len(eligible), "dry_run": True}

    folder = args.output / species.replace(" ", "_")
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "manifest.jsonl"
    old = previous_files(manifest)
    target = args.hours * 3600
    total = 0.0
    successful = 0
    failed = 0
    pending = []
    rng = random.Random(f"{args.seed}:{species}")
    rng.shuffle(eligible)
    if manifest.is_file() and manifest.stat().st_size:
        with manifest.open("rb") as source:
            source.seek(-1, 2)
            needs_newline = source.read(1) != b"\n"
        if needs_newline:
            with manifest.open("ab") as output:
                output.write(b"\n")
    with manifest.open("a", encoding="utf-8") as output:
        def log(recording, status, **extra):
            output.write(json.dumps({"recording": recording, "status": status, **extra}, ensure_ascii=False) + "\n")
            output.flush()
        for recording, reason in rejected:
            log(recording, "excluded", reason=reason)
        for recording in eligible:
            name = old.get(str(recording["id"]))
            if name is not None and (folder / name).is_file():
                try:
                    seconds = duration(folder / name)
                except (OSError, RuntimeError, ValueError):
                    pending.append(recording)
                    continue
                total += seconds
                successful += 1
                log(recording, "existing", filename=name, duration_s=seconds)
            else:
                pending.append(recording)
        existing = successful
        with tqdm(total=args.hours, initial=min(total / 3600, args.hours),
                  desc=species, unit="h audio", position=1, leave=False, dynamic_ncols=True,
                  bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:.2f}/{total:.2f} h audio [{elapsed}<{remaining}] {postfix}") as progress:
            progress.set_postfix(recordings=successful, existing=existing, failed=failed, refresh=False)
            while total < target and pending:
                recording = choose_next(pending, target - total)
                name, status = None, None
                try:
                    name, status = xc.download(recording, folder)
                    seconds = duration(folder / name)
                    total += seconds
                    successful += 1
                    log(recording, status, filename=name, duration_s=seconds)
                    progress.update(min(total / 3600, args.hours) - progress.n)
                except (OSError, RuntimeError, ValueError):
                    failed += 1
                    if status == "downloaded" and name is not None:
                        (folder / name).unlink(missing_ok=True)
                    log(recording, "failed", reason="Download or audio validation failed; rerun to retry")
                    tqdm.write(f"{species}: XC{recording['id']} failed; continuing", file=sys.stderr)
                progress.set_postfix(recordings=successful, existing=existing, failed=failed, refresh=False)
                time.sleep(.2)
    complete = total >= target
    tqdm.write(f"{species}: {'COMPLETE' if complete else 'SHORTFALL'} — {total / 3600:.3f} h, "
               f"{successful} recordings, {failed} failed; {folder}")
    return {"species": species, "hours": total / 3600, "recordings": successful,
            "failed": failed, "complete": complete}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("data/negatives"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--hours", type=float, default=5.)
    parser.add_argument("--species", nargs="+", choices=SPECIES, help="Optional subset, with quoted scientific names")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Show metadata availability without creating or downloading files")
    args = parser.parse_args()
    if not math.isfinite(args.hours) or args.hours <= 0:
        parser.error("--hours must be a positive finite number")
    key = xc.load_key(args.env_file)
    if not key:
        parser.error("Set XC_KEY in .env or the environment")
    summaries = []
    selected = list(dict.fromkeys(args.species or SPECIES))
    try:
        with tqdm(selected, desc="Species", unit="species", position=0, dynamic_ncols=True) as progress:
            for species in progress:
                try:
                    summary = collect(species, args, key)
                except (OSError, RuntimeError, ValueError) as error:
                    # Do not emit raw network exceptions or request URLs containing the key.
                    tqdm.write(f"{species}: failed ({type(error).__name__}: {str(error).replace(key, '[redacted]')}); rerun to retry", file=sys.stderr)
                    summary = {"species": species, "complete": False, "hours": 0, "error": True}
                summaries.append(summary)
        print("\nSpecies totals:")
        for item in summaries:
            label = "estimated eligible" if item.get("dry_run") else ("complete" if item.get("complete") else "shortfall/error")
            print(f"{item['species']}: {item['hours']:.3f} h ({label})")
        if args.dry_run:
            return 1 if any(item.get("error") for item in summaries) else 0
        return 0 if all(item.get("complete") for item in summaries) else 1
    except KeyboardInterrupt:
        print("\nInterrupted. Completed recordings are preserved; run the same command to resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
