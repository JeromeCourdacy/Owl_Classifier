#!/usr/bin/env python3
"""
xc_download.py — Download Xeno-canto recordings for any species and convert them
to mono 16 kHz WAV (the format used by the ESP32 / the model).

The API key is read from a .env file (line  XC_KEY=your_key ), looked up in the
current directory first, then in the script's directory. Never put the key in code.

EXAMPLES
  # 30 Boreal Owl recordings (qualities A and B, the default)
  python xc_download.py "Aegolius funereus" -n 30 -o data/boreal_train_full

  # 10 TEST recordings, excluding recordings (and recordists) already downloaded
  python xc_download.py "Aegolius funereus" -n 10 -o data/boreal_test_full \
         --exclude data/boreal_train_full

  # Rare species: exclude only identical recordings, not identical recordists
  python xc_download.py "Aegolius funereus" -n 10 -o data/boreal_test_full \
         --exclude data/boreal_train_full --allow-same-recordists

  # Eurasian Pygmy Owl, max 2 per recordist (songs, qualities A and B: the defaults)
  python xc_download.py "Glaucidium passerinum" -n 20 --max-per-recordist 2

  # Only the best quality
  python xc_download.py "Aegolius funereus" -n 30 -q A

  # Barn Owl: its screech is a "call", and skip very short recordings (single flight calls)
  python xc_download.py "Tyto alba" -n 8 --type any --min-duration 20 -o data/barn_train_full

  # Calls instead of songs, or every type
  python xc_download.py "Aegolius funereus" --type call
  python xc_download.py "Aegolius funereus" --type any

By default only recordings whose type is exactly "song" are kept: mixed types such as
"call, song" or "aberrant, song" are skipped (add --include-mixed to keep them).
Recordings longer than 4 minutes are skipped too (change with --max-duration SECONDS,
or --max-duration 0 for no limit).

Show all options:  python xc_download.py -h

OUTPUT:
  <output>/XC<id>.wav                    the recordings (mono, 16 kHz)
  <parent>/metadata/<output name>.csv    recordist, country, call type, quality, licence...
                                         e.g. data/metadata/boreal_train_full.csv (cite them in the report)
Re-running the command does not download files that are already there.
Requires: Python 3 and ffmpeg. Nothing else to install.
"""

import argparse
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://xeno-canto.org/api/3/recordings"
UA = {"User-Agent": "boreal-owl-classifier/1.0 (student project)"}
FIELDS = ["id", "gen", "sp", "en", "rec", "cnt", "loc", "date", "type", "q", "length", "lic", "url", "file_name"]


# ---------------------------------------------------------------- API key (.env)
def read_key():
    """Read XC_KEY from the environment, otherwise from a .env file
    (current directory first, then the script's directory). No library needed."""
    for path in (os.path.join(os.getcwd(), ".env"),
                 os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")):
        if os.path.isfile(path):
            for line in open(path, encoding="utf-8"):
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    key = os.getenv("XC_KEY")
    if not key:
        sys.exit("API key not found: create a .env file containing the line  XC_KEY=your_key")
    return key


# ---------------------------------------------------------------- helpers
def get_json(url):
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f"Xeno-canto error ({e.code}): {e.read().decode('utf-8', 'replace')[:400]}")
    except urllib.error.URLError as e:
        sys.exit(f"Cannot reach Xeno-canto: {e.reason}")


def duration_s(text):
    """'1:23' -> 83"""
    try:
        total = 0
        for p in str(text).split(":"):
            total = total * 60 + int(p)
        return total
    except ValueError:
        return None


def type_ok(rec_type, wanted, include_mixed):
    """'song' matches 'song'; 'call, song' matches only with include_mixed; 'any' matches everything."""
    if not wanted or wanted.lower() == "any":
        return True
    types = {t.strip().lower() for t in str(rec_type).split(",") if t.strip()}
    w = wanted.strip().lower()
    return w in types and (include_mixed or types == {w})


def metadata_path(folder):
    """Metadata are kept out of the sound folders, in the nearest existing 'metadata' folder above:
       data/boreal_full/boreal_train_full  ->  data/metadata/boreal_train_full.csv
    If there is none yet, it is created next to the folder's parent."""
    folder = os.path.abspath(os.path.normpath(folder))
    name = os.path.basename(folder)
    parent = os.path.dirname(folder)
    d = parent
    while True:
        if os.path.isdir(os.path.join(d, "metadata")):
            return os.path.join(d, "metadata", name + ".csv")
        up = os.path.dirname(d)
        if up == d or os.path.exists(os.path.join(d, ".git")):   # stop at the project root
            break
        d = up
    return os.path.join(parent, "metadata", name + ".csv")


def already_used(folders):
    """XC ids and recordists already present in other folders (to avoid train/test overlap)."""
    ids, recordists = set(), set()
    for d in folders:
        for f in glob.glob(os.path.join(d, "**", "*.wav"), recursive=True):
            m = re.search(r"XC(\d+)", os.path.basename(f))
            if m:
                ids.add(m.group(1))
        for meta in (metadata_path(d), os.path.join(d, "metadata.csv")):
            if not os.path.isfile(meta):
                continue
            for row in csv.DictReader(open(meta, encoding="utf-8")):
                    ids.add(str(row.get("id", "")))
                    if row.get("rec"):
                        recordists.add(row["rec"])
    return ids, recordists


def download(url, dest):
    req = urllib.request.Request(url, headers=UA)
    tmp = dest + ".part"
    with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    os.replace(tmp, dest)


def to_wav(src, dest):
    cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", src, "-ac", "1", "-ar", "16000", dest]
    return subprocess.run(cmd).returncode == 0


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description="Download Xeno-canto recordings as mono 16 kHz WAV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="EXAMPLES" + __doc__.split("EXAMPLES")[1].split("OUTPUT")[0])
    ap.add_argument("species", help='scientific name in quotes, e.g. "Aegolius funereus"')
    ap.add_argument("-n", "--number", type=int, default=30, help="number of recordings wanted (default 30)")
    ap.add_argument("-o", "--output", help="output folder (default: genus_species)")
    ap.add_argument("-q", "--quality", default="A,B", help="accepted qualities, e.g. A or A,B,C (default A,B)")
    ap.add_argument("--type", default="song",
                    help="keep only recordings of exactly this type (default: song). "
                         "Use --type any to keep every type")
    ap.add_argument("--include-mixed", action="store_true",
                    help="also keep mixed types that contain --type, e.g. 'call, song' or 'aberrant, song'")
    ap.add_argument("--max-per-recordist", type=int, default=3, help="max recordings per recordist (default 3)")
    ap.add_argument("--max-duration", type=int, default=240,
                    help="skip recordings longer than this, in seconds (default 240 = 4 min; 0 = no limit)")
    ap.add_argument("--min-duration", type=int, default=0,
                    help="skip recordings shorter than this, in seconds (default 0 = no minimum)")
    ap.add_argument("--exclude", nargs="+", default=[], metavar="FOLDER",
                    help="skip recordings and recordists already present in these folders")
    ap.add_argument("--allow-same-recordists", action="store_true",
                    help="with --exclude: skip only identical recordings, not identical recordists "
                         "(useful for rare species)")
    args = ap.parse_args()

    parts = args.species.split()
    if len(parts) != 2:
        sys.exit('Give the scientific name as two words, e.g. "Aegolius funereus"')
    genus, species = parts[0].capitalize(), parts[1].lower()
    output = args.output or f"{genus}_{species}"
    qualities = {q.strip().upper() for q in args.quality.split(",") if q.strip()}

    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found: activate the conda environment where it is installed.")
    key = read_key()
    excluded_ids, excluded_recordists = already_used(args.exclude)
    if args.allow_same_recordists:
        excluded_recordists = set()
    print(f"Call type: {args.type}" + (" (including mixed types)" if args.include_mixed else
                                        " (exact)" if args.type.lower() != "any" else "")
          + (f", max duration {args.max_duration // 60}:{args.max_duration % 60:02d}" if args.max_duration
             else ", no duration limit"))
    if args.exclude:
        print(f"Excluding: {len(excluded_ids)} recordings and {len(excluded_recordists)} recordists already used")

    # --- search (all pages needed)
    query = f"gen:{genus} sp:{species}" + (f" q:{next(iter(qualities))}" if len(qualities) == 1 else "")
    selected, per_recordist, page, n_pages = [], {}, 1, 1
    while len(selected) < args.number and page <= n_pages:
        data = get_json(f"{API}?{urllib.parse.urlencode({'query': query, 'key': key, 'page': page})}")
        if "error" in data:
            sys.exit(f"Xeno-canto error: {data.get('message', data)}")
        n_pages = int(data.get("numPages", 1))
        if page == 1:
            print(f"{data.get('numRecordings', '?')} recordings found for '{query}'")
        for r in data.get("recordings", []):
            rec = r.get("rec", "?")
            if (r.get("q", "").upper() not in qualities
                    or str(r.get("id")) in excluded_ids or rec in excluded_recordists
                    or not type_ok(r.get("type", ""), args.type, args.include_mixed)
                    or (args.max_duration and (duration_s(r.get("length")) or 0) > args.max_duration)
                    or (args.min_duration and (duration_s(r.get("length")) or 0) < args.min_duration)
                    or per_recordist.get(rec, 0) >= args.max_per_recordist
                    or not r.get("file")):
                continue
            per_recordist[rec] = per_recordist.get(rec, 0) + 1
            selected.append(r)
            if len(selected) >= args.number:
                break
        page += 1
        time.sleep(1)

    if not selected:
        hint = ("\nTip: with --exclude, recordists already used are excluded too; "
                "add --allow-same-recordists or widen the quality (-q A,B,C).") if excluded_recordists else ""
        sys.exit("No recording matches these criteria." + hint)
    if len(selected) < args.number:
        print(f"Warning: only {len(selected)} recordings match (requested: {args.number}).")
    print(f"{len(selected)} recordings selected, from {len(per_recordist)} different recordists\n")

    # --- download + conversion
    os.makedirs(output, exist_ok=True)
    meta_path = metadata_path(output)
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    rows = {}
    if os.path.isfile(meta_path):
        rows = {row["id"]: row for row in csv.DictReader(open(meta_path, encoding="utf-8"))}
    ok = 0
    for i, r in enumerate(selected, 1):
        xc = str(r["id"])
        wav = os.path.join(output, f"XC{xc}.wav")
        print(f"[{i}/{len(selected)}] XC{xc}  q:{r.get('q')}  {r.get('length')}  {r.get('type')}  ({r.get('rec')})")
        if not os.path.exists(wav):
            mp3 = os.path.join(output, f"XC{xc}.mp3")
            try:
                download(r["file"], mp3)
                if not to_wav(mp3, wav):
                    print("   ! ffmpeg conversion failed")
                    continue
            except Exception as e:
                print(f"   ! failed: {e}")
                continue
            finally:
                if os.path.exists(mp3):
                    os.remove(mp3)
            time.sleep(0.5)
        row = {k: r.get(k, "") for k in FIELDS}
        for k in ("url", "lic"):
            if str(row[k]).startswith("//"):
                row[k] = "https:" + row[k]
        row["file_name"] = os.path.basename(wav)
        rows[xc] = row
        ok += 1

    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows.values())
    print(f"\nDone: {ok} recordings in '{output}' (metadata: {meta_path})")


if __name__ == "__main__":
    main()
