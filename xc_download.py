#!/usr/bin/env python3
"""Download European foreground owl recordings from xeno-canto API v3."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

API = "https://xeno-canto.org/api/3/recordings"
SPECIES = {
    "pygmy": ("Glaucidium passerinum", "Aegolius funereus"),
    "boreal": ("Aegolius funereus", "Glaucidium passerinum"),
}
ALIASES = {
    "Aegolius funereus": ("aegolius funereus", "boreal owl", "tengmalm's owl", "tengmalm’s owl"),
    "Glaucidium passerinum": ("glaucidium passerinum", "eurasian pygmy owl"),
}


def load_key(env_file):
    """Read XC_KEY from the environment or a simple .env file without executing it."""
    if os.environ.get("XC_KEY"):
        return os.environ["XC_KEY"]
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            name, sep, value = line.strip().partition("=")
            if sep and name.strip() == "XC_KEY":
                return value.strip().strip("\"'")
    return None


def open_url(url):
    """Retry transient failures; never expose URLs containing API credentials."""
    for attempt in range(4):
        try:
            return urlopen(Request(url, headers={"User-Agent": "OwlDatasetDownloader/1.0"}), timeout=90)
        except HTTPError as exc:
            code = exc.code
            exc.close()
            if code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise RuntimeError(f"HTTP {code} from xeno-canto") from None
        except (URLError, TimeoutError, OSError):
            if attempt == 3:
                raise RuntimeError("Network request failed after four attempts") from None
        time.sleep(2 ** attempt)


def recordings(species, key):
    # gen/sp constrain the foreground taxon; also: would search background taxa.
    genus, epithet = species.split()
    query = f'gen:{genus} sp:{epithet} area:europe'
    page = 1
    while True:
        params = urlencode({"query": query, "key": key, "page": page})
        with open_url(f"{API}?{params}") as response:
            try:
                data = json.load(response)
            except (ValueError, UnicodeError):
                raise RuntimeError("API returned invalid JSON") from None
        if not isinstance(data, dict) or "recordings" not in data or "numPages" not in data:
            raise RuntimeError("Unexpected API response; check your XC_KEY and API access")
        if not isinstance(data["recordings"], list):
            raise RuntimeError("API recordings field is not a list")
        yield from data["recordings"]
        if page >= int(data["numPages"]):
            break
        page += 1


def exclusion_reason(recording, foreground, forbidden):
    actual = f'{recording.get("gen", "")} {recording.get("sp", "")}'
    if actual.casefold() != foreground.casefold():
        return "foreground species mismatch"
    background = recording.get("also")
    if not isinstance(background, list) or any(not isinstance(x, str) for x in background):
        return "missing or invalid background metadata"
    if any(alias in name.casefold() for name in background for alias in ALIASES[forbidden]):
        return "other target owl in background"
    if not recording.get("file"):
        return "audio unavailable"
    return None


def download(recording, folder):
    recording_id = str(recording["id"])
    if not recording_id.isdigit():
        raise RuntimeError("Invalid recording ID")
    suffix = Path(str(recording.get("file-name") or "")).suffix.lower()
    if not re.fullmatch(r"\.[a-z0-9]{1,8}", suffix):
        suffix = Path(urlsplit(recording["file"]).path).suffix.lower()
    if not re.fullmatch(r"\.[a-z0-9]{1,8}", suffix):
        suffix = ".audio"
    target = folder / f"XC{recording_id}{suffix}"
    if target.is_file() and target.stat().st_size:
        return target.name, "existing"
    url = recording["file"]
    if url.startswith("//"):
        url = "https:" + url
    if urlsplit(url).scheme != "https":
        raise RuntimeError("Expected an HTTPS audio URL")
    partial = target.with_suffix(target.suffix + ".part")
    try:
        with open_url(url) as response, partial.open("wb") as output:
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/" in content_type or "json" in content_type:
                raise RuntimeError("Audio endpoint returned text instead of audio")
            shutil.copyfileobj(response, output)
            expected = response.headers.get("Content-Length")
        if not partial.stat().st_size or (expected and partial.stat().st_size != int(expected)):
            raise RuntimeError("Empty or incomplete audio download")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target.name, "downloaded"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data"), help="Parent of pygmy/ and boreal/ (default: data)")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--dry-run", action="store_true", help="Fetch and filter metadata without downloading audio")
    args = parser.parse_args()
    key = load_key(args.env_file)
    if not key:
        parser.error("Set XC_KEY in your environment or .env file")
    failures = 0
    for label, (foreground, forbidden) in SPECIES.items():
        folder = args.output / label
        folder.mkdir(parents=True, exist_ok=True)
        counts = {"eligible": 0, "excluded": 0, "downloaded": 0, "existing": 0, "failed": 0}
        seen = set()
        manifest_name = "preview.jsonl" if args.dry_run else "manifest.jsonl"
        # Append preserves provenance of earlier downloads if interrupted or API results change.
        with (folder / manifest_name).open("a", encoding="utf-8") as manifest:
            for rec in recordings(foreground, key):
                rid = str(rec.get("id", ""))
                if rid in seen:
                    continue
                seen.add(rid)
                reason = exclusion_reason(rec, foreground, forbidden)
                entry = {"recording": rec, "status": "excluded" if reason else "eligible", "reason": reason}
                if reason:
                    counts["excluded"] += 1
                else:
                    counts["eligible"] += 1
                    if not args.dry_run:
                        try:
                            filename, status = download(rec, folder)
                            entry.update(status=status, filename=filename)
                            counts[status] += 1
                        except (RuntimeError, OSError, ValueError):
                            entry.update(status="failed", reason="Audio download failed; rerun to retry")
                            counts["failed"] += 1
                            failures += 1
                            print(f"{label}: XC{rid} download failed", file=sys.stderr)
                manifest.write(json.dumps(entry, ensure_ascii=False) + "\n")
                manifest.flush()
        print(f"{label}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
