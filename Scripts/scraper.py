"""
SubDL subtitle scraper.
Reads movies from Scripts/movies.txt and downloads English SRT files.
Usage: python Scripts/scraper.py
"""

import io
import os
import re
import time
import zipfile
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / "backend/.env")

API_KEY = os.getenv("SUBDL_API_KEY")
DATA_DIR = Path(__file__).parent / "data"
MOVIES_TXT = Path(__file__).parent / "movies.txt"

API_URL = "https://api.subdl.com/api/v1/subtitles"
DL_BASE = "https://dl.subdl.com"
DELAY = 3


def movie_to_filename(name: str, year: int) -> str:
    safe = re.sub(r"[^\w\s]", "", name).strip()
    safe = re.sub(r"\s+", "_", safe)
    return f"{safe}_{year}.srt"


def parse_movie_line(line: str) -> tuple[str, int]:
    line = line.strip()
    match = re.match(r"^(.+?)\s*\((\d{4})\)\s*$", line)
    if not match:
        raise ValueError(f"Cannot parse line: {line!r}")
    return match.group(1).strip(), int(match.group(2))


def fetch_subtitle_url(name: str, year: int) -> str | None:
    params = {
        "api_key": API_KEY,
        "film_name": name,
        "year": year,
        "type": "movie",
        "languages": "EN",
    }
    resp = requests.get(API_URL, params=params, timeout=15)
    resp.raise_for_status()
    subtitles = resp.json().get("subtitles") or []
    if not subtitles:
        return None
    return subtitles[0].get("url")


def download_srt(zip_url: str) -> bytes | None:
    full_url = f"{DL_BASE}{zip_url}" if zip_url.startswith("/") else zip_url
    resp = requests.get(full_url, timeout=30)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        srt_names = [n for n in zf.namelist() if n.lower().endswith(".srt")]
        if not srt_names:
            return None
        return zf.read(srt_names[0])


def main():
    if not API_KEY:
        raise SystemExit("SUBDL_API_KEY not set in backend/.env")

    lines = [l for l in MOVIES_TXT.read_text().splitlines() if l.strip()]
    print(f"Processing {len(lines)} movies...\n")

    ok = skipped = failed = 0

    for line in lines:
        try:
            name, year = parse_movie_line(line)
        except ValueError as e:
            print(f"SKIP  {e}")
            failed += 1
            continue

        out_path = DATA_DIR / movie_to_filename(name, year)
        if out_path.exists():
            print(f"EXISTS  {name} ({year})")
            skipped += 1
            continue

        try:
            zip_url = fetch_subtitle_url(name, year)
            if not zip_url:
                print(f"NOT FOUND  {name} ({year})")
                failed += 1
            else:
                srt_bytes = download_srt(zip_url)
                if not srt_bytes:
                    print(f"NO SRT  {name} ({year})")
                    failed += 1
                else:
                    out_path.write_bytes(srt_bytes)
                    print(f"OK  {name} ({year}) → {out_path.name}")
                    ok += 1
        except Exception as e:
            print(f"ERROR  {name} ({year}) — {e}")
            failed += 1

        time.sleep(DELAY)

    print(f"\nDone — OK:{ok}  SKIPPED:{skipped}  FAILED:{failed}")


if __name__ == "__main__":
    main()
