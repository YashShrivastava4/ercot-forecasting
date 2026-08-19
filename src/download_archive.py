"""Downloads ERCOT's official historical load archive files directly into
data/raw/manual_archive/.

These are the same static files anyone can download by hand from
https://www.ercot.com/gridinfo/load/load_hist -- one zip (or xls) per year,
hosted at a fixed URL, no login and no API key. This script tries to fetch
them automatically with a normal browser User-Agent. If ERCOT's WAF blocks
scripted requests, this prints the direct link for you to open in a real
browser instead -- either way you end up with the same files in the same
folder, and everything downstream doesn't care which path got them there.

This only ever needs to run once per year of data. Once the files are
downloaded and data/processed/load_weather_joined.parquet is built and
committed to the repo, nobody -- including the deployed app -- needs to run
this again.
"""

import requests

from src import config

# a normal desktop browser UA -- ERCOT's WAF has been seen blocking requests
# with no UA or an obviously scripted one (e.g. python-requests' default),
# this is the one thing that reliably distinguishes "script" from "browser"
# for a plain GET against a static file, no cookies or JS execution needed
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}


def _already_have(year: int) -> bool:
    """True if some file for this year is already sitting in the archive folder."""
    return any(config.ARCHIVE_DIR.glob(f"*{year}*"))


def download_one_year(year: int, url: str) -> bool:
    """Download a single year's archive file. Returns True on success."""
    config.ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    dest = config.ARCHIVE_DIR / url.split("/")[-1]

    try:
        response = requests.get(url, headers=BROWSER_HEADERS, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  {year}: auto-download failed ({e}). Download by hand instead:")
        print(f"    {url}")
        print(f"    then save the file into {config.ARCHIVE_DIR}")
        return False

    dest.write_bytes(response.content)
    print(f"  {year}: downloaded {dest.name} ({len(response.content) / 1024:.0f} KB)")
    return True


def ensure_archive_files_present() -> None:
    """Make sure every year in the configured range has an archive file, downloading
    whatever's missing. Never raises -- a failed download just gets logged with
    the direct link, and load_manual_archive_files() reports what's actually
    present afterward rather than assuming this step fully succeeded.
    """
    years_needed = range(config.LOAD_START_YEAR, config.LOAD_END_YEAR + 1)
    print(f"checking ERCOT archive files for {years_needed.start}-{years_needed.stop - 1}...")

    for year in years_needed:
        if _already_have(year):
            print(f"  {year}: already have a file, skipping")
            continue

        url = config.ERCOT_ARCHIVE_URLS.get(year)
        if url is None:
            print(f"  {year}: no known URL yet (probably not published, or this project's "
                  f"URL map hasn't been updated for it). Check "
                  f"https://www.ercot.com/gridinfo/load/load_hist for the current link, "
                  f"add it to ERCOT_ARCHIVE_URLS in src/config.py, or just download it by "
                  f"hand into {config.ARCHIVE_DIR}")
            continue

        download_one_year(year, url)


if __name__ == "__main__":
    ensure_archive_files_present()
