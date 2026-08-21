"""Downloads ERCOT's official per-year load archive files into
data/raw/manual_archive/. If a download gets blocked, it prints the direct
link so the file can be grabbed by hand instead.
"""

import requests

from src import config

# a normal browser user-agent, ERCOT's server can block obviously scripted requests
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
    """Download whatever archive years are missing. Never raises -- a failed
    download just gets logged with the direct link instead.
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
