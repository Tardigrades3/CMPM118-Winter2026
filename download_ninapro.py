#!/usr/bin/env python3
"""
Download NinaPro EMG datasets.

Usage:
    python download_ninapro.py                   # DB1 only (default)
    python download_ninapro.py --db db2          # DB2 only
    python download_ninapro.py --db db1 db2      # DB1 + DB2
    python download_ninapro.py --db all          # all supported DBs

Supported databases:
    db1  27 subjects, 10 ch, Otto Bock MES,   Exercises 1-3
    db2  40 subjects, 12 ch, Delsys Trigno,   Exercises 1-3  (cross-device validation)
    db5  10 subjects, 16 ch, Myo armband,     Exercises 1-3  (consumer-grade device)
"""
import argparse
import urllib.request
import urllib.error
import zipfile
import os
import sys
import shutil

_S3_BASE = "https://nina-pro-dataset.s3.us-east-2.amazonaws.com"

# Zenodo DOIs for fallback via `zenodo_get`.  Verify current DOIs at:
#   https://ninapro.hevs.ch  →  "Download" section
_DB_CONFIG = {
    'db1': {
        'url': f"{_S3_BASE}/db1/NinaProData.zip",
        'out_dir': 'NinaProData_DB1',
        'zenodo_doi': '10.5281/zenodo.1000116',
    },
    'db2': {
        'url': f"{_S3_BASE}/db2/NinaProData.zip",
        'out_dir': 'NinaProData_DB2',
        'zenodo_doi': '10.5281/zenodo.1876587',
    },
    'db5': {
        'url': f"{_S3_BASE}/db5/NinaProData.zip",
        'out_dir': 'NinaProData_DB5',
        'zenodo_doi': None,
    },
}

ALL_DBS = list(_DB_CONFIG)


def _check_url(url):
    """Return True if the URL responds with HTTP 200."""
    try:
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status == 200
    except Exception:
        return False


def _progress_hook(block_num, block_size, total_size):
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    pct = min(downloaded / total_size * 100, 100)
    bar_len = 40
    filled = int(bar_len * pct / 100)
    bar = '#' * filled + '-' * (bar_len - filled)
    mb_done = downloaded / 1e6
    mb_total = total_size / 1e6
    print(f"\r  [{bar}] {pct:5.1f}%  {mb_done:.1f}/{mb_total:.1f} MB", end='', flush=True)


def download_db(db_key):
    cfg = _DB_CONFIG[db_key]
    out_dir = cfg['out_dir']
    url = cfg['url']
    doi = cfg['zenodo_doi']
    zip_name = f"{db_key}_NinaProData.zip"

    if os.path.exists(out_dir):
        print(f"[{db_key.upper()}] '{out_dir}' already exists — skipping.")
        return True

    print(f"\n[{db_key.upper()}] Checking S3 source...")
    if _check_url(url):
        print(f"[{db_key.upper()}] Downloading from S3...")
        try:
            urllib.request.urlretrieve(url, zip_name, reporthook=_progress_hook)
            print()
        except Exception as e:
            print(f"\n[{db_key.upper()}] S3 download failed: {e}")
            os.path.exists(zip_name) and os.remove(zip_name)
            _print_fallback(db_key, doi)
            return False
    else:
        print(f"[{db_key.upper()}] S3 URL not available.")
        _print_fallback(db_key, doi)
        return False

    print(f"[{db_key.upper()}] Extracting to {out_dir}/...")
    try:
        with zipfile.ZipFile(zip_name, 'r') as zf:
            for member in zf.infolist():
                raw = member.filename
                normalized = raw.replace("\\", "/")
                parts = [p for p in normalized.split("/") if p not in ("", ".", "..")]
                if not parts:
                    continue
                out_path = os.path.join(out_dir, *parts)
                if raw.endswith("/") or raw.endswith("\\"):
                    os.makedirs(out_path, exist_ok=True)
                    continue
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with zf.open(member) as src, open(out_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    except Exception as e:
        print(f"[{db_key.upper()}] Extraction failed: {e}")
        return False
    finally:
        if os.path.exists(zip_name):
            os.remove(zip_name)

    print(f"[{db_key.upper()}] Done → ./{out_dir}/")
    return True


def _print_fallback(db_key, doi):
    print(f"\n  Fallback: install zenodo_get and run:")
    print(f"    pip install zenodo-get")
    if doi:
        print(f"    zenodo_get {doi}")
        print(f"    # then move the extracted folder to ./{_DB_CONFIG[db_key]['out_dir']}/")
    else:
        print(f"    # Look up the {db_key.upper()} DOI at https://ninapro.hevs.ch")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--db', nargs='+', default=['db1'],
                        metavar='DB',
                        help="Which database(s) to download: db1 db2 db5, or 'all' "
                             "(default: db1)")
    args = parser.parse_args()

    requested = ALL_DBS if 'all' in args.db else [d.lower() for d in args.db]

    unknown = [d for d in requested if d not in _DB_CONFIG]
    if unknown:
        print(f"Unknown database(s): {', '.join(unknown)}")
        print(f"Supported: {', '.join(ALL_DBS)}")
        sys.exit(1)

    ok = all(download_db(db) for db in requested)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
