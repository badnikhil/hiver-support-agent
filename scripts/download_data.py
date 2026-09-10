"""Download twcs.csv into data/raw/. Source: Kaggle public dataset (no credentials needed)."""
import io
import sys
import zipfile
from pathlib import Path

import requests

URL = "https://www.kaggle.com/api/v1/datasets/download/thoughtvector/customer-support-on-twitter"
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def main():
    out = RAW / "twcs.csv"
    if out.exists():
        print(f"{out} exists, skipping")
        return
    RAW.mkdir(parents=True, exist_ok=True)
    zpath = RAW / "twcs.zip"
    if not zpath.exists():
        print("downloading", URL)
        r = requests.get(URL, stream=True, timeout=60)
        r.raise_for_status()
        with open(zpath, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    with zipfile.ZipFile(zpath) as z:
        with z.open("twcs/twcs.csv") as src, open(out, "wb") as dst:
            while chunk := src.read(1 << 20):
                dst.write(chunk)
    print("wrote", out, out.stat().st_size, "bytes")


if __name__ == "__main__":
    sys.exit(main())
