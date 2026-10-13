"""Download CONDA, build the processed train/test CSVs and load them."""
import urllib.request

import pandas as pd

from .config import CONDA_BASE_URL, CONDA_FILES, PROCESSED_DIR, RAW_DIR
from .preprocess import build_frame


def download_raw() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for fname in CONDA_FILES.values():
        dest = RAW_DIR / fname
        if not dest.exists():
            print(f"Downloading {fname} ...")
            urllib.request.urlretrieve(CONDA_BASE_URL + fname, dest)


def build_processed(force: bool = False) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if not force and all((PROCESSED_DIR / f"{s}.csv").exists() for s in CONDA_FILES):
        return
    download_raw()
    for split, fname in CONDA_FILES.items():
        raw = pd.read_csv(RAW_DIR / fname)
        build_frame(raw).to_csv(PROCESSED_DIR / f"{split}.csv", index=False)


def load_split(split: str) -> pd.DataFrame:
    build_processed()
    return pd.read_csv(PROCESSED_DIR / f"{split}.csv").dropna()


if __name__ == "__main__":
    build_processed(force=True)
    for s in CONDA_FILES:
        d = load_split(s)
        print(f"\n{s}: {len(d)} messages\n{d['label'].value_counts().to_string()}")
