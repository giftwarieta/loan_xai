"""
data_loader.py
===============
Loads the accepted-loan and rejected-application datasets.

Load order:
    1. Local file in DATA_DIR (accepted_2007_to_2018Q4.csv / rejected_2007_to_2018Q4.csv).
    2. Kaggle API download (wordsforthewise/lending-club) if local files are missing.

Usage
-----
    from src.data_loader import load_accepted_data, load_rejected_data
    accepted_df = load_accepted_data()
    rejected_df = load_rejected_data()
"""

import logging
import subprocess
import zipfile
from pathlib import Path

import pandas as pd

from . import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _download_from_kaggle(filename: str, dest_dir: Path) -> Path:
    """Download and extract a target file using the Kaggle CLI."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", config.KAGGLE_DATASET,
                "-f", filename,
                "-p", str(dest_dir),
                "--force",
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Kaggle CLI returned an error:\n{result.stderr.strip()}")
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            f"Kaggle CLI unavailable ({exc}). Ensure `kaggle` is installed and "
            "~/.kaggle/kaggle.json is configured."
        ) from exc

    zip_path = dest_dir / f"{filename}.zip"
    csv_path = dest_dir / filename

    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest_dir)
        zip_path.unlink()

    if not csv_path.exists():
        raise FileNotFoundError(f"Kaggle download reported success, but {csv_path} was not found.")

    logger.info("Successfully fetched and extracted %s", csv_path)
    return csv_path


def load_accepted_data(force_refresh: bool = False) -> pd.DataFrame:
    """Load accepted loan data from data/accepted_2007_to_2018Q4.csv, downloading from Kaggle if missing."""
    if not force_refresh and config.ACCEPTED_PATH.exists():
        logger.info("Loading local accepted-loan data from %s", config.ACCEPTED_PATH)
        return pd.read_csv(config.ACCEPTED_PATH, low_memory=False)

    logger.info("Local file %s not found. Initiating Kaggle download...", config.ACCEPTED_PATH)
    file_path = _download_from_kaggle(config.ACCEPTED_FILENAME, config.DATA_DIR)
    return pd.read_csv(file_path, low_memory=False)


def load_rejected_data(force_refresh: bool = False) -> pd.DataFrame:
    """Load rejected application data from data/rejected_2007_to_2018Q4.csv, downloading from Kaggle if missing."""
    if not force_refresh and config.REJECTED_PATH.exists():
        logger.info("Loading local rejected-application data from %s", config.REJECTED_PATH)
        return pd.read_csv(config.REJECTED_PATH, low_memory=False)

    logger.info("Local file %s not found. Initiating Kaggle download...", config.REJECTED_PATH)
    file_path = _download_from_kaggle(config.REJECTED_FILENAME, config.DATA_DIR)
    return pd.read_csv(file_path, low_memory=False)


if __name__ == "__main__":
    try:
        accepted = load_accepted_data()
        print(f"Accepted data: {accepted.shape}")
    except Exception as e:
        print(f"[Accepted data error] {e}")

    try:
        rejected = load_rejected_data()
        print(f"Rejected data: {rejected.shape}")
    except Exception as e:
        print(f"[Rejected data error] {e}")